"""Score hybrid retrieval offline, or as a comparable LangSmith experiment."""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.utils import LangSmithConflictError

from app.agent import ProductionAgent
from app.config import Settings, configure_langsmith_tracing, get_settings

GOLD_DATASET_SUFFIX = "retrieval-gold"


def score_retrieval(
    examples: list[dict[str, Any]],
    retrieved: list[list[str]],
    k: int,
) -> dict[str, float]:
    if not examples or len(examples) != len(retrieved):
        raise ValueError("Examples and retrieval results must have equal non-zero length.")
    hits = 0
    reciprocal_rank_sum = 0.0
    for example, sources in zip(examples, retrieved, strict=True):
        expected = set(example["expected_sources"])
        relevant_rank = next(
            (rank for rank, source in enumerate(sources[:k], 1) if source in expected),
            None,
        )
        if relevant_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / relevant_rank
    count = len(examples)
    return {
        f"hit_rate@{k}": hits / count,
        f"mrr@{k}": reciprocal_rank_sum / count,
        "queries": float(count),
    }


def gold_dataset_name(settings: Settings) -> str:
    """Return the LangSmith dataset that holds the repo's gold queries."""
    return f"{settings.effective_langsmith_project}-{GOLD_DATASET_SUFFIX}"


def gold_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map gold queries to dataset rows with stable IDs so syncs update in place."""
    return [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rag-chain-gold:{example['query']}")),
            "inputs": {"query": example["query"]},
            "reference_outputs": {"expected_sources": list(example["expected_sources"])},
        }
        for example in examples
    ]


def sync_gold_dataset(
    client: Client,
    dataset_name: str,
    examples: list[dict[str, Any]],
) -> None:
    """Create the gold dataset on first run and refresh its rows from the repo file."""
    try:
        client.create_dataset(
            dataset_name,
            description="Gold retrieval queries from evaluation/gold_queries.jsonl.",
        )
    except LangSmithConflictError:
        pass
    client.create_examples(
        dataset_name=dataset_name,
        examples=gold_examples(examples),
    )


def _relevance_evaluator(k: int):
    """Return a per-query evaluator that LangSmith averages into the experiment view."""

    def score_sources(
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, float]:
        sources = list(outputs.get("sources") or [])
        expected = set(reference_outputs.get("expected_sources") or [])
        rank = next(
            (position for position, source in enumerate(sources[:k], 1) if source in expected),
            None,
        )
        return {
            f"hit@{k}": 1.0 if rank else 0.0,
            f"mrr@{k}": 1.0 / rank if rank else 0.0,
        }

    return score_sources


def _retrieval_target(agent: ProductionAgent):
    """Return the experiment target: one hybrid retrieval call per gold query."""

    def retrieve(inputs: dict[str, Any]) -> dict[str, list[str]]:
        documents = agent.retrieve(str(inputs.get("query", "")))
        return {
            "sources": [
                str(document.metadata.get("source", "unknown")) for document in documents
            ]
        }

    return retrieve


def score_experiment_rows(rows: list[Any], k: int) -> dict[str, float]:
    """Score the retrieval sources that the LangSmith experiment run recorded."""
    examples = [
        {
            "expected_sources": list((row.example.outputs or {}).get("expected_sources") or [])
        }
        for row in rows
    ]
    retrieved = [list((row.run.outputs or {}).get("sources") or []) for row in rows]
    if not examples:
        return {}
    return score_retrieval(examples, retrieved, k)


def run_langsmith_evaluation(
    agent: ProductionAgent,
    examples: list[dict[str, Any]],
    k: int,
    settings: Settings,
    client: Client | None = None,
) -> tuple[dict[str, float], str | None]:
    """Upload the gold queries, run them as an experiment, and return scored metrics."""
    langsmith_client = client or Client()
    dataset_name = gold_dataset_name(settings)
    sync_gold_dataset(langsmith_client, dataset_name, examples)
    results = evaluate(
        _retrieval_target(agent),
        data=dataset_name,
        evaluators=[_relevance_evaluator(k)],
        metadata={
            "environment": settings.app_env,
            "indexing_version": settings.indexing_version,
            "top_k": k,
        },
        experiment_prefix=f"{settings.effective_langsmith_project}-{GOLD_DATASET_SUFFIX}",
        description=f"Hybrid retrieval quality at k={k}.",
        client=langsmith_client,
    )
    return score_experiment_rows(list(results), k), results.comparison_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid RAG retrieval using source-labeled gold queries."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Run the queries as a LangSmith experiment for cross-run comparison.",
    )
    arguments = parser.parse_args()
    if arguments.k < 1:
        parser.error("--k must be at least 1")
    examples = [
        json.loads(line)
        for line in arguments.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = get_settings()
    agent = ProductionAgent(settings)
    try:
        if arguments.langsmith:
            if not settings.langsmith_tracing:
                parser.error(
                    "--langsmith requires LANGSMITH_TRACING=true and LANGSMITH_API_KEY"
                )
            configure_langsmith_tracing(settings)
            metrics, experiment_url = run_langsmith_evaluation(
                agent, examples, arguments.k, settings
            )
            print(
                json.dumps(
                    {"metrics": metrics, "experiment_url": experiment_url},
                    indent=2,
                )
            )
            return
        retrieved = [
            [
                str(document.metadata.get("source", "unknown"))
                for document in agent.retrieve(example["query"])
            ]
            for example in examples
        ]
    finally:
        agent.close()
    print(json.dumps(score_retrieval(examples, retrieved, arguments.k), indent=2))


if __name__ == "__main__":
    main()
