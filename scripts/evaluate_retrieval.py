import argparse
import json
from pathlib import Path
from typing import Any

from app.agent import ProductionAgent
from app.config import get_settings


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid RAG retrieval using source-labeled gold queries."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.k < 1:
        parser.error("--k must be at least 1")
    examples = [
        json.loads(line)
        for line in arguments.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    agent = ProductionAgent(get_settings())
    try:
        retrieved = [
            [
                str(document.metadata.get("source", "unknown"))
                for document in agent._retrieve_documents([example["query"]])
            ]
            for example in examples
        ]
    finally:
        agent.close()
    print(json.dumps(score_retrieval(examples, retrieved, arguments.k), indent=2))


if __name__ == "__main__":
    main()
