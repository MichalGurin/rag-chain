"""Coverage for the LangSmith retrieval experiment run by the eval script."""

import json
import sys
from types import SimpleNamespace

from langchain_core.documents import Document
from langsmith.utils import LangSmithConflictError
import pytest

from app.config import Settings
from scripts import evaluate_retrieval
from scripts.evaluate_retrieval import (
    _relevance_evaluator,
    _retrieval_target,
    gold_dataset_name,
    gold_examples,
    run_langsmith_evaluation,
    score_experiment_rows,
    sync_gold_dataset,
)
from tests.test_agent import make_agent

GOLD = [
    {"query": "How many leave days?", "expected_sources": ["handbook/leave-policy"]},
    {"query": "How do I report an incident?", "expected_sources": ["security/incident"]},
]


class RecordingClient:
    """Stand in for the LangSmith API calls the experiment path makes."""

    def __init__(self):
        self.datasets: list[tuple[str, str]] = []
        self.example_batches: list[tuple[str, list[dict]]] = []

    def create_dataset(self, name: str, description: str = "") -> None:
        self.datasets.append((name, description))

    def create_examples(self, *, dataset_name: str, examples: list[dict]) -> None:
        self.example_batches.append((dataset_name, examples))


class ConflictClient(RecordingClient):
    def create_dataset(self, name: str, description: str = "") -> None:
        raise LangSmithConflictError(f"{name} already exists")


def test_gold_queries_map_to_dataset_rows_with_stable_ids() -> None:
    first = gold_examples(GOLD)
    second = gold_examples(GOLD)

    assert first[0]["inputs"] == {"query": "How many leave days?"}
    assert first[0]["reference_outputs"] == {
        "expected_sources": ["handbook/leave-policy"]
    }
    # Stable IDs make every sync update the same row instead of duplicating it.
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_gold_dataset_lives_next_to_the_project_it_traces(monkeypatch) -> None:
    settings = Settings(_env_file=None, langsmith_project="team-rag")

    assert gold_dataset_name(settings) == "team-rag-retrieval-gold"
    assert gold_dataset_name(Settings(_env_file=None)) == (
        "prod-rag-development-retrieval-gold"
    )


def test_gold_dataset_sync_uploads_the_repo_queries() -> None:
    client = RecordingClient()

    sync_gold_dataset(client, "prod-rag-development-retrieval-gold", GOLD)

    assert client.datasets[0][0] == "prod-rag-development-retrieval-gold"
    assert "evaluation/gold_queries.jsonl" in client.datasets[0][1]
    dataset_name, rows = client.example_batches[0]
    assert dataset_name == "prod-rag-development-retrieval-gold"
    assert [row["inputs"]["query"] for row in rows] == [
        example["query"] for example in GOLD
    ]


def test_gold_dataset_sync_reuses_an_existing_dataset() -> None:
    client = ConflictClient()

    sync_gold_dataset(client, "prod-rag-development-retrieval-gold", GOLD)

    assert [name for name, _ in client.example_batches] == [
        "prod-rag-development-retrieval-gold"
    ]


def test_experiment_evaluator_scores_every_query_for_the_experiment_view() -> None:
    score_sources = _relevance_evaluator(3)

    assert score_sources(
        {"sources": ["x", "handbook/leave-policy"]},
        {"expected_sources": ["handbook/leave-policy"]},
    ) == {"hit@3": 1.0, "mrr@3": 0.5}
    assert score_sources(
        {"sources": ["x", "y"]},
        {"expected_sources": ["handbook/leave-policy"]},
    ) == {"hit@3": 0.0, "mrr@3": 0.0}


def test_experiment_target_runs_hybrid_retrieval_per_query() -> None:
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="Employees receive 20 days.",
            metadata={"source": "handbook/leave-policy"},
        )
    ]

    outputs = _retrieval_target(agent)({"query": "How many leave days?"})

    assert outputs == {"sources": ["handbook/leave-policy"]}


def test_experiment_metrics_come_from_the_recorded_run_sources() -> None:
    rows = [
        SimpleNamespace(
            example=SimpleNamespace(outputs={"expected_sources": ["guide"]}),
            run=SimpleNamespace(outputs={"sources": ["x", "guide"]}),
        ),
        SimpleNamespace(
            example=SimpleNamespace(outputs={"expected_sources": ["guide"]}),
            run=SimpleNamespace(outputs={"sources": ["x", "y"]}),
        ),
    ]

    assert score_experiment_rows(rows, k=2) == {
        "hit_rate@2": 0.5,
        "mrr@2": 0.25,
        "queries": 2.0,
    }


class FakeResults:
    """Stand in for the experiment rows LangSmith returns after a run."""

    def __init__(self, rows: list, comparison_url: str):
        self.rows = rows
        self.comparison_url = comparison_url

    def __iter__(self):
        return iter(self.rows)


def test_langsmith_experiment_syncs_scores_and_returns_the_comparison_url(
    monkeypatch,
) -> None:
    settings = Settings(_env_file=None)
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="Employees receive 20 days.",
            metadata={"source": "handbook/leave-policy"},
        )
    ]
    client = RecordingClient()
    rows = [
        SimpleNamespace(
            example=SimpleNamespace(
                outputs={"expected_sources": ["handbook/leave-policy"]}
            ),
            run=SimpleNamespace(outputs={"sources": ["handbook/leave-policy"]}),
        )
    ]

    captured: dict = {}

    def fake_evaluate(target, **kwargs):
        assert target({"query": "How many leave days?"}) == {
            "sources": ["handbook/leave-policy"]
        }
        captured.update(kwargs)
        return FakeResults(rows, "https://smith.langchain.com/compare?session=exp-1")

    monkeypatch.setattr(evaluate_retrieval, "evaluate", fake_evaluate)

    metrics, comparison_url = run_langsmith_evaluation(
        agent,
        GOLD,
        k=1,
        settings=settings,
        client=client,
    )

    assert metrics == {"hit_rate@1": 1.0, "mrr@1": 1.0, "queries": 1.0}
    assert comparison_url == "https://smith.langchain.com/compare?session=exp-1"
    assert captured["data"] == "prod-rag-development-retrieval-gold"
    assert captured["experiment_prefix"] == "prod-rag-development-retrieval-gold"
    assert captured["metadata"]["indexing_version"] == settings.indexing_version
    assert captured["metadata"]["top_k"] == 1
    assert [name for name, _ in client.example_batches] == [captured["data"]]


def read_gold(dataset: str) -> list[dict]:
    return [json.loads(line) for line in dataset.splitlines() if line.strip()]


def test_script_reports_offline_retrieval_metrics(monkeypatch, tmp_path, capsys) -> None:
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(example) for example in GOLD) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_retrieval.py", "--dataset", str(dataset), "--k", "2"],
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "get_settings",
        lambda: Settings(_env_file=None),
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "ProductionAgent",
        lambda _settings: SimpleNamespace(
            retrieve=lambda _query: [
                Document(
                    page_content="Employees receive 20 days.",
                    metadata={"source": "handbook/leave-policy"},
                )
            ],
            close=lambda: None,
        ),
    )

    evaluate_retrieval.main()

    reported = json.loads(capsys.readouterr().out)
    assert reported == {"hit_rate@2": 0.5, "mrr@2": 0.5, "queries": 2.0}


def test_script_refuses_langsmith_experiments_without_tracing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        json.dumps(GOLD[0]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_retrieval.py", "--dataset", str(dataset), "--langsmith"],
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "get_settings",
        lambda: Settings(_env_file=None),
    )
    monkeypatch.setattr(
        evaluate_retrieval,
        "ProductionAgent",
        lambda _settings: SimpleNamespace(close=lambda: None),
    )

    with pytest.raises(SystemExit):
        evaluate_retrieval.main()

    assert "--langsmith requires LANGSMITH_TRACING" in capsys.readouterr().err


def test_gold_queries_file_matches_the_experiment_contract() -> None:
    examples = read_gold(
        evaluate_retrieval.Path("evaluation/gold_queries.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert examples
    assert all(
        example["query"].strip() and example["expected_sources"]
        for example in examples
    )
