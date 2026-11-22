"""Coverage for the LangSmith trace shape the application emits."""

from contextlib import nullcontext
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pydantic import SecretStr

from app import agent as agent_module
from app import main as main_module
from app.agent import (
    ProductionAgent,
    _trace_ingestion_inputs,
    _trace_step_inputs,
)
from app.api import documents as document_api
from app.config import (
    Settings,
    configure_langsmith_tracing,
    flush_langsmith_traces,
)
from app.ingestion import tasks as task_module
from app.ingestion.tasks import _trace_task_inputs
from app.models import DocumentInput
from tests.test_agent import make_agent


def traced_settings(**overrides) -> Settings:
    """Build settings that turn tracing on without touching a real project."""
    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("test-langsmith-key"),
        **overrides,
    )
    configure_langsmith_tracing(settings)
    return settings


def recorded_names(runs: list[dict]) -> set[str]:
    return {run["name"] for run in runs if run.get("name")}


def recorded_run(runs: list[dict], name: str) -> dict:
    return next(run for run in runs if run.get("name") == name)


def test_agent_traces_a_named_span_for_every_rag_step(recorded_traces):
    traced_settings()
    agent = make_agent()
    agent.query_expansion_enabled = True
    agent.query_rewriter.invoke.return_value = "Leave days per year?"
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="Employees receive 20 days.",
            metadata={"source": "handbook/leave-policy"},
        )
    ]
    agent.primary_chain.invoke.return_value = "Employees receive 20 days [1]."

    result = agent.invoke("How many leave days?")
    flush_langsmith_traces()

    assert result["grounding_status"] == "supported"
    assert {
        "rag.invoke",
        "rag.query_expansion",
        "rag.retrieve",
        "rag.grounding",
    } <= recorded_names(recorded_traces["runs"])
    traced_inputs = {
        run["name"]: run["inputs"]
        for run in recorded_traces["runs"]
        if run.get("name") and run.get("inputs") is not None
    }
    assert traced_inputs["rag.query_expansion"] == {
        "message": "How many leave days?"
    }
    assert all("self" not in inputs for inputs in traced_inputs.values())


def test_agent_scores_grounding_and_citations_as_trace_feedback(recorded_traces):
    traced_settings()
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="Employees receive 20 days.",
            metadata={"source": "handbook/leave-policy"},
        )
    ]
    agent.primary_chain.invoke.return_value = "Employees receive 20 days [1]."

    agent.invoke("How many leave days?")
    flush_langsmith_traces()

    scores = {entry["key"]: entry for entry in recorded_traces["feedback"]}
    assert scores["grounded"]["score"] == 1
    assert scores["grounded"]["value"] == "supported"
    assert scores["cited"]["score"] == 1
    assert scores["cited"]["value"] == 1
    assert {entry["trace_id"] for entry in recorded_traces["feedback"]} == {
        scores["grounded"]["trace_id"]
    }


def test_unsupported_answers_score_zero_on_the_trace(recorded_traces):
    traced_settings()
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="The policy grants 20 days.",
            metadata={"source": "guide"},
        )
    ]
    agent.primary_chain.invoke.return_value = "The policy grants 30 days [1]."
    agent.grounding_chain.invoke.return_value = "DRIFT"
    agent.general_chain.invoke.return_value = "A general answer."

    result = agent.invoke("How many days?")
    flush_langsmith_traces()

    assert result["grounding_status"] == "potential_drift"
    scores = {entry["key"]: entry["score"] for entry in recorded_traces["feedback"]}
    assert scores["grounded"] == 0
    assert scores["cited"] == 0


def test_agent_maps_quality_signals_to_feedback(monkeypatch) -> None:
    recorded: list[dict] = []
    monkeypatch.setattr(
        agent_module,
        "get_current_run_tree",
        lambda: SimpleNamespace(trace_id="trace-1"),
    )
    monkeypatch.setattr(
        agent_module,
        "record_langsmith_feedback",
        lambda **kwargs: recorded.append(kwargs),
    )

    ProductionAgent._record_trace_feedback(["guide", "faq"], "supported")
    ProductionAgent._record_trace_feedback([], "potential_drift")

    assert recorded == [
        {
            "trace_id": "trace-1",
            "key": "grounded",
            "value": "supported",
            "score": 1,
        },
        {"trace_id": "trace-1", "key": "cited", "value": 2, "score": 1},
        {
            "trace_id": "trace-1",
            "key": "grounded",
            "value": "potential_drift",
            "score": 0,
        },
        {"trace_id": "trace-1", "key": "cited", "value": 0, "score": 0},
    ]


def test_feedback_is_skipped_while_no_trace_is_active(monkeypatch) -> None:
    recorded: list[dict] = []
    monkeypatch.setattr(agent_module, "get_current_run_tree", lambda: None)
    monkeypatch.setattr(
        agent_module,
        "record_langsmith_feedback",
        lambda **kwargs: recorded.append(kwargs),
    )

    ProductionAgent._record_trace_feedback(["guide"], "supported")

    assert recorded == []


def test_traced_inputs_keep_objects_the_team_cannot_read() -> None:
    agent = make_agent()
    assert _trace_step_inputs(
        {"self": agent, "message": "Question?"}
    ) == {"message": "Question?"}
    assert _trace_task_inputs(
        {
            "task": task_module.process_document,
            "document_id": "document-1",
        }
    ) == {"document_id": "document-1"}
    assert _trace_ingestion_inputs(
        {
            "documents": [
                DocumentInput(source="guide", content="Secret full text."),
                DocumentInput(source="faq", content="Another secret."),
            ]
        }
    ) == {"document_count": 2, "sources": ["faq", "guide"]}


def test_worker_nests_indexing_under_the_trace_that_queued_it(
    monkeypatch,
) -> None:
    opened_contexts: list[dict] = []

    def capture_context(**kwargs):
        opened_contexts.append(kwargs)
        return nullcontext()

    indexed: list[tuple[object, str]] = []
    monkeypatch.setattr(task_module, "tracing_context", capture_context)
    monkeypatch.setattr(
        task_module,
        "_index_document",
        lambda task, document_id: (
            indexed.append((task, document_id))
            or {"document_id": document_id, "status": "completed"}
        ),
    )

    result = task_module.process_document(
        "document-1",
        langsmith_parent_trace="20261121T0000000000000000-parent",
    )

    assert opened_contexts == [{"parent": "20261121T0000000000000000-parent"}]
    assert indexed == [(task_module.process_document, "document-1")]
    assert result == {"document_id": "document-1", "status": "completed"}


def test_worker_still_indexes_without_a_parent_trace(monkeypatch) -> None:
    opened_contexts: list[dict] = []

    def capture_context(**kwargs):
        opened_contexts.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(task_module, "tracing_context", capture_context)
    monkeypatch.setattr(
        task_module,
        "_index_document",
        lambda _task, document_id: {"document_id": document_id, "status": "stale"},
    )

    result = task_module.process_document("document-1")

    assert opened_contexts == []
    assert result == {"document_id": "document-1", "status": "stale"}


def test_worker_indexes_as_a_named_span_without_task_objects(
    recorded_traces,
    monkeypatch,
) -> None:
    traced_settings()
    monkeypatch.setattr(
        task_module,
        "create_database_engine",
        lambda _settings: SimpleNamespace(dispose=lambda: None),
    )
    monkeypatch.setattr(
        task_module,
        "DocumentRepository",
        lambda _engine, **_kwargs: SimpleNamespace(get=lambda _document_id: None),
    )

    result = task_module._index_document(
        task_module.process_document,
        "document-1",
    )
    flush_langsmith_traces()

    assert result == {"document_id": "document-1", "status": "stale"}
    span = recorded_run(recorded_traces["runs"], "document.index")
    assert span["inputs"] == {"document_id": "document-1"}
    assert span["tags"] == ["worker", "ingestion"]
    assert span["extra"]["metadata"]["indexing_version"] == (
        task_module.settings.indexing_version
    )


def test_dispatch_links_queued_jobs_to_the_request_trace(
    recorded_traces,
    monkeypatch,
) -> None:
    traced_settings()
    published: list[dict] = []
    repository = SimpleNamespace(
        dispatch_pending=lambda publish: sum(
            1
            for document_id in ("document-1", "document-2")
            if publish(
                args=[document_id],
                task_id=document_id,
                retry=False,
            )
        )
    )
    queued_task = SimpleNamespace(
        apply_async=lambda **kwargs: published.append(kwargs) or True
    )
    monkeypatch.setattr(document_api, "process_document", queued_task)
    monkeypatch.setattr(
        document_api,
        "langsmith_parent_trace",
        lambda: "20261121T0000000000000000-parent",
    )

    dispatched = document_api._dispatch_queued_documents(repository, "document-1")
    flush_langsmith_traces()

    assert dispatched == 2
    assert [entry["kwargs"]["langsmith_parent_trace"] for entry in published] == [
        "20261121T0000000000000000-parent",
        "20261121T0000000000000000-parent",
    ]
    span = recorded_run(recorded_traces["runs"], "document.dispatch")
    assert span["inputs"] == {"document_id": "document-1"}
    assert span["tags"] == ["api", "ingestion"]


def test_dispatch_sends_no_parent_trace_when_tracing_is_off(monkeypatch) -> None:
    published: list[dict] = []
    repository = SimpleNamespace(
        dispatch_pending=lambda publish: int(
            bool(
                publish(
                    args=["document-1"],
                    task_id="document-1",
                    retry=False,
                )
            )
        )
    )
    queued_task = SimpleNamespace(
        apply_async=lambda **kwargs: published.append(kwargs) or True
    )
    monkeypatch.setattr(document_api, "process_document", queued_task)
    configure_langsmith_tracing(Settings(_env_file=None))

    assert document_api._dispatch_queued_documents(repository, "document-1") == 1
    assert published[0]["args"] == ["document-1"]
    assert "kwargs" not in published[0]


def test_ingestion_traces_a_summary_instead_of_document_text(
    recorded_traces,
) -> None:
    traced_settings()
    agent = make_agent()
    agent.text_splitter = SimpleNamespace(
        split_documents=lambda documents: documents
    )

    agent.ingest_documents(
        [DocumentInput(source="handbook", content="Employees receive 20 days.")]
    )
    flush_langsmith_traces()

    span = recorded_run(recorded_traces["runs"], "document.ingest")
    assert span["inputs"] == {"document_count": 1, "sources": ["handbook"]}


def test_worker_process_installs_the_same_langsmith_policy_as_the_api(
    monkeypatch,
) -> None:
    configured: list[Settings] = []
    monkeypatch.setattr(
        task_module,
        "configure_langsmith_tracing",
        lambda settings: configured.append(settings),
    )

    task_module.configure_worker_tracing()

    assert configured == [task_module.settings]


def test_worker_process_flushes_traces_before_it_exits(monkeypatch) -> None:
    flushes: list[float] = []
    monkeypatch.setattr(
        task_module,
        "flush_langsmith_traces",
        lambda timeout=5.0: flushes.append(timeout),
    )

    task_module.flush_worker_traces()

    assert flushes == [5.0]


def agent_factory(events: list[str], flushes: list[float]):
    """Build the stand-in agent class the app lifespan can start and stop."""

    class RecordingAgent:
        def __init__(self, _settings: Settings) -> None:
            events.append("start")

        def close(self) -> None:
            events.append("close")

    return RecordingAgent


def test_api_flushes_traces_before_closing_the_agent(monkeypatch) -> None:
    events: list[str] = []
    flushes: list[float] = []
    monkeypatch.setattr(
        main_module,
        "flush_langsmith_traces",
        lambda timeout=5.0: flushes.append(timeout) or True,
    )
    settings = Settings(_env_file=None)
    settings.database_url = None
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "ProductionAgent",
        agent_factory(events, flushes),
    )

    with TestClient(main_module.app):
        assert events == ["start"]
        assert flushes == []

    assert events == ["start", "close"]
    assert flushes == [5.0]
