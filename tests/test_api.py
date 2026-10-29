from fastapi.testclient import TestClient
import pytest
from pydantic import SecretStr
from langchain_core.documents import Document

from app import main
from app.agent import ProductionAgent
from app.config import Settings

pytestmark = pytest.mark.integration


class FakeAgent:
    def __init__(self, _settings):
        self.queries: list[str] = []
        self.documents: list[object] = []

    def database_ready(self) -> bool:
        return True

    def invoke(self, message: str) -> dict[str, object]:
        self.queries.append(message)
        return {
            "response": "Employees receive 20 days [1].",
            "model_used": "primary",
            "sources": ["handbook/leave-policy"],
        }

    def ingest_documents(self, documents):
        self.documents.extend(documents)
        return len(documents) * 2


@pytest.fixture
def api_client(monkeypatch):
    settings = Settings(
        _env_file=None,
        ingest_api_key=SecretStr("ingest-secret"),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)
    with TestClient(main.app) as client:
        yield client


def test_chat_returns_sources_and_caches_response(api_client):
    first = api_client.post(
        "/chat",
        json={"message": "How many leave days?", "thread_id": "test-thread"},
    )
    second = api_client.post(
        "/chat",
        json={"message": "How many leave days?", "thread_id": "test-thread"},
    )

    assert first.status_code == 200
    assert first.headers["X-Request-ID"]
    assert first.json()["sources"] == ["handbook/leave-policy"]
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["model_used"] == "primary"
    assert api_client.get("/metrics").json()["total_requests"] == 2
    assert api_client.get("/cache/stats").json()["entries"] == 1


def test_ingestion_requires_key_and_indexes_documents(api_client):
    payload = {
        "documents": [
            {
                "source": "handbook/leave-policy",
                "content": "Employees receive 20 days.",
            }
        ]
    }
    chat_body = {"message": "How many leave days?"}
    first_chat = api_client.post("/chat", json=chat_body)
    cached_chat = api_client.post("/chat", json=chat_body)

    unauthorized = api_client.post("/documents", json=payload)
    indexed = api_client.post(
        "/documents",
        json=payload,
        headers={"X-Ingest-API-Key": "ingest-secret"},
    )
    refreshed_chat = api_client.post("/chat", json=chat_body)

    assert unauthorized.status_code == 401
    assert indexed.status_code == 200
    assert indexed.json() == {"documents_received": 1, "chunks_indexed": 2}
    assert first_chat.json()["cached"] is False
    assert cached_chat.json()["cached"] is True
    assert refreshed_chat.json()["cached"] is False
    assert len(api_client.app.state.agent.queries) == 2


def test_ingestion_is_disabled_without_an_ingest_key(monkeypatch):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)

    with TestClient(main.app) as client:
        response = client.post(
            "/documents",
            json={
                "documents": [
                    {"source": "guide", "content": "Document content."}
                ]
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Document ingestion is disabled."


def test_readiness_checks_database(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["checks"] == {"agent": True, "database": True}
    assert api_client.get("/health/live").json() == {"status": "alive"}


def test_readiness_returns_service_unavailable_when_database_is_down(
    monkeypatch,
):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    class UnhealthyAgent(FakeAgent):
        def database_ready(self) -> bool:
            return False

    monkeypatch.setattr(main, "ProductionAgent", UnhealthyAgent)
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["checks"]["database"] is False


def test_production_chat_requires_valid_api_key(monkeypatch):
    settings = Settings(
        _env_file=None,
        app_env="production",
        chat_api_key=SecretStr("chat-secret"),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)

    with TestClient(main.app) as client:
        denied = client.post("/chat", json={"message": "Hello"})
        allowed = client.post(
            "/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": "chat-secret"},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_rag_failure_returns_safe_upstream_error(monkeypatch):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    class FailedAgent(FakeAgent):
        def invoke(self, message: str) -> dict[str, object]:
            raise RuntimeError("provider secret details")

    monkeypatch.setattr(main, "ProductionAgent", FailedAgent)
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 502
    assert "provider secret details" not in response.text


def test_chat_integrates_retrieval_citations_and_response_validation(monkeypatch):
    settings = Settings(_env_file=None, query_expansion_enabled=False)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    agent = ProductionAgent.__new__(ProductionAgent)
    agent.vector_store = MockVectorStore(
        [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={"source": "handbook/leave-policy"},
            )
        ]
    )
    agent.retrieval_k = 1
    agent.retrieval_candidates_k = 2
    agent.max_context_chars = 1200
    agent.query_expansion_enabled = False
    agent.query_expansion_count = 2
    agent.query_rewriter = None
    agent.primary_chain = type(
        "AnswerChain",
        (),
        {
            "invoke": lambda self, values: (
                "Employees receive 20 days of annual leave [1]."
                if "annual leave" in values["context"]
                else "Unsupported answer [2]."
            )
        },
    )()
    agent.fallback_chain = None
    agent.logger = main.logger
    monkeypatch.setattr(main, "ProductionAgent", lambda _settings: agent)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"message": "How many annual leave days?"},
        )

    assert response.status_code == 200
    assert response.json()["response"].endswith("[1].")
    assert response.json()["sources"] == ["handbook/leave-policy"]


class MockVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def max_marginal_relevance_search(self, _query, *, k, fetch_k):
        assert k == 2
        assert fetch_k == 4
        return self.documents