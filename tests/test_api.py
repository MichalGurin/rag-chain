from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
from pydantic import SecretStr
from langchain_core.documents import Document

from app import main
from app.api import documents as document_api
from app.agent import ProductionAgent
from app.config import Settings

pytestmark = pytest.mark.integration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


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
    settings.database_url = None
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


def test_chat_adds_correlation_metadata_to_langsmith_context(
    api_client,
    monkeypatch,
):
    trace_context = Mock(side_effect=lambda **_kwargs: nullcontext())
    monkeypatch.setattr(main, "tracing_context", trace_context)

    response = api_client.post(
        "/chat",
        json={"message": "How many leave days?", "thread_id": "test-thread"},
    )

    assert response.status_code == 200
    trace_context.assert_called_once()
    options = trace_context.call_args.kwargs
    assert options["tags"] == ["api", "chat"]
    assert options["metadata"] == {
        "request_id": response.headers["X-Request-ID"],
        "environment": "development",
        "tenant_id": "local",
        "thread_id": "test-thread",
    }


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


def test_local_default_ingest_key_enables_ingestion(monkeypatch):
    settings = Settings(_env_file=None)
    settings.database_url = None
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)

    with TestClient(main.app) as client:
        response = client.post(
            "/documents",
            headers={"X-Ingest-API-Key": "local-ingest-key"},
            json={
                "documents": [
                    {"source": "guide", "content": "Document content."}
                ]
            },
        )

    assert response.status_code == 200
    assert response.json() == {"documents_received": 1, "chunks_indexed": 2}


def test_ingestion_rejects_batches_over_character_limit(monkeypatch):
    settings = Settings(
        _env_file=None,
        ingest_api_key=SecretStr("ingest-secret"),
        max_ingest_chars=10,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)

    with TestClient(main.app) as client:
        response = client.post(
            "/documents",
            json={
                "documents": [
                    {"source": "guide", "content": "More than ten characters"}
                ]
            },
            headers={"X-Ingest-API-Key": "ingest-secret"},
        )

    assert response.status_code == 413
    assert "10-character ingestion limit" in response.json()["detail"]


def test_ingestion_endpoint_applies_its_rate_limit(monkeypatch):
    settings = Settings(
        _env_file=None,
        ingest_api_key=SecretStr("ingest-secret"),
    )
    settings.database_url = None
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)
    payload = {
        "documents": [{"source": "guide", "content": "Document content."}]
    }

    with TestClient(main.app, client=("198.51.100.27", 12345)) as client:
        responses = [
            client.post(
                "/documents",
                json=payload,
                headers={"X-Ingest-API-Key": "ingest-secret"},
            )
            for _ in range(6)
        ]

    assert [response.status_code for response in responses[:5]] == [200] * 5
    assert responses[5].status_code == 429


def test_upload_queues_file_and_exposes_protected_job_status(
    tmp_path,
    monkeypatch,
):
    settings = Settings(
        _env_file=None,
        ingest_api_key=SecretStr("ingest-secret"),
        database_url=SecretStr("postgresql+psycopg://user:pass@localhost/rag"),
        upload_dir=str(tmp_path),
    )
    repository = FakeDocumentRepository()
    engine = SimpleNamespace(dispose=Mock())
    queued_task = SimpleNamespace(
        apply_async=Mock(side_effect=[RuntimeError("broker down"), None])
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)
    monkeypatch.setattr(main, "create_database_engine", lambda _settings: engine)
    monkeypatch.setattr(
        main,
        "DocumentRepository",
        lambda _engine, **_kwargs: repository,
    )
    monkeypatch.setattr(document_api, "process_document", queued_task)
    headers = {"X-Ingest-API-Key": "ingest-secret"}
    upload = {
        "file": (
            "handbook.txt",
            b"Employees receive 20 days of annual leave.",
            "text/plain",
        )
    }

    with TestClient(main.app) as client:
        accepted = client.post(
            "/documents/upload",
            files=upload,
            headers=headers,
        )
        repeated = client.post(
            "/documents/upload",
            files=upload,
            headers=headers,
        )
        status = client.get(
            f"/documents/{accepted.json()['document_id']}",
            headers=headers,
        )
        unauthorized = client.get(
            f"/documents/{accepted.json()['document_id']}"
        )

    assert accepted.status_code == 202
    assert repeated.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert accepted.json()["source"] == "handbook.txt"
    assert repeated.json()["duplicate"] is True
    assert repeated.json()["document_id"] == accepted.json()["document_id"]
    assert status.status_code == 200
    assert status.json()["status"] == "queued"
    assert status.json()["page_count"] == 0
    assert unauthorized.status_code == 401
    assert queued_task.apply_async.call_count == 2
    expected_dispatch = {
        "args": [accepted.json()["document_id"]],
        "task_id": accepted.json()["document_id"],
        "retry": False,
    }
    assert queued_task.apply_async.call_args_list[0].kwargs == expected_dispatch
    assert queued_task.apply_async.call_args_list[1].kwargs == expected_dispatch
    assert engine.dispose.called


def test_upload_and_job_status_reject_missing_or_invalid_ingest_keys(api_client):
    upload = {
        "file": ("handbook.txt", b"Employees receive 20 days.", "text/plain")
    }

    missing_key = api_client.post("/documents/upload", files=upload)
    invalid_key = api_client.post(
        "/documents/upload",
        files=upload,
        headers={"X-Ingest-API-Key": "wrong-key"},
    )
    hidden_job = api_client.get("/documents/private-job")

    assert missing_key.status_code == 401
    assert invalid_key.status_code == 401
    assert hidden_job.status_code == 401
    assert "private-job" not in hidden_job.text


class FakeDocumentRepository:
    def __init__(self):
        self.rows = {}
        self.published = set()

    def register(
        self,
        *,
        document_id,
        source,
        sha256,
        indexing_version,
        object_path,
        content_type,
    ):
        existing = self.rows.get(source)
        if (
            existing is not None
            and existing["sha256"] == sha256
            and existing["indexing_version"] == indexing_version
        ):
            return existing, existing["status"] == "queued"
        now = datetime.now(timezone.utc)
        row = {
            "id": document_id,
            "source": source,
            "sha256": sha256,
            "indexing_version": indexing_version,
            "object_path": object_path,
            "content_type": content_type,
            "status": "queued",
            "chunks_indexed": 0,
            "page_count": 0,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self.rows[source] = row
        return row, True

    def get(self, document_id):
        return next(
            (
                row
                for row in self.rows.values()
                if row["id"] == document_id
            ),
            None,
        )

    def dispatch_pending(self, publish):
        queued = [
            row
            for row in self.rows.values()
            if row["status"] == "queued" and row["id"] not in self.published
        ]
        for row in queued:
            publish(args=[row["id"]], task_id=row["id"], retry=False)
            self.published.add(row["id"])
        return len(queued)


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


def test_security_headers_cover_success_and_error_responses(api_client):
    responses = [
        api_client.get("/health/live"),
        api_client.get("/health"),
        api_client.get("/metrics"),
        api_client.get("/cache/stats"),
        api_client.get("/openapi.json"),
        api_client.get("/documents/missing"),
        api_client.post("/chat", json={"message": "Hello"}),
    ]

    for response in responses:
        assert response.headers["X-Request-ID"]
        for header, expected_value in SECURITY_HEADERS.items():
            assert response.headers[header] == expected_value


def test_service_endpoint_inventory_stays_explicit():
    service_routes = {
        (path, method.upper())
        for path, methods in main.app.application.openapi()["paths"].items()
        for method in methods
    }

    assert service_routes == {
        ("/chat", "POST"),
        ("/config", "GET"),
        ("/documents", "POST"),
        ("/documents/upload", "POST"),
        ("/documents/{document_id}", "GET"),
        ("/health/live", "GET"),
        ("/health", "GET"),
        ("/metrics", "GET"),
        ("/cache/stats", "GET"),
    }


def test_config_endpoint_exposes_effective_upload_limit(api_client):
    api_client.app.state.settings.max_upload_bytes = 123_456
    response = api_client.get("/config")

    assert response.status_code == 200
    assert response.json()["max_upload_bytes"] == 123_456


def test_production_https_responses_include_hsts(monkeypatch):
    settings = Settings(
        _env_file=None,
        app_env="production",
        chat_api_key=SecretStr("chat-secret"),
        tenant_id="test-tenant",
        database_url=SecretStr("postgresql+psycopg://db/rag"),
        redis_url=SecretStr("redis://redis:6379/0"),
        llm_base_url="https://llm.example.com/v1",
        s3_endpoint_url="https://s3.example.com",
        s3_bucket="documents",
        s3_access_key_id=SecretStr("test-access"),
        s3_secret_access_key=SecretStr("test-secret"),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)

    with TestClient(main.app, base_url="https://testserver") as client:
        secure_response = client.get("/health/live")
    with TestClient(main.app, base_url="http://testserver") as client:
        plain_response = client.get("/health/live")

    assert secure_response.headers["Strict-Transport-Security"] == (
        "max-age=31536000; includeSubDomains"
    )
    assert "Strict-Transport-Security" not in plain_response.headers


def test_upload_registration_failure_is_sanitized_and_cleans_file(
    tmp_path,
    monkeypatch,
):
    settings = Settings(
        _env_file=None,
        ingest_api_key=SecretStr("ingest-secret"),
        database_url=SecretStr("postgresql+psycopg://db/rag"),
        upload_dir=str(tmp_path),
    )
    engine = SimpleNamespace(dispose=Mock())

    class FailedRepository:
        def __init__(self, _engine, **_kwargs):
            pass

        def register(self, **_kwargs):
            raise RuntimeError("database password must not be exposed")

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", FakeAgent)
    monkeypatch.setattr(main, "create_database_engine", lambda _settings: engine)
    monkeypatch.setattr(main, "DocumentRepository", FailedRepository)

    with TestClient(main.app) as client:
        response = client.post(
            "/documents/upload",
            headers={"X-Ingest-API-Key": "ingest-secret"},
            files={"file": ("policy.txt", b"policy", "text/plain")},
        )

    assert response.status_code == 503
    assert "database password" not in response.text
    assert list(tmp_path.iterdir()) == []


def test_production_chat_requires_valid_api_key(monkeypatch):
    settings = Settings(
        _env_file=None,
        app_env="production",
        chat_api_key=SecretStr("chat-secret"),
        tenant_id="test-tenant",
        database_url=SecretStr("postgresql+psycopg://db/rag"),
        redis_url=SecretStr("redis://redis:6379/0"),
        llm_base_url="https://llm.example.com/v1",
        s3_endpoint_url="https://s3.example.com",
        s3_bucket="documents",
        s3_access_key_id=SecretStr("test-access"),
        s3_secret_access_key=SecretStr("test-secret"),
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
    settings.database_url = None
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    agent = ProductionAgent.__new__(ProductionAgent)
    agent.tenant_id = "local"
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

    def max_marginal_relevance_search(self, _query, *, k, fetch_k, filter):
        assert k == 2
        assert fetch_k == 4
        assert filter == {"tenant_id": "local"}
        return self.documents