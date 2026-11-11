from unittest.mock import MagicMock, Mock

from pydantic import SecretStr
from sqlalchemy.dialects.postgresql import dialect

from app.config import Settings
from app.database import DocumentRepository, engine as database_engine


def _result(row=None, *, one_or_none=None):
    mappings = Mock()
    mappings.one.return_value = row
    mappings.one_or_none.return_value = one_or_none
    result = Mock()
    result.mappings.return_value = mappings
    return result


def _repository_with_results(results):
    connection = Mock()
    connection.execute.side_effect = results
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    return DocumentRepository(engine), connection


def test_register_queued_retry_resets_existing_outbox_event():
    existing = {
        "id": "existing-id",
        "source": "policy.pdf",
        "sha256": "content-hash",
        "indexing_version": "index-version",
        "object_path": "uploads/existing-id.pdf",
        "previous_object_path": None,
        "content_type": "application/pdf",
        "status": "queued",
        "task_id": "existing-id",
    }
    repository, connection = _repository_with_results(
        [Mock(), _result(one_or_none=existing), Mock()]
    )

    document, enqueue = repository.register(
        document_id="new-id",
        source="policy.pdf",
        sha256="content-hash",
        indexing_version="index-version",
        object_path="uploads/new.pdf",
        content_type="application/pdf",
    )

    statement = connection.execute.call_args_list[2].args[0]
    compiled = statement.compile(dialect=dialect())
    sql = str(compiled)
    assert "ON CONFLICT (document_id) DO UPDATE" in sql
    assert "published_at = " in sql
    assert "locked_until = " in sql
    assert "attempts = " in sql
    assert list(compiled.params.values()).count(None) == 3
    assert 0 in compiled.params.values()
    assert document == existing
    assert enqueue is True


def test_register_creates_first_document_without_previous_object():
    row = {
        "id": "new-id",
        "source": "policy.pdf",
        "sha256": "content-hash",
        "indexing_version": "index-version",
        "object_path": "uploads/new-id.pdf",
        "previous_object_path": None,
        "content_type": "application/pdf",
        "status": "queued",
        "task_id": "new-id",
        "chunks_indexed": 0,
        "page_count": 0,
        "error": None,
    }
    repository, connection = _repository_with_results(
        [Mock(), _result(one_or_none=None), Mock(), Mock(), _result(row=row)]
    )

    created, enqueue = repository.register(
        document_id="new-id",
        source="policy.pdf",
        sha256="content-hash",
        indexing_version="index-version",
        object_path="uploads/new-id.pdf",
        content_type="application/pdf",
    )

    assert created == row
    assert enqueue is True
    assert connection.execute.call_count == 5
    insert_statement = connection.execute.call_args_list[2].args[0]
    assert insert_statement.compile().params["tenant_id"] == "local"


def test_register_replacement_retains_previous_file_until_indexing_finishes():
    existing = {
        "id": "old-id",
        "source": "policy.pdf",
        "sha256": "old-hash",
        "indexing_version": "old-version",
        "object_path": "uploads/old-id.pdf",
        "previous_object_path": None,
        "status": "completed",
    }
    replacement = {
        **existing,
        "id": "new-id",
        "sha256": "new-hash",
        "indexing_version": "index-version",
        "object_path": "uploads/new-id.pdf",
        "previous_object_path": "uploads/old-id.pdf",
        "status": "queued",
    }
    repository, connection = _repository_with_results(
        [
            Mock(),
            _result(one_or_none=existing),
            Mock(),
            Mock(),
            Mock(),
            _result(row=replacement),
        ]
    )

    created, enqueue = repository.register(
        document_id="new-id",
        source="policy.pdf",
        sha256="new-hash",
        indexing_version="index-version",
        object_path="uploads/new-id.pdf",
        content_type="application/pdf",
    )

    update_statement = connection.execute.call_args_list[3].args[0]
    assert update_statement.compile().params["previous_object_path"] == (
        "uploads/old-id.pdf"
    )
    assert created["previous_object_path"] == "uploads/old-id.pdf"
    assert enqueue is True


def test_document_status_lookup_is_scoped_to_repository_tenant():
    connection = MagicMock()
    connection.execute.return_value.mappings.return_value.one_or_none.return_value = None
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    repository = DocumentRepository(engine, tenant_id="tenant-a")

    assert repository.get("other-tenant-document") is None

    statement = connection.execute.call_args.args[0]
    assert statement.compile().params["tenant_id_1"] == "tenant-a"


def test_database_engine_uses_normalized_url_and_configured_timeouts(monkeypatch):
    settings = Settings(
        _env_file=None,
        database_url=SecretStr("postgres://localhost/db"),
        database_pool_timeout_seconds=45,
        database_connect_timeout_seconds=15,
    )
    create_engine = Mock()
    monkeypatch.setattr(database_engine, "create_engine", create_engine)

    database_engine.create_database_engine(settings)

    create_engine.assert_called_once_with(
        "postgresql+psycopg://localhost/db",
        pool_size=5,
        max_overflow=10,
        pool_timeout=45,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_use_lifo=True,
        connect_args={"connect_timeout": 15},
    )
