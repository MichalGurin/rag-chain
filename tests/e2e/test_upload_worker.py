"""End-to-end tests for API upload, PostgreSQL jobs, Redis, and Celery workers."""

import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, select

from app import main
from app.api import documents as document_api
from app.config import Settings
from app.database.schema import document_outbox, index_generation
from app.ingestion.tasks import celery_app
from tests.document_samples import supported_upload_samples

E2E_UPLOAD_CASES = [
    (*sample, "completed") for sample in supported_upload_samples()
] + [
    (
        "broken.pdf",
        b"%PDF-1.4\nthis is not a valid PDF document",
        "application/pdf",
        "failed",
    )
]


WORKER_BOOTSTRAP = """
import app.ingestion.tasks as tasks

class TestIndexer:
    def __init__(self, *_args, **_kwargs):
        pass

    def replace_document(self, document):
        assert "E2E annual leave policy" in document.content
        return 1

    def close(self):
        pass

tasks.ProductionAgent = TestIndexer
tasks.celery_app.worker_main([
    "worker",
    "--pool=solo",
    "--concurrency=1",
    "--loglevel=WARNING",
    "--without-gossip",
    "--without-mingle",
    "--without-heartbeat",
])
"""


def _migration_config(database_url: str) -> Config:
    """Build an Alembic config that targets the dedicated end-to-end database."""
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def _wait_for_job(client: TestClient, document_id: str, headers: dict[str, str]):
    """Poll job status until terminal state, while failing promptly on worker exit."""
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        status = client.get(f"/documents/{document_id}", headers=headers)
        assert status.status_code == 200
        body = status.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.25)
    pytest.fail("The ingestion job did not reach a terminal state in 45 seconds.")


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_status"),
    E2E_UPLOAD_CASES,
    ids=[
        f"{sample[0].split('.')[-1]}-{sample[3]}"
        for sample in E2E_UPLOAD_CASES
    ],
)
def test_upload_is_processed_by_a_real_celery_worker(
    tmp_path,
    monkeypatch,
    filename,
    content,
    content_type,
    expected_status,
):
    """Verify the complete upload-to-database workflow using live Postgres/Redis."""
    database_url = os.getenv("E2E_DATABASE_URL")
    redis_url = os.getenv("E2E_REDIS_URL")
    if not database_url or not redis_url:
        pytest.skip("Set E2E_DATABASE_URL and E2E_REDIS_URL to run service tests.")

    monkeypatch.setenv("DATABASE_URL", database_url)
    command.upgrade(_migration_config(database_url), "head")
    engine = create_engine(database_url, pool_pre_ping=True)
    settings = Settings(
        _env_file=None,
        app_env="e2e",
        tenant_id="local",
        database_url=SecretStr(database_url),
        redis_url=SecretStr(redis_url),
        upload_dir=str(tmp_path),
        ingest_api_key=SecretStr("e2e-ingestion-key"),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(document_api, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "ProductionAgent", lambda _settings: _ApiAgent())
    celery_app.conf.update(broker_url=redis_url, task_always_eager=False)

    worker_environment = os.environ.copy()
    worker_environment.update(
        {
            "APP_ENV": "e2e",
            "TENANT_ID": "local",
            "DATABASE_URL": database_url,
            "REDIS_URL": redis_url,
            "UPLOAD_DIR": str(tmp_path),
            "LLM_BASE_URL": settings.llm_base_url,
            "EMBEDDING_MODEL": settings.embedding_model,
            "VECTOR_COLLECTION_NAME": settings.vector_collection_name,
            "CHUNK_SIZE": str(settings.chunk_size),
            "CHUNK_OVERLAP": str(settings.chunk_overlap),
        }
    )
    worker_log = (tmp_path / "celery-worker.log").open("w", encoding="utf-8")
    worker = subprocess.Popen(
        [sys.executable, "-c", WORKER_BOOTSTRAP],
        cwd=Path(__file__).parents[2],
        env=worker_environment,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
    )
    try:
        headers = {"X-Ingest-API-Key": "e2e-ingestion-key"}
        filename = f"policy-{uuid4().hex}.{filename.rsplit('.', 1)[-1]}"
        rate_limit_ip = f"198.51.100.{uuid4().int % 250 + 1}"
        with TestClient(
            main.app,
            client=(rate_limit_ip, 12345),
        ) as client:
            accepted = client.post(
                "/documents/upload",
                headers=headers,
                files={
                    "file": (
                        filename,
                        content,
                        content_type,
                    )
                },
            )
            assert accepted.status_code == 202, accepted.text
            completed = _wait_for_job(
                client,
                accepted.json()["document_id"],
                headers,
            )

        assert completed["status"] == expected_status
        if expected_status == "completed":
            assert completed["chunks_indexed"] == 1
            assert completed["page_count"] == 1
        else:
            assert completed["chunks_indexed"] == 0
            assert completed["page_count"] == 0
            assert "Processing failed" in completed["error"]
            assert "PDF" not in completed["error"]
        with engine.connect() as connection:
            generation = connection.execute(
                select(index_generation.c.revision).where(
                    index_generation.c.tenant_id
                    == settings.effective_tenant_id
                )
            ).scalar_one()
            pending_events = connection.execute(
                select(document_outbox.c.document_id).where(
                    document_outbox.c.document_id == completed["document_id"]
                )
            ).all()
        if expected_status == "completed":
            assert generation >= 1
        assert pending_events == []
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=10)
        worker_log.close()
        engine.dispose()


class _ApiAgent:
    """Keep the HTTP app's readiness path independent of external LLM services."""

    def database_ready(self) -> bool:
        """Report readiness because this test uses its own live PostgreSQL server."""
        return True

    def close(self) -> None:
        """Release no resources because the test agent creates none."""
        return None
