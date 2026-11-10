"""Opt-in checks against explicitly configured live model, vector, and S3 services."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr

from app.agent import ProductionAgent
from app.config import Settings
from app.ingestion.storage import (
    delete_upload,
    local_document_path,
    upload_path,
)
from app.models import ParsedDocument

pytestmark = pytest.mark.live_services


def _live_services_enabled() -> bool:
    return os.getenv("RUN_LIVE_SERVICE_TESTS") == "1"


@pytest.fixture
def live_settings():
    if not _live_services_enabled():
        pytest.skip("Set RUN_LIVE_SERVICE_TESTS=1 to run live integrations.")
    required = (
        "LIVE_DATABASE_URL",
        "LIVE_LLM_BASE_URL",
        "LIVE_LLM_API_KEY",
        "LIVE_CHAT_MODEL",
        "LIVE_EMBEDDING_MODEL",
        "LIVE_TENANT_ID",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing live model/vector settings: {', '.join(missing)}")
    return Settings(
        _env_file=None,
        app_env="integration",
        tenant_id=os.environ["LIVE_TENANT_ID"],
        database_url=SecretStr(os.environ["LIVE_DATABASE_URL"]),
        llm_base_url=os.environ["LIVE_LLM_BASE_URL"],
        llm_api_key=SecretStr(os.environ["LIVE_LLM_API_KEY"]),
        primary_model=os.environ["LIVE_CHAT_MODEL"],
        fallback_model=os.environ["LIVE_CHAT_MODEL"],
        embedding_model=os.environ["LIVE_EMBEDDING_MODEL"],
        vector_collection_name=f"live-test-{uuid4().hex}",
        query_expansion_enabled=False,
        retrieval_k=2,
        retrieval_candidates_k=4,
        rerank_candidates_k=4,
    )


@pytest.fixture
def live_s3_settings():
    if not _live_services_enabled():
        pytest.skip("Set RUN_LIVE_SERVICE_TESTS=1 to run live integrations.")
    names = (
        "LIVE_S3_ENDPOINT_URL",
        "LIVE_S3_BUCKET",
        "LIVE_S3_ACCESS_KEY_ID",
        "LIVE_S3_SECRET_ACCESS_KEY",
        "LIVE_TENANT_ID",
    )
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing live S3 settings: {', '.join(missing)}")
    return Settings(
        _env_file=None,
        app_env="integration",
        tenant_id=os.environ["LIVE_TENANT_ID"],
        s3_endpoint_url=os.environ["LIVE_S3_ENDPOINT_URL"],
        s3_bucket=os.environ["LIVE_S3_BUCKET"],
        s3_access_key_id=SecretStr(os.environ["LIVE_S3_ACCESS_KEY_ID"]),
        s3_secret_access_key=SecretStr(
            os.environ["LIVE_S3_SECRET_ACCESS_KEY"]
        ),
        s3_region=os.getenv("LIVE_S3_REGION", "us-east-1"),
    )


def test_live_chat_embedding_and_pgvector_round_trip(live_settings):
    agent = ProductionAgent(live_settings)
    source = f"live-integration/{uuid4().hex}"
    content = (
        "The live integration test plan has exactly thirty-seven blue widgets."
    )
    try:
        embedding = agent.embeddings.embed_query(content)
        assert embedding
        assert all(isinstance(value, float) for value in embedding)

        assert agent.replace_document(
            ParsedDocument(source=source, content=content)
        ) >= 1
        result = agent.invoke("How many blue widgets are in the test plan?")

        assert result["model_used"] in {"primary", "fallback"}
        assert source in result["sources"]
        assert "[1]" in result["response"]
    finally:
        try:
            agent.vector_store.delete_collection()
        finally:
            agent.close()


def test_live_s3_put_get_delete_round_trip(live_s3_settings, tmp_path):
    settings = live_s3_settings
    document_id = f"prod-rag-live-tests-{uuid4().hex}"
    body = b"live S3 integration round trip"
    local_file = tmp_path / "live-s3-test.txt"
    local_file.write_bytes(body)
    object_path = upload_path(local_file, document_id, ".txt", settings)
    assert not local_file.exists()
    try:
        with local_document_path(object_path, "s3-live-test", settings) as path:
            assert Path(path).read_bytes() == body
    finally:
        delete_upload(object_path, settings)
