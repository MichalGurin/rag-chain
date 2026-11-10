import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_production_requires_chat_api_key():
    with pytest.raises(ValidationError, match="CHAT_API_KEY"):
        Settings(_env_file=None, app_env="production")


def test_chunk_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_candidate_retrieval_count_must_cover_final_retrieval_count():
    with pytest.raises(ValidationError, match="RETRIEVAL_CANDIDATES_K"):
        Settings(
            _env_file=None,
            retrieval_k=5,
            retrieval_candidates_k=4,
        )


def test_production_settings_accept_secret_chat_key():
    settings = Settings(
        _env_file=None,
        app_env="production",
        chat_api_key=SecretStr("test-key"),
        tenant_id="tenant-a",
        database_url=SecretStr("postgresql+psycopg://db/rag"),
        redis_url=SecretStr("redis://redis:6379/0"),
        llm_base_url="https://llm.example.com/v1",
        s3_endpoint_url="https://s3.example.com",
        s3_bucket="documents",
        s3_access_key_id=SecretStr("test-access"),
        s3_secret_access_key=SecretStr("test-secret"),
    )

    assert settings.is_production


def test_production_requires_shared_object_storage():
    with pytest.raises(ValidationError, match="S3-compatible shared storage"):
        Settings(
            _env_file=None,
            app_env="production",
            chat_api_key=SecretStr("test-key"),
            tenant_id="tenant-a",
            database_url=SecretStr("postgresql+psycopg://db/rag"),
            redis_url=SecretStr("redis://redis:6379/0"),
            llm_base_url="https://llm.example.com/v1",
        )


def test_local_llm_defaults_need_no_provider_api_key():
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key.get_secret_value() == "ollama"
    assert settings.primary_model == "llama3.2:1b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chat_api_key is None
    assert settings.ingest_api_key.get_secret_value() == "local-ingest-key"
    assert settings.database_url.get_secret_value() == (
        "postgresql+psycopg://rag:rag@localhost:5432/rag"
    )
    assert settings.effective_tenant_id == "local"


def test_local_http_model_endpoint_is_allowed():
    settings = Settings(
        _env_file=None,
        llm_base_url="http://ollama:11434/v1",
    )

    assert settings.llm_base_url == "http://ollama:11434/v1"


def test_production_requires_a_tenant_id():
    with pytest.raises(ValidationError, match="TENANT_ID"):
        Settings(
            _env_file=None,
            app_env="production",
            chat_api_key=SecretStr("test-key"),
            database_url=SecretStr("postgresql+psycopg://db/rag"),
            redis_url=SecretStr("redis://redis:6379/0"),
            llm_base_url="https://llm.example.com/v1",
            s3_endpoint_url="https://s3.example.com",
            s3_bucket="documents",
            s3_access_key_id=SecretStr("test-access"),
            s3_secret_access_key=SecretStr("test-secret"),
        )


def test_nonlocal_environment_requires_a_tenant_id():
    with pytest.raises(ValidationError, match="TENANT_ID"):
        Settings(_env_file=None, app_env="staging")


def test_production_rejects_plain_http_llm_endpoint():
    with pytest.raises(ValidationError, match="LLM_BASE_URL must use HTTPS"):
        Settings(
            _env_file=None,
            app_env="production",
            chat_api_key=SecretStr("test-key"),
            tenant_id="tenant-a",
            database_url=SecretStr("postgresql+psycopg://db/rag"),
            redis_url=SecretStr("redis://redis:6379/0"),
            llm_base_url="http://llm.example.com/v1",
            s3_endpoint_url="https://s3.example.com",
            s3_bucket="documents",
            s3_access_key_id=SecretStr("test-access"),
            s3_secret_access_key=SecretStr("test-secret"),
        )


def test_indexing_version_tracks_embedding_configuration():
    original = Settings(_env_file=None)
    updated = Settings(_env_file=None, embedding_model="other-embedding-model")
    different_endpoint = Settings(
        _env_file=None,
        llm_base_url="https://llm.example.com/v1",
    )
    different_collection = Settings(
        _env_file=None,
        vector_collection_name="another-collection",
    )
    different_tenant = Settings(_env_file=None, tenant_id="tenant-a")

    assert original.indexing_version != updated.indexing_version
    assert original.indexing_version != different_endpoint.indexing_version
    assert original.indexing_version != different_collection.indexing_version
    assert original.indexing_version != different_tenant.indexing_version


def test_s3_configuration_must_be_complete():
    with pytest.raises(ValidationError, match="S3_ENDPOINT_URL"):
        Settings(
            _env_file=None,
            s3_endpoint_url="https://s3.example.com",
            s3_bucket="documents",
        )
