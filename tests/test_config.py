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
    )

    assert settings.is_production


def test_local_llm_defaults_need_no_provider_api_key():
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key.get_secret_value() == "ollama"
    assert settings.primary_model == "llama3.2:1b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chat_api_key is None
    assert settings.ingest_api_key is None
