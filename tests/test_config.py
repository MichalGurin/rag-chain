import os
from types import SimpleNamespace
from uuid import UUID

from langsmith import Client
from langsmith.run_trees import get_cached_client
from langsmith.utils import tracing_is_enabled
from pydantic import SecretStr, ValidationError
import pytest

from app import config as config_module
from app.config import (
    Settings,
    build_langsmith_client,
    configure_langsmith_tracing,
    flush_langsmith_traces,
    langsmith_parent_trace,
    new_langsmith_run_id,
    record_langsmith_feedback,
)


def clear_langsmith_environment(monkeypatch):
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_TRACING_SAMPLING_RATE",
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_langsmith_tracing_is_opt_in(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = Settings(_env_file=None)

    configure_langsmith_tracing(settings)

    assert settings.langsmith_tracing is False
    assert settings.effective_langsmith_project == "prod-rag-development"
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert not tracing_is_enabled()


def test_langsmith_payload_visibility_follows_the_environment(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    development = Settings(_env_file=None)
    production = Settings(
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
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("test-langsmith-key"),
    )

    assert development.langsmith_hide_inputs is None
    assert development.effective_langsmith_hide_inputs is False
    assert development.effective_langsmith_hide_outputs is False
    assert production.effective_langsmith_hide_inputs is True
    assert production.effective_langsmith_hide_outputs is True


def test_langsmith_payload_visibility_can_be_overridden(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    traced_production = Settings(
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
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_hide_inputs=False,
        langsmith_hide_outputs=False,
    )
    private_development = Settings(
        _env_file=None,
        langsmith_hide_inputs=True,
        langsmith_hide_outputs=True,
    )

    assert traced_production.effective_langsmith_hide_inputs is False
    assert traced_production.effective_langsmith_hide_outputs is False
    assert private_development.effective_langsmith_hide_inputs is True
    assert private_development.effective_langsmith_hide_outputs is True


def test_langsmith_configuration_is_exported_to_sdk_environment(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_endpoint="https://smith.example.com",
        langsmith_tracing_sampling_rate=0.25,
    )
    monkeypatch.setenv("LANGSMITH_HIDE_INPUTS", "true")

    configure_langsmith_tracing(settings)
    client = get_cached_client()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-langsmith-key"
    assert os.environ["LANGSMITH_PROJECT"] == "prod-rag-development"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://smith.example.com"
    assert os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] == "0.25"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "false"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "false"
    assert client.tracing_sample_rate == 0.25
    assert tracing_is_enabled()


def test_disabling_langsmith_clears_stale_api_credentials(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "stale-key")
    settings = Settings(_env_file=None, langsmith_api_key=None)

    configure_langsmith_tracing(settings)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ
    assert not tracing_is_enabled()


def test_langsmith_tracing_requires_api_key(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        Settings(_env_file=None, langsmith_tracing=True)


def traced_settings(**overrides) -> Settings:
    """Build the settings a team member gets when they turn tracing on."""
    return Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("test-langsmith-key"),
        **overrides,
    )


def test_langsmith_client_scrubs_pii_from_traced_payloads(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = traced_settings()

    payload = build_langsmith_client(settings)._run_transform(
        {
            "inputs": {"message": "Email jane.doe@example.com or call 555-123-4567."},
            "outputs": {"answer": "Contact jane.doe@example.com."},
            "error": "Authorization: Bearer jane.doe@example.com failed",
        }
    )

    assert payload["inputs"] == {
        "message": "Email [REDACTED EMAIL] or call [REDACTED PHONE]."
    }
    assert payload["outputs"] == {"answer": "Contact [REDACTED EMAIL]."}
    assert "jane.doe@example.com" not in payload["error"]


def test_langsmith_client_omits_payloads_the_policy_hides(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = traced_settings(
        langsmith_hide_inputs=True,
        langsmith_hide_outputs=True,
    )

    payload = build_langsmith_client(settings)._run_transform(
        {
            "inputs": {"message": "A private question"},
            "outputs": {"answer": "A private answer"},
        }
    )

    assert payload["inputs"] == {}
    assert payload["outputs"] == {}


def test_langsmith_client_pins_the_teams_trace_policy(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = traced_settings(
        langsmith_tracing_sampling_rate=0.25,
        langsmith_endpoint="https://eu.api.smith.langchain.com",
    )

    client = build_langsmith_client(settings)

    assert client._anonymizer is config_module.PII_ANONYMIZER
    assert client._hide_inputs is False
    assert client._hide_outputs is False
    assert client.tracing_sample_rate == 0.25
    assert client._tracing_error_callback is config_module._log_tracing_error
    assert client.api_url == "https://eu.api.smith.langchain.com"


def test_langsmith_endpoint_defaults_to_the_us_cloud_api(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    settings = traced_settings()

    # API keys carry no region hint, so a blank endpoint must be explicit here.
    assert (
        settings.effective_langsmith_endpoint
        == "https://api.smith.langchain.com"
    )

    configure_langsmith_tracing(settings)

    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert get_cached_client().api_url == "https://api.smith.langchain.com"


def test_langsmith_feedback_scores_quality_signals_without_waiting(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(traced_settings())
    recorded: list[dict] = []
    monkeypatch.setattr(
        Client,
        "create_feedback",
        lambda self, **kwargs: recorded.append(kwargs),
    )

    queued = record_langsmith_feedback(
        trace_id="trace-1",
        key="grounded",
        value="supported",
        score=1,
    )

    assert queued is True
    assert recorded == [
        {
            "key": "grounded",
            "trace_id": "trace-1",
            "value": "supported",
            "score": 1,
            "feedback_source_type": "model",
        }
    ]


def test_langsmith_feedback_is_skipped_while_tracing_is_off(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(Settings(_env_file=None))
    calls: list[str] = []
    monkeypatch.setattr(
        Client,
        "create_feedback",
        lambda self, **kwargs: calls.append(kwargs["key"]),
    )

    queued = record_langsmith_feedback(
        trace_id="trace-1",
        key="grounded",
        value="supported",
        score=1,
    )

    assert queued is False
    assert calls == []


def test_langsmith_feedback_failures_never_break_an_answer(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(traced_settings())

    def fail(self, *args, **kwargs):
        raise RuntimeError("LangSmith is unavailable")

    monkeypatch.setattr(Client, "create_feedback", fail)

    assert (
        record_langsmith_feedback(
            trace_id="trace-1",
            key="cited",
            value=2,
            score=1,
        )
        is False
    )


def test_langsmith_parent_trace_needs_an_active_run(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(traced_settings())
    monkeypatch.setattr(config_module, "get_current_run_tree", lambda: None)

    assert langsmith_parent_trace() is None

    monkeypatch.setattr(
        config_module,
        "get_current_run_tree",
        lambda: SimpleNamespace(dotted_order="20261121T0000000000000000-request"),
    )

    assert langsmith_parent_trace() == "20261121T0000000000000000-request"


def test_langsmith_run_ids_are_only_promised_while_traced(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(Settings(_env_file=None))

    assert new_langsmith_run_id() is None

    configure_langsmith_tracing(traced_settings())
    traced_run_id = new_langsmith_run_id()

    assert traced_run_id
    assert UUID(traced_run_id)
    assert UUID(traced_run_id) != UUID(new_langsmith_run_id())


def test_shutdown_flushes_buffered_langsmith_traces(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(traced_settings())
    timeouts: list[float] = []
    monkeypatch.setattr(
        Client,
        "flush",
        lambda self, timeout=None: timeouts.append(timeout),
    )

    assert flush_langsmith_traces(timeout=1.5) is True
    assert timeouts == [1.5]


def test_shutdown_flush_is_skipped_while_tracing_is_off(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(Settings(_env_file=None))
    calls: list[float] = []
    monkeypatch.setattr(Client, "flush", lambda self, timeout=None: calls.append(timeout))

    assert flush_langsmith_traces() is False
    assert calls == []


def test_shutdown_flush_failures_are_logged_not_raised(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    configure_langsmith_tracing(traced_settings())

    def fail(self, timeout=None):
        raise RuntimeError("LangSmith is unreachable")

    monkeypatch.setattr(Client, "flush", fail)

    assert flush_langsmith_traces(timeout=0.1) is False


def test_production_langsmith_endpoint_requires_https(monkeypatch):
    clear_langsmith_environment(monkeypatch)
    with pytest.raises(ValidationError, match="LANGSMITH_ENDPOINT must use HTTPS"):
        Settings(
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
            langsmith_endpoint="http://smith.example.com",
        )


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


def test_tests_ignore_a_developers_tracing_setup(tmp_path, monkeypatch):
    """A local ``.env`` must not decide whether a unit test traces."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=ls__local\n",
        encoding="utf-8",
    )

    # The application honours that file when it reads its own configuration...
    assert Settings().langsmith_tracing is True
    # ...while every test starts with the SDK switched off and no key exported.
    assert not tracing_is_enabled()
    assert new_langsmith_run_id() is None


def test_s3_configuration_must_be_complete():
    with pytest.raises(ValidationError, match="S3_ENDPOINT_URL"):
        Settings(
            _env_file=None,
            s3_endpoint_url="https://s3.example.com",
            s3_bucket="documents",
        )
