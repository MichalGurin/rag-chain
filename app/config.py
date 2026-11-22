import hashlib
import os
from functools import lru_cache
from uuid import uuid4
from urllib.parse import urlparse

from langsmith import Client
from langsmith import configure as langsmith_configure
from langsmith.anonymizer import create_anonymizer
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import get_cached_client
from langsmith.utils import tracing_is_enabled
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.monitoring import get_logger
from app.security import PIIDetector

logger = get_logger("prod-rag.config")

# LangSmith applies this to every traced input, output, metadata value, and error
# message, so PII is masked before it leaves the process even when payloads are visible.
PII_ANONYMIZER = create_anonymizer(PIIDetector().mask)

# The SDK default. Regional organizations must set LANGSMITH_ENDPOINT, because API
# keys carry no region hint and are rejected by every other region's API.
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


def _log_tracing_error(error: BaseException) -> None:
    """Report trace export failures without breaking the traced request."""
    logger.warning("LangSmith trace export failed", exc_info=error)


class Settings(BaseSettings):
    """Validated runtime settings shared by API, workers, and dispatchers."""

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    primary_model: str = "llama3.2:1b"
    fallback_model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"

    database_url: SecretStr | None = None
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    upload_dir: str = "data/uploads"
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_region: str = "us-east-1"
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    database_pool_timeout_seconds: int = Field(default=30, ge=1, le=300)
    database_connect_timeout_seconds: int = Field(default=10, ge=1, le=120)
    max_upload_bytes: int = Field(default=50_000_000, ge=1_000_000)
    max_document_pages: int = Field(default=1000, ge=1, le=10000)
    max_document_chars: int = Field(default=5_000_000, ge=50_000, le=50_000_000)
    vector_collection_name: str = "documents"
    retrieval_k: int = Field(default=4, ge=1, le=20)
    retrieval_candidates_k: int = Field(default=12, ge=1, le=100)
    max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    query_expansion_enabled: bool = True
    query_expansion_count: int = Field(default=2, ge=1, le=3)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    cohere_api_key: SecretStr | None = None
    rerank_model: str = "rerank-v3.5"
    rerank_candidates_k: int = Field(default=30, ge=1, le=100)
    max_estimated_llm_cost_per_request_usd: float = Field(
        default=0.05,
        ge=0.0,
        le=100.0,
    )
    max_total_llm_cost_usd: float | None = Field(default=None, ge=0.0)

    app_env: str = "development"
    log_level: str = "INFO"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    langsmith_endpoint: str | None = None
    langsmith_tracing_sampling_rate: float = Field(default=1.0, ge=0, le=1)
    langsmith_hide_inputs: bool | None = None
    langsmith_hide_outputs: bool | None = None
    rate_limit: str = "20/minute"
    ingest_rate_limit: str = "5/minute"
    max_ingest_chars: int = Field(default=250_000, ge=1, le=5_000_000)
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_max_entries: int = Field(default=1000, ge=1)
    chat_api_key: SecretStr | None = None
    ingest_api_key: SecretStr | None = None
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    @field_validator("tenant_id")
    @classmethod
    def normalize_tenant_id(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("langsmith_project")
    @classmethod
    def normalize_langsmith_project(cls, value: str | None) -> str | None:
        if value is None:
            return None
        project = value.strip()
        if not project:
            raise ValueError("LANGSMITH_PROJECT must not be blank")
        return project

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Reject inconsistent indexing, storage, and production service settings."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_candidates_k < self.retrieval_k:
            raise ValueError(
                "RETRIEVAL_CANDIDATES_K must be at least RETRIEVAL_K"
            )
        if self.rerank_candidates_k < self.retrieval_k:
            raise ValueError(
                "RERANK_CANDIDATES_K must be at least RETRIEVAL_K"
            )
        s3_values = (
            self.s3_endpoint_url,
            self.s3_bucket,
            self.s3_access_key_id,
            self.s3_secret_access_key,
        )
        if any(s3_values) and not all(s3_values):
            raise ValueError(
                "S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY_ID, and "
                "S3_SECRET_ACCESS_KEY must be configured together"
            )
        if self.is_production and self.chat_api_key is None:
            raise ValueError("CHAT_API_KEY is required in production")
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError(
                "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled"
            )
        if (
            self.is_production
            and self.langsmith_endpoint
            and urlparse(self.langsmith_endpoint).scheme != "https"
        ):
            raise ValueError("LANGSMITH_ENDPOINT must use HTTPS in production")
        if (
            self.is_production
            and self.s3_endpoint_url
            and urlparse(self.s3_endpoint_url).scheme != "https"
        ):
            raise ValueError("S3_ENDPOINT_URL must use HTTPS in production")
        if self.is_local:
            if self.database_url is None:
                self.database_url = SecretStr(
                    "postgresql+psycopg://rag:rag@localhost:5432/rag"
                )
            if self.ingest_api_key is None:
                self.ingest_api_key = SecretStr("local-ingest-key")
        elif not self.tenant_id:
            raise ValueError("TENANT_ID is required outside local development")
        if self.is_production:
            if self.tenant_id == "local":
                raise ValueError("TENANT_ID must not be 'local' in production")
            if self.database_url is None:
                raise ValueError("DATABASE_URL is required in production")
            if urlparse(self.database_url.get_secret_value()).scheme not in {
                "postgres",
                "postgresql",
                "postgresql+psycopg",
            }:
                raise ValueError("DATABASE_URL must use PostgreSQL in production")
            if urlparse(self.redis_url.get_secret_value()).scheme not in {
                "redis",
                "rediss",
            } or urlparse(self.redis_url.get_secret_value()).hostname in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError("REDIS_URL must point to a remote broker in production")
            llm_url = urlparse(self.llm_base_url)
            if (
                llm_url.scheme != "https"
                or not llm_url.hostname
                or llm_url.hostname in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError(
                    "LLM_BASE_URL must use HTTPS and point to a configured "
                    "service in production"
                )
            if self.s3_bucket is None:
                raise ValueError(
                    "S3-compatible shared storage is required in production"
                )
        return self

    @property
    def effective_tenant_id(self) -> str:
        """Return the configured tenant, or the isolated local-dev namespace."""
        return self.tenant_id or "local"

    @property
    def is_local(self) -> bool:
        """Return whether local-only development defaults may be used."""
        return self.app_env.lower() in {"development", "local"}

    @property
    def is_production(self) -> bool:
        """Return whether production-only validation and security policies apply."""
        return self.app_env.lower() == "production"

    @property
    def indexing_version(self) -> str:
        """Hash embedding and chunking inputs to detect API/worker drift."""
        version = (
            f"{self.effective_tenant_id}|{self.llm_base_url}|"
            f"{self.embedding_model}|"
            f"{self.vector_collection_name}|{self.chunk_size}|"
            f"{self.chunk_overlap}|parser-chunker-v1"
        )
        return hashlib.sha256(version.encode("utf-8")).hexdigest()

    @property
    def effective_langsmith_project(self) -> str:
        """Separate traces by environment unless a project is explicitly set."""
        return self.langsmith_project or f"prod-rag-{self.app_env.lower()}"

    @property
    def effective_langsmith_endpoint(self) -> str:
        """Report the API host traces are posted to, which keys are region bound to."""
        return self.langsmith_endpoint or DEFAULT_LANGSMITH_ENDPOINT

    @property
    def effective_langsmith_hide_inputs(self) -> bool:
        """Hide trace inputs in production unless explicitly configured."""
        if self.langsmith_hide_inputs is None:
            return self.is_production
        return self.langsmith_hide_inputs

    @property
    def effective_langsmith_hide_outputs(self) -> bool:
        """Hide trace outputs in production unless explicitly configured."""
        if self.langsmith_hide_outputs is None:
            return self.is_production
        return self.langsmith_hide_outputs


@lru_cache
def get_settings() -> Settings:
    """Load and cache environment configuration once per process."""
    settings = Settings()
    configure_langsmith_tracing(settings)
    return settings


def build_langsmith_client(settings: Settings) -> Client:
    """Build the single LangSmith client this process reports traces through."""
    return Client(
        api_url=settings.effective_langsmith_endpoint,
        api_key=settings.langsmith_api_key.get_secret_value()
        if settings.langsmith_api_key is not None
        else None,
        anonymizer=PII_ANONYMIZER,
        hide_inputs=settings.effective_langsmith_hide_inputs,
        hide_outputs=settings.effective_langsmith_hide_outputs,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
        tracing_error_callback=_log_tracing_error,
    )


def configure_langsmith_tracing(settings: Settings) -> None:
    """Apply this service's tracing policy to the LangSmith SDK.

    The environment variables keep every SDK code path, including clients built
    by dependencies and child processes, on the same project and payload policy.
    The configured client additionally pins PII masking and trace-export errors
    for this process.
    """
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_PROJECT"] = settings.effective_langsmith_project
    os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] = str(
        settings.langsmith_tracing_sampling_rate
    )
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(
        settings.effective_langsmith_hide_inputs
    ).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(
        settings.effective_langsmith_hide_outputs
    ).lower()
    if settings.langsmith_api_key is None:
        os.environ.pop("LANGSMITH_API_KEY", None)
    else:
        os.environ["LANGSMITH_API_KEY"] = (
            settings.langsmith_api_key.get_secret_value()
        )
    # Always export the host: dependencies that build their own LangSmith client
    # must post to the same regional API as the configured client.
    os.environ["LANGSMITH_ENDPOINT"] = settings.effective_langsmith_endpoint

    if not settings.langsmith_tracing:
        langsmith_configure(enabled=False)
        return

    langsmith_configure(
        client=build_langsmith_client(settings),
        enabled=True,
        project_name=settings.effective_langsmith_project,
    )


def flush_langsmith_traces(timeout: float = 5.0) -> bool:
    """Export buffered traces before a process exits so shutdown is not lossy."""
    if not tracing_is_enabled():
        return False
    try:
        get_cached_client().flush(timeout=timeout)
    except Exception as error:
        _log_tracing_error(error)
        return False
    return True


def new_langsmith_run_id() -> str | None:
    """Return the trace identifier this request's root run will use, if traced.

    Knowing the identifier before the RAG chain runs lets the API return it to
    the caller, who can then open that exact trace in LangSmith.
    """
    if not tracing_is_enabled():
        return None
    return str(uuid4())


def langsmith_parent_trace() -> str | None:
    """Return the current run's dotted order for background jobs to nest under."""
    if not tracing_is_enabled():
        return None
    run_tree = get_current_run_tree()
    if run_tree is None or not run_tree.dotted_order:
        return None
    return run_tree.dotted_order


def record_langsmith_feedback(
    *,
    trace_id: str,
    key: str,
    value: str | int,
    score: int,
) -> bool:
    """Attach automatic feedback to a trace for LangSmith analytics.

    The trace identifier is supplied so feedback is batched in the background;
    answering a request never waits on the LangSmith API.
    """
    if not tracing_is_enabled():
        return False
    try:
        get_cached_client().create_feedback(
            key=key,
            trace_id=trace_id,
            value=value,
            score=score,
            feedback_source_type="model",
        )
        return True
    except Exception as error:
        _log_tracing_error(error)
        return False
