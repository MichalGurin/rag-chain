import hashlib
import os
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True
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


@lru_cache
def get_settings() -> Settings:
    """Load and cache environment configuration once per process."""
    settings = Settings()
    configure_langsmith_tracing(settings)
    return settings


def configure_langsmith_tracing(settings: Settings) -> None:
    """Set LangSmith's supported environment-based tracing configuration."""
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_PROJECT"] = settings.effective_langsmith_project
    os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] = str(
        settings.langsmith_tracing_sampling_rate
    )
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(
        settings.langsmith_hide_inputs
    ).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(
        settings.langsmith_hide_outputs
    ).lower()

    if settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = (
            settings.langsmith_api_key.get_secret_value()
        )
    else:
        os.environ.pop("LANGSMITH_API_KEY", None)

    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    else:
        os.environ.pop("LANGSMITH_ENDPOINT", None)
