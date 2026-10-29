from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    primary_model: str = "llama3.2:1b"
    fallback_model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"

    database_url: SecretStr | None = None
    vector_collection_name: str = "documents"
    retrieval_k: int = Field(default=4, ge=1, le=20)
    retrieval_candidates_k: int = Field(default=12, ge=1, le=100)
    max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    query_expansion_enabled: bool = True
    query_expansion_count: int = Field(default=2, ge=1, le=3)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)

    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_max_entries: int = Field(default=1000, ge=1)
    chat_api_key: SecretStr | None = None
    ingest_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_candidates_k < self.retrieval_k:
            raise ValueError(
                "RETRIEVAL_CANDIDATES_K must be at least RETRIEVAL_K"
            )
        if self.is_production and self.chat_api_key is None:
            raise ValueError(
                "CHAT_API_KEY is required when APP_ENV=production"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
