from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


GroundingStatus = Literal["supported", "potential_drift", "unverified"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    thread_id: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("message", "thread_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank")
        return value


class ChatResponse(BaseModel):
    response: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    thread_id: str = "default"
    model_used: str
    sources: list[str] = Field(default_factory=list)
    grounding_status: GroundingStatus = "unverified"
    cached: bool = False
    processing_time_ms: float = 0.0


class DocumentInput(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    source: str = Field(min_length=1, max_length=500)

    @field_validator("content", "source")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank")
        return value


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    content: str


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    documents_received: int
    chunks_indexed: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    source: str
    status: str
    duplicate: bool = False


class DocumentStatusResponse(BaseModel):
    document_id: str
    source: str
    status: str
    chunks_indexed: int
    page_count: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str = "1.0.0"
    checks: dict[str, bool]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MetricResponse(BaseModel):
    total_requests: int = 0
    total_errors: int = 0
    error_rate: str = "0.00%"
    avg_latency_ms: float = 0.0
    cache_hit_rate: str = "0.00%"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ConfigResponse(BaseModel):
    max_upload_bytes: int = 50_000_000
