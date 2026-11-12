"""SQLAlchemy metadata for persistent ingestion and index state."""

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)

metadata = MetaData()

documents = Table(
    "ingested_documents",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("source", String(500), nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("indexing_version", String(64), nullable=False),
    Column("object_path", Text, nullable=False),
    Column("previous_object_path", Text),
    Column("content_type", String(128), nullable=False),
    Column("status", String(16), nullable=False, index=True),
    Column("task_id", String(36), nullable=False),
    Column("chunks_indexed", Integer, nullable=False, default=0),
    Column("page_count", Integer, nullable=False, default=0),
    Column("error", Text),
    Column(
        "created_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
    Column(
        "updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
    UniqueConstraint(
        "tenant_id",
        "source",
        name="uq_ingested_documents_tenant_source",
    ),
)

document_outbox = Table(
    "document_job_outbox",
    metadata,
    Column("document_id", String(36), primary_key=True),
    Column(
        "created_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
    Column("locked_until", DateTime(timezone=True)),
    Column("published_at", DateTime(timezone=True)),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("last_error", Text),
)

index_generation = Table(
    "rag_index_generation",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("revision", Integer, nullable=False),
    Column(
        "updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
)
