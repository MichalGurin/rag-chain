"""Compatibility exports for the database package."""

from app.database import (
    DocumentRepository,
    create_database_engine,
    document_outbox,
    documents,
    index_generation,
    metadata,
)

__all__ = [
    "DocumentRepository",
    "create_database_engine",
    "document_outbox",
    "documents",
    "index_generation",
    "metadata",
]
