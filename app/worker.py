"""Compatibility exports for the ingestion Celery application."""

from app.ingestion.tasks import (
    celery_app,
    dispatch_document_outbox,
    process_document,
)

__all__ = ["celery_app", "dispatch_document_outbox", "process_document"]
