"""Compatibility exports for ingestion file storage."""

from app.ingestion.storage import (
    delete_upload,
    local_document_path,
    s3_client,
    upload_path,
)

__all__ = ["delete_upload", "local_document_path", "s3_client", "upload_path"]
