"""Shared local and S3-compatible storage operations for uploaded files."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import boto3
from botocore.client import BaseClient

from app.config import Settings


def s3_client(settings: Settings) -> BaseClient | None:
    """Build an S3 client or return None when local storage is configured."""
    if not settings.s3_bucket:
        return None
    if (
        settings.s3_access_key_id is None
        or settings.s3_secret_access_key is None
    ):
        raise RuntimeError("S3 credentials are incomplete.")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.s3_secret_access_key.get_secret_value(),
        region_name=settings.s3_region,
    )


def upload_path(
    local_path: Path,
    document_id: str,
    suffix: str,
    settings: Settings,
) -> str:
    """Upload to shared object storage, retaining local files on upload failure."""
    client = s3_client(settings)
    if client is None:
        return str(local_path)
    key = f"documents/{document_id}{suffix}"
    client.upload_file(str(local_path), settings.s3_bucket, key)
    local_path.unlink(missing_ok=True)
    return f"s3://{settings.s3_bucket}/{key}"


def delete_upload(object_path: str, settings: Settings) -> None:
    """Delete an upload only from the configured bucket or local filesystem."""
    if object_path.startswith("s3://"):
        bucket_and_key = object_path[5:]
        if "/" not in bucket_and_key:
            raise ValueError("The stored object path is malformed.")
        bucket, key = bucket_and_key.split("/", 1)
        client = s3_client(settings)
        if client is None or bucket != settings.s3_bucket:
            raise RuntimeError("The document object bucket is not configured.")
        client.delete_object(Bucket=bucket, Key=key)
    else:
        Path(object_path).unlink(missing_ok=True)


@contextmanager
def local_document_path(
    object_path: str,
    document_id: str,
    settings: Settings,
) -> Generator[str, None, None]:
    """Yield a local path, downloading and cleaning temporary S3 files as needed."""
    if not object_path.startswith("s3://"):
        yield object_path
        return
    bucket_and_key = object_path[5:]
    if "/" not in bucket_and_key:
        raise ValueError("The stored object path is malformed.")
    bucket, key = bucket_and_key.split("/", 1)
    client = s3_client(settings)
    if client is None or bucket != settings.s3_bucket:
        raise RuntimeError("Document object storage is not configured.")
    suffix = Path(key).suffix
    with TemporaryDirectory(prefix="rag-document-") as directory:
        local_path = Path(directory) / f"{document_id}{suffix}"
        client.download_file(bucket, key, str(local_path))
        yield str(local_path)
