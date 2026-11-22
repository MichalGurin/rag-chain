"""HTTP routes and validation for document upload and job status."""

import hashlib
import hmac
import zipfile
from functools import partial
from pathlib import Path, PurePath
from uuid import uuid4

from fastapi import APIRouter, File, Header, HTTPException, Request, Response, UploadFile
from langsmith import traceable
from starlette.concurrency import run_in_threadpool

from app.api.limits import limiter
from app.config import (
    Settings,
    get_settings,
    langsmith_parent_trace,
    new_langsmith_run_id,
)
from app.database import DocumentRepository
from app.ingestion.tasks import process_document
from app.ingestion.storage import delete_upload, upload_path
from app.models import DocumentStatusResponse, DocumentUploadResponse
from app.monitoring import get_logger


logger = get_logger("prod-rag.api.documents")
router = APIRouter()
SUPPORTED_UPLOADS = {".pdf", ".pptx", ".docx", ".txt", ".md"}


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


def _upload_source(filename: str | None) -> tuple[str, str]:
    """Reduce client paths to an allowed basename, rejecting empty/unknown types."""
    if not filename or "\x00" in filename:
        raise HTTPException(status_code=400, detail="A valid filename is required.")
    source = PurePath(filename.replace("\\", "/")).name.strip()
    suffix = Path(source).suffix.lower()
    if (
        not source
        or source in {".", ".."}
        or len(source) > 255
        or suffix not in SUPPORTED_UPLOADS
    ):
        raise HTTPException(
            status_code=415,
            detail="Supported file types are PDF, PPTX, DOCX, TXT, and MD.",
        )
    return source, suffix


def _persist_upload(upload: UploadFile, destination: Path, limit: int) -> str:
    """Stream and validate one file, rejecting oversize, empty, or bomb archives."""
    digest = hashlib.sha256()
    total_bytes = 0
    first_chunk = True
    with upload.file:
        with destination.open("xb") as target:
            while chunk := upload.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > limit:
                    raise UploadTooLargeError(
                        f"Upload exceeds the {limit}-byte limit."
                    )
                if first_chunk:
                    suffix = destination.suffix.lower()
                    if suffix == ".pdf" and not chunk.startswith(b"%PDF-"):
                        raise ValueError("The uploaded file is not a valid PDF.")
                    if suffix in {".pptx", ".docx"} and not chunk.startswith(b"PK"):
                        raise ValueError("The uploaded Office file is invalid.")
                    first_chunk = False
                digest.update(chunk)
                target.write(chunk)
    if total_bytes == 0:
        raise ValueError("Empty files cannot be uploaded.")
    if destination.suffix.lower() in {".pptx", ".docx"}:
        try:
            with zipfile.ZipFile(destination) as office_file:
                entries = office_file.infolist()
                expected_file = (
                    "ppt/presentation.xml"
                    if destination.suffix.lower() == ".pptx"
                    else "word/document.xml"
                )
                total_uncompressed = sum(entry.file_size for entry in entries)
                if (
                    expected_file not in office_file.namelist()
                    or len(entries) > 5000
                    or total_uncompressed > min(limit * 20, 250_000_000)
                    or any(
                        entry.file_size > 1_000_000
                        and entry.file_size / max(entry.compress_size, 1) > 200
                        for entry in entries
                    )
                ):
                    raise ValueError(
                        "The uploaded Office file is invalid or too large."
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError("The uploaded Office file is invalid.") from exc
    return digest.hexdigest()


def _authenticate_ingestion(request: Request, supplied_key: str | None) -> None:
    """Require the configured ingestion secret using constant-time comparison."""
    configured_key = request.app.state.settings.ingest_api_key
    if configured_key is None:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is disabled.",
        )
    if supplied_key is None or not hmac.compare_digest(
        configured_key.get_secret_value(),
        supplied_key,
    ):
        raise HTTPException(status_code=401, detail="Invalid ingestion API key.")


@traceable(
    name="document.dispatch",
    run_type="chain",
    tags=["api", "ingestion"],
    process_inputs=lambda inputs: {"document_id": inputs.get("document_id")},
)
def _dispatch_queued_documents(
    repository: DocumentRepository,
    document_id: str,
) -> int:
    """Publish queued jobs under this request's trace so indexing stays linked."""
    parent_trace = langsmith_parent_trace()
    publish = process_document.apply_async
    if parent_trace is not None:
        publish = partial(
            publish,
            kwargs={"langsmith_parent_trace": parent_trace},
        )
    return repository.dispatch_pending(publish)


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=202,
)
@limiter.limit(lambda: get_settings().ingest_rate_limit)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    x_ingest_api_key: str | None = Header(
        default=None,
        alias="X-Ingest-API-Key",
    ),
) -> DocumentUploadResponse:
    """Persist an upload and outbox event, retaining jobs through broker outages."""
    _authenticate_ingestion(request, x_ingest_api_key)
    settings = request.app.state.settings
    repository: DocumentRepository | None = request.app.state.document_repository
    if repository is None:
        raise HTTPException(
            status_code=503,
            detail="Document storage is not configured.",
        )

    document_id = uuid4().hex
    upload_directory = Path(settings.upload_dir)
    object_path: Path | None = None
    stored_path: str | None = None
    try:
        source, suffix = _upload_source(file.filename)
        object_path = upload_directory / f"{document_id}{suffix}"
        await run_in_threadpool(
            upload_directory.mkdir,
            parents=True,
            exist_ok=True,
        )
        digest = await run_in_threadpool(
            _persist_upload,
            file,
            object_path,
            settings.max_upload_bytes,
        )
        stored_path = await run_in_threadpool(
            upload_path,
            object_path,
            document_id,
            suffix,
            settings,
        )
        row, enqueue = await run_in_threadpool(
            repository.register,
            document_id=document_id,
            source=source,
            sha256=digest,
            indexing_version=settings.indexing_version,
            object_path=stored_path,
            content_type=file.content_type or "application/octet-stream",
        )
        duplicate = row["id"] != document_id or not enqueue
        if duplicate:
            try:
                await run_in_threadpool(delete_upload, stored_path, settings)
            except Exception:
                logger.exception(
                    "Could not remove the redundant duplicate upload",
                    extra={"extra_data": {"document_id": row["id"]}},
                )

        dispatch_run_id = new_langsmith_run_id()
        langsmith_extra = {
            "metadata": {
                "request_id": request.state.request_id,
                "environment": settings.app_env,
                "tenant_id": settings.effective_tenant_id,
                "document_id": str(row["id"]),
                "indexing_version": settings.indexing_version,
            }
        }
        if dispatch_run_id is not None:
            langsmith_extra["run_id"] = dispatch_run_id

        try:
            await run_in_threadpool(
                _dispatch_queued_documents,
                repository,
                str(row["id"]),
                langsmith_extra=langsmith_extra,
            )
        except Exception:
            logger.exception(
                "Document remains in the durable outbox after broker failure",
                extra={"extra_data": {"document_id": row["id"]}},
            )

        request.app.state.cache.clear()
        response.headers["Location"] = f"/documents/{row['id']}"
        if dispatch_run_id is not None:
            response.headers["X-LangSmith-Run-Id"] = dispatch_run_id
        logger.info(
            "Document upload accepted",
            extra={
                "extra_data": {
                    "document_id": row["id"],
                    "source": source,
                    "duplicate": duplicate,
                }
            },
        )
        return DocumentUploadResponse(
            document_id=str(row["id"]),
            source=source,
            status=str(row["status"]),
            duplicate=duplicate,
        )
    except HTTPException:
        raise
    except UploadTooLargeError as exc:
        await _cleanup_failed_upload(
            stored_path,
            object_path,
            settings,
            str(document_id),
        )
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        await _cleanup_failed_upload(
            stored_path,
            object_path,
            settings,
            str(document_id),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        await _cleanup_failed_upload(
            stored_path,
            object_path,
            settings,
            str(document_id),
        )
        logger.exception(
            "Document upload could not be persisted",
            extra={"extra_data": {"request_id": request.state.request_id}},
        )
        raise HTTPException(
            status_code=503,
            detail="Document storage could not accept this upload.",
        ) from None
    finally:
        await file.close()


async def _cleanup_failed_upload(
    stored_path: str | None,
    local_path: Path | None,
    settings: Settings,
    document_id: str,
) -> None:
    """Best-effort cleanup after registration failure, logging orphan risk."""
    try:
        if stored_path is not None:
            await run_in_threadpool(delete_upload, stored_path, settings)
        elif local_path is not None:
            await run_in_threadpool(local_path.unlink, missing_ok=True)
    except Exception:
        logger.exception(
            "Could not clean up an upload after request failure",
            extra={"extra_data": {"document_id": document_id}},
        )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
)
def get_document_status(
    document_id: str,
    request: Request,
    x_ingest_api_key: str | None = Header(
        default=None,
        alias="X-Ingest-API-Key",
    ),
) -> DocumentStatusResponse:
    """Return sanitized job state only to callers with the ingestion key."""
    _authenticate_ingestion(request, x_ingest_api_key)
    repository: DocumentRepository | None = request.app.state.document_repository
    if repository is None:
        raise HTTPException(
            status_code=503,
            detail="Document storage is not configured.",
        )
    row = repository.get(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document job not found.")
    return DocumentStatusResponse(
        document_id=str(row["id"]),
        source=str(row["source"]),
        status=str(row["status"]),
        chunks_indexed=int(row["chunks_indexed"]),
        page_count=int(row["page_count"]),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
