"""Celery tasks for document indexing and outbox delivery."""

from celery import Celery
from celery.app.task import Task
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.agent import ProductionAgent
from app.config import get_settings
from app.database import DocumentRepository, create_database_engine
from app.ingestion.parser import parse_document
from app.models import ParsedDocument
from app.monitoring import get_logger
from app.ingestion.storage import delete_upload, local_document_path


settings = get_settings()
logger = get_logger("prod-rag.worker", settings.log_level)
celery_app = Celery(
    "prod-rag",
    broker=settings.redis_url.get_secret_value(),
    include=["app.ingestion.tasks"],
)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    broker_heartbeat=30,
    broker_transport_options={"visibility_timeout": 3600},
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_default_queue="document-ingestion",
    task_default_delivery_mode=2,
    task_soft_time_limit=1800,
    task_time_limit=1860,
    beat_schedule={
        "dispatch-document-outbox": {
            "task": "app.dispatch_document_outbox",
            "schedule": 5.0,
        }
    },
)


@celery_app.task(
    bind=True,
    name="app.process_document",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_backoff_max=60,
)
def process_document(self, document_id: str) -> dict[str, int | str]:
    """Index the current revision; ignore stale IDs and retry transient failures."""
    engine = create_database_engine(settings)
    tenant_id = settings.effective_tenant_id
    repository = DocumentRepository(engine, tenant_id=tenant_id)
    lock_connection = None
    lock_acquired = False
    source: str | None = None

    try:
        row = repository.get(document_id)
        if row is None:
            logger.warning(
                "Ignoring stale document task",
                extra={"extra_data": {"document_id": document_id}},
            )
            return {"document_id": document_id, "status": "stale"}

        source = str(row["source"])
        lock_connection = engine.connect()
        lock_connection.execute(text("SET lock_timeout = '30s'"))
        lock_connection.execute(
            text(
                "SELECT pg_advisory_lock("
                "hashtextextended('index:' || :tenant_id || ':' || :source, 0))"
            ),
            {"tenant_id": tenant_id, "source": source},
        )
        lock_acquired = True
        row = repository.get(document_id)
        if row is None or not repository.is_current(
            document_id,
            str(row["sha256"]),
        ):
            return {"document_id": document_id, "status": "stale"}
        if row["status"] in {"completed", "failed"}:
            repository.clear_outbox_event(document_id)
            return {
                "document_id": document_id,
                "status": str(row["status"]),
                "chunks_indexed": int(row["chunks_indexed"]),
            }

        repository.set_status(document_id, "processing")
        try:
            if row["indexing_version"] != settings.indexing_version:
                raise ValueError("Worker and API indexing settings do not match.")

            with local_document_path(
                str(row["object_path"]),
                document_id,
                settings,
            ) as local_path:
                content, page_count = parse_document(
                    local_path,
                    settings.max_document_pages,
                )
                if not content.strip():
                    raise ValueError(
                        "No text could be extracted. Scanned PDFs require OCR."
                    )
                if len(content) > settings.max_document_chars:
                    raise ValueError(
                        "Extracted text exceeds the configured document limit."
                    )
                agent = ProductionAgent(settings, database_engine=engine)
                try:
                    chunks_indexed = agent.replace_document(
                        ParsedDocument(source=source, content=content)
                    )
                finally:
                    agent.close()

            repository.bump_generation()
            repository.set_status(
                document_id,
                "completed",
                chunks_indexed=chunks_indexed,
                page_count=page_count,
            )
            repository.clear_outbox_event(document_id)
            previous_object_path = row["previous_object_path"]
            if previous_object_path:
                _delete_replaced_upload(
                    repository,
                    document_id,
                    str(previous_object_path),
                    source,
                )
            logger.info(
                "Document ingestion completed",
                extra={
                    "extra_data": {
                        "document_id": document_id,
                        "source": source,
                        "chunks_indexed": chunks_indexed,
                    }
                },
            )
            return {
                "document_id": document_id,
                "status": "completed",
                "chunks_indexed": chunks_indexed,
            }
        except Exception as exc:
            _record_processing_failure(
                self,
                repository,
                document_id,
                source,
                exc,
            )
            raise
    finally:
        try:
            if lock_connection is not None:
                if lock_acquired and source is not None:
                    lock_connection.execute(
                        text(
                            "SELECT pg_advisory_unlock("
                            "hashtextextended("
                            "'index:' || :tenant_id || ':' || :source, 0))"
                        ),
                        {"tenant_id": tenant_id, "source": source},
                    )
                lock_connection.close()
        except SQLAlchemyError:
            logger.exception(
                "Could not release document advisory lock",
                extra={
                    "extra_data": {
                        "document_id": document_id,
                        "source": source,
                    }
                },
            )
        finally:
            engine.dispose()


def _record_processing_failure(
    task: Task,
    repository: DocumentRepository,
    document_id: str,
    source: str,
    error: Exception,
) -> None:
    """Persist retry/failure state without exposing internal errors to clients."""
    retries = int(task.request.retries)
    max_retries = int(task.max_retries or 3)
    can_retry = not isinstance(error, ValueError) and retries < max_retries
    logger.exception(
        "Document ingestion failed",
        extra={
            "extra_data": {
                "document_id": document_id,
                "source": source,
                "retry": can_retry,
            }
        },
    )
    repository.set_status(
        document_id,
        "queued" if can_retry else "failed",
        error=(
            "Retry scheduled. Check service logs using the document ID."
            if can_retry
            else "Processing failed. Check service logs using the document ID."
        ),
    )
    if not can_retry:
        repository.clear_outbox_event(document_id)
    if can_retry:
        raise task.retry(exc=error, countdown=min(2 ** retries, 60))


def _delete_replaced_upload(
    repository: DocumentRepository,
    document_id: str,
    previous_object_path: str,
    source: str,
) -> None:
    """Delete an old object after replacement indexing, logging cleanup failures."""
    try:
        delete_upload(previous_object_path, settings)
        repository.clear_previous_upload(document_id, previous_object_path)
    except Exception:
        logger.exception(
            "Could not clean up the replaced document file",
            extra={
                "extra_data": {
                    "document_id": document_id,
                    "source": source,
                }
            },
        )


@celery_app.task(name="app.dispatch_document_outbox")
def dispatch_document_outbox() -> dict[str, int]:
    """Deliver pending DB outbox events; the beat schedule recovers broker outages."""
    engine = create_database_engine(settings)
    try:
        repository = DocumentRepository(
            engine,
            tenant_id=settings.effective_tenant_id,
        )
        published = repository.dispatch_pending(process_document.apply_async)
        return {"published": published}
    finally:
        engine.dispose()
