"""Transactional repositories for ingestion jobs and corpus generations."""

from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.dialects.postgresql import insert

from app.database.schema import document_outbox, documents, index_generation


class DocumentRepository:
    """Persist document revisions, job states, and durable queue publications."""

    def __init__(self, engine: Engine, tenant_id: str = "local") -> None:
        """Bind the repository to the shared, application-managed engine."""
        self.engine = engine
        self.tenant_id = tenant_id

    def _create_outbox_event(
        self,
        connection: Connection,
        document_id: str,
    ) -> None:
        """Insert or reset the durable dispatch event in the job transaction."""
        statement = insert(document_outbox).values(document_id=document_id)
        statement = statement.on_conflict_do_update(
            index_elements=[document_outbox.c.document_id],
            set_={
                "created_at": func.now(),
                "locked_until": None,
                "published_at": None,
                "attempts": 0,
                "last_error": None,
            },
        )
        connection.execute(statement)

    def register(
        self,
        *,
        document_id: str,
        source: str,
        sha256: str,
        indexing_version: str,
        object_path: str,
        content_type: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically register a revision and outbox event, deduplicating retries."""
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "hashtextextended('register:' || :tenant_id || ':' || :source, 0))"
                ),
                {"tenant_id": self.tenant_id, "source": source},
            )
            existing = (
                connection.execute(
                    select(documents).where(
                        documents.c.tenant_id == self.tenant_id,
                        documents.c.source == source,
                    )
                )
                .mappings()
                .one_or_none()
            )

            if (
                existing is not None
                and existing["sha256"] == sha256
                and existing["indexing_version"] == indexing_version
            ):
                if existing["status"] == "queued":
                    connection.execute(
                        insert(document_outbox)
                        .values(document_id=existing["id"])
                        .on_conflict_do_nothing(
                            index_elements=[document_outbox.c.document_id]
                        )
                    )
                    return dict(existing), True
                if existing["status"] in {"processing", "completed"}:
                    return dict(existing), False

                connection.execute(
                    update(documents)
                    .where(
                        documents.c.id == existing["id"],
                        documents.c.tenant_id == self.tenant_id,
                    )
                    .values(
                        status="queued",
                        object_path=object_path,
                        previous_object_path=(
                            existing["object_path"]
                            if existing["object_path"] != object_path
                            else existing["previous_object_path"]
                        ),
                        content_type=content_type,
                        task_id=existing["id"],
                        error=None,
                        updated_at=func.now(),
                    )
                )
                self._create_outbox_event(connection, str(existing["id"]))
                refreshed = (
                    connection.execute(
                        select(documents).where(
                            documents.c.id == existing["id"],
                            documents.c.tenant_id == self.tenant_id,
                        )
                    )
                    .mappings()
                    .one()
                )
                return dict(refreshed), True

            if existing is None:
                connection.execute(
                    documents.insert().values(
                        id=document_id,
                        tenant_id=self.tenant_id,
                        source=source,
                        sha256=sha256,
                        indexing_version=indexing_version,
                        object_path=object_path,
                        content_type=content_type,
                        status="queued",
                        task_id=document_id,
                    )
                )
            else:
                connection.execute(
                    delete(document_outbox).where(
                        document_outbox.c.document_id == existing["id"]
                    )
                )
                connection.execute(
                    update(documents)
                    .where(
                        documents.c.tenant_id == self.tenant_id,
                        documents.c.source == source,
                    )
                    .values(
                        id=document_id,
                        sha256=sha256,
                        indexing_version=indexing_version,
                        object_path=object_path,
                        previous_object_path=existing["object_path"],
                        content_type=content_type,
                        status="queued",
                        task_id=document_id,
                        chunks_indexed=0,
                        page_count=0,
                        error=None,
                        updated_at=func.now(),
                    )
                )
            self._create_outbox_event(connection, document_id)
            created = (
                connection.execute(
                    select(documents).where(
                        documents.c.id == document_id,
                        documents.c.tenant_id == self.tenant_id,
                    )
                )
                .mappings()
                .one()
            )
            return dict(created), True

    def get(self, document_id: str) -> dict[str, Any] | None:
        """Return a job snapshot, or None when the ID is unknown."""
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(documents).where(
                        documents.c.id == document_id,
                        documents.c.tenant_id == self.tenant_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            return dict(row) if row is not None else None

    def set_status(
        self,
        document_id: str,
        status: str,
        *,
        chunks_indexed: int | None = None,
        page_count: int | None = None,
        error: str | None = None,
    ) -> bool:
        """Update a job while preserving result counts omitted by the caller."""
        values: dict[str, Any] = {
            "status": status,
            "error": error,
            "updated_at": func.now(),
        }
        if chunks_indexed is not None:
            values["chunks_indexed"] = chunks_indexed
        if page_count is not None:
            values["page_count"] = page_count
        with self.engine.begin() as connection:
            result = connection.execute(
                update(documents)
                .where(
                    documents.c.id == document_id,
                    documents.c.tenant_id == self.tenant_id,
                )
                .values(**values)
            )
            return result.rowcount == 1

    def is_current(self, document_id: str, sha256: str) -> bool:
        """Check that the job still represents the active source revision."""
        with self.engine.connect() as connection:
            return (
                connection.execute(
                    select(documents.c.id).where(
                        documents.c.id == document_id,
                        documents.c.tenant_id == self.tenant_id,
                        documents.c.sha256 == sha256,
                    )
                ).scalar_one_or_none()
                is not None
            )

    def clear_outbox_event(self, document_id: str) -> None:
        """Remove a dispatch event once its job is terminal or superseded."""
        with self.engine.begin() as connection:
            connection.execute(
                delete(document_outbox).where(
                    document_outbox.c.document_id == document_id,
                    document_outbox.c.document_id.in_(
                        select(documents.c.id).where(
                            documents.c.tenant_id == self.tenant_id
                        )
                    ),
                )
            )

    def dispatch_pending(
        self,
        publish: Callable[..., Any],
        *,
        batch_size: int = 20,
    ) -> int:
        """Publish leased jobs and retain failed events for later retry.

        A crash after broker publication may result in duplicate delivery; tasks
        are idempotent. A five-minute redispatch window recovers broker message
        loss without repeatedly flooding a healthy worker queue.
        """
        with self.engine.begin() as connection:
            pending = (
                select(document_outbox.c.document_id)
                .join(
                    documents,
                    documents.c.id == document_outbox.c.document_id,
                )
                .where(
                    documents.c.tenant_id == self.tenant_id,
                    documents.c.status == "queued",
                    or_(
                        document_outbox.c.published_at.is_(None),
                        document_outbox.c.published_at
                        < func.now() - text("INTERVAL '5 minutes'"),
                    ),
                    or_(
                        document_outbox.c.locked_until.is_(None),
                        document_outbox.c.locked_until < func.now(),
                    ),
                )
                .order_by(document_outbox.c.created_at)
                .limit(batch_size)
                .with_for_update(
                    of=document_outbox,
                    skip_locked=True,
                )
            )
            document_ids = list(connection.execute(pending).scalars())
            if document_ids:
                connection.execute(
                    update(document_outbox)
                    .where(document_outbox.c.document_id.in_(document_ids))
                    .values(
                        locked_until=func.now() + text("INTERVAL '1 minute'"),
                        attempts=document_outbox.c.attempts + 1,
                    )
                )

        published = 0
        for document_id in document_ids:
            try:
                publish(
                    args=[document_id],
                    task_id=document_id,
                    retry=False,
                )
            except Exception:
                with self.engine.begin() as connection:
                    connection.execute(
                        update(document_outbox)
                        .where(document_outbox.c.document_id == document_id)
                        .values(
                            locked_until=func.now()
                            + text("INTERVAL '1 minute'"),
                            last_error="Broker publication failed.",
                        )
                    )
                raise
            with self.engine.begin() as connection:
                connection.execute(
                    update(document_outbox)
                    .where(document_outbox.c.document_id == document_id)
                    .values(
                        locked_until=None,
                        published_at=func.now(),
                        last_error=None,
                    )
                )
            published += 1
        return published

    def bump_generation(self) -> None:
        """Increment the shared corpus revision after successful indexing."""
        statement = insert(index_generation).values(id=1, revision=1)
        statement = statement.on_conflict_do_update(
            index_elements=[index_generation.c.id],
            set_={
                "revision": index_generation.c.revision + 1,
                "updated_at": func.now(),
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)

    def clear_previous_upload(
        self,
        document_id: str,
        object_path: str,
    ) -> None:
        """Clear a replaced-file reference only if it still matches the path."""
        with self.engine.begin() as connection:
            connection.execute(
                update(documents)
                .where(
                    documents.c.id == document_id,
                    documents.c.tenant_id == self.tenant_id,
                    documents.c.previous_object_path == object_path,
                )
                .values(previous_object_path=None, updated_at=func.now())
            )
