"""Add a transactional outbox for durable worker dispatch.

Revision ID: 20261110_0002
Revises: 20261110_0001
"""

from alembic import op

from app.database.schema import document_outbox, metadata


revision = "20261110_0002"
down_revision = "20261110_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the outbox table when it is not already present."""
    metadata.create_all(
        bind=op.get_bind(),
        tables=[document_outbox],
        checkfirst=True,
    )


def downgrade() -> None:
    """Remove only the outbox table introduced by this revision."""
    document_outbox.drop(bind=op.get_bind(), checkfirst=True)
