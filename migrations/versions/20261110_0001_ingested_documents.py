"""Add persistent document ingestion job registry.

Revision ID: 20261110_0001
Revises:
"""

from alembic import op

from app.database.schema import documents, index_generation, metadata


revision = "20261110_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the initial job registry and corpus-generation tables."""
    metadata.create_all(
        bind=op.get_bind(),
        tables=[documents, index_generation],
        checkfirst=True,
    )


def downgrade() -> None:
    """Drop only the tables owned by the initial migration."""
    metadata.drop_all(
        bind=op.get_bind(),
        tables=[index_generation, documents],
        checkfirst=True,
    )
