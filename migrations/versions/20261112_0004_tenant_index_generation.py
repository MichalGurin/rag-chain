"""Scope corpus generations to tenants.

Revision ID: 20261112_0004
Revises: 20261110_0003
"""

from alembic import op
from sqlalchemy import Column, Integer, String, inspect, text


revision = "20261112_0004"
down_revision = "20261110_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert the global generation row to a tenant-keyed table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("rag_index_generation"):
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("rag_index_generation")
    }
    if "tenant_id" in columns:
        return

    op.add_column(
        "rag_index_generation",
        Column("tenant_id", String(128), nullable=True),
    )
    bind.execute(
        text("UPDATE rag_index_generation SET tenant_id = 'local'")
    )
    op.alter_column("rag_index_generation", "tenant_id", nullable=False)

    primary_key = inspect(bind).get_pk_constraint("rag_index_generation")
    if primary_key["name"]:
        op.drop_constraint(
            primary_key["name"],
            "rag_index_generation",
            type_="primary",
        )
    if "id" in columns:
        op.drop_column("rag_index_generation", "id")
    op.create_primary_key(
        "rag_index_generation_pkey",
        "rag_index_generation",
        ["tenant_id"],
    )


def downgrade() -> None:
    """Collapse tenant generations into the old global revision."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("rag_index_generation"):
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("rag_index_generation")
    }
    if "tenant_id" not in columns:
        return
    primary_key = inspector.get_pk_constraint("rag_index_generation")
    if primary_key["name"]:
        op.drop_constraint(
            primary_key["name"],
            "rag_index_generation",
            type_="primary",
        )
    op.add_column(
        "rag_index_generation",
        Column("id", Integer, nullable=True),
    )
    bind.execute(
        text(
            """
            DELETE FROM rag_index_generation
            WHERE tenant_id <> (
                SELECT tenant_id
                FROM rag_index_generation
                ORDER BY revision DESC, tenant_id
                LIMIT 1
            )
            """
        )
    )
    bind.execute(text("UPDATE rag_index_generation SET id = 1"))
    op.drop_column("rag_index_generation", "tenant_id")
    op.alter_column("rag_index_generation", "id", nullable=False)
    op.create_primary_key(
        "rag_index_generation_pkey",
        "rag_index_generation",
        ["id"],
    )
