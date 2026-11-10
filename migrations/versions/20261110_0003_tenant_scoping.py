"""Scope document records and legacy vectors to a tenant.

Revision ID: 20261110_0003
Revises: 20261110_0002
"""

from alembic import op
from sqlalchemy import Column, String, inspect, text

revision = "20261110_0003"
down_revision = "20261110_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Assign old data to local and add tenant scoping."""
    bind = op.get_bind()
    inspector = inspect(bind)

    if inspector.has_table("ingested_documents"):
        columns = {
            column["name"]
            for column in inspector.get_columns("ingested_documents")
        }
        if "tenant_id" not in columns:
            op.add_column(
                "ingested_documents",
                Column(
                    "tenant_id",
                    String(128),
                    nullable=False,
                    server_default="local",
                ),
            )
            op.alter_column(
                "ingested_documents",
                "tenant_id",
                server_default=None,
            )

        unique_constraints = inspector.get_unique_constraints(
            "ingested_documents"
        )
        has_scoped_constraint = any(
            constraint["name"] == "uq_ingested_documents_tenant_source"
            for constraint in unique_constraints
        )
        if not has_scoped_constraint:
            for constraint in unique_constraints:
                if constraint["column_names"] == ["source"]:
                    op.drop_constraint(
                        constraint["name"],
                        "ingested_documents",
                        type_="unique",
                    )
            for index in inspector.get_indexes("ingested_documents"):
                if index["unique"] and index["column_names"] == ["source"]:
                    op.drop_index(
                        index["name"], table_name="ingested_documents"
                    )
            op.create_unique_constraint(
                "uq_ingested_documents_tenant_source",
                "ingested_documents",
                ["tenant_id", "source"],
            )

    if inspect(bind).has_table("langchain_pg_embedding"):
        bind.execute(
            text(
                """
                UPDATE langchain_pg_embedding
                SET cmetadata = jsonb_set(
                    coalesce(cmetadata, '{}'::jsonb),
                    '{tenant_id}',
                    '"local"'::jsonb,
                    true
                )
                """
            )
        )


def downgrade() -> None:
    """Restore the pre-tenant source uniqueness constraint."""
    inspector = inspect(op.get_bind())
    if not inspector.has_table("ingested_documents"):
        return

    constraints = inspector.get_unique_constraints("ingested_documents")
    for constraint in constraints:
        if constraint["name"] == "uq_ingested_documents_tenant_source":
            op.drop_constraint(
                constraint["name"],
                "ingested_documents",
                type_="unique",
            )
    op.create_unique_constraint(
        "ingested_documents_source_key",
        "ingested_documents",
        ["source"],
    )
    op.drop_column("ingested_documents", "tenant_id")
