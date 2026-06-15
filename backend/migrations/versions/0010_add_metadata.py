"""add kb tags/creator and document audit metadata

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # knowledge_bases
    op.add_column(
        "knowledge_bases",
        sa.Column("tags", JSONB, nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_bases_created_by_users",
        "knowledge_bases", "users",
        ["created_by"], ["id"],
        ondelete="SET NULL",
    )

    # documents
    op.add_column(
        "documents",
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("updated_by", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_documents_created_by_users",
        "documents", "users",
        ["created_by"], ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_documents_updated_by_users",
        "documents", "users",
        ["updated_by"], ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # documents
    op.drop_constraint("fk_documents_updated_by_users", "documents", type_="foreignkey")
    op.drop_constraint("fk_documents_created_by_users", "documents", type_="foreignkey")
    op.drop_column("documents", "updated_at")
    op.drop_column("documents", "updated_by")
    op.drop_column("documents", "created_by")

    # knowledge_bases
    op.drop_constraint("fk_knowledge_bases_created_by_users", "knowledge_bases", type_="foreignkey")
    op.drop_column("knowledge_bases", "created_by")
    op.drop_column("knowledge_bases", "tags")
