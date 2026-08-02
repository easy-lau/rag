"""add KB-owned, revisioned terminology registry

Revision ID: 0032
Revises: 0031
Create Date: 2026-08-02

The registry is deliberately partitioned by knowledge base.  A term or scope
binding cannot reference a concept from another KB because both tables carry a
composite ``(concept_id, kb_id)`` foreign key.  Every existing and newly
created KB receives its own revision state; there is no global registry
revision that could invalidate unrelated snapshots.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision: str = "0032"
down_revision: Union[str, None] = "0031"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_NULL_DOCUMENT_SENTINEL = "00000000-0000-0000-0000-000000000000"


def upgrade() -> None:
    # PostgreSQL requires an exact unique target key for the composite
    # document/KB FK below.  ``documents.id`` is already globally unique, so
    # this only records the ownership invariant explicitly.
    op.create_unique_constraint(
        "uq_documents_id_kb_id", "documents", ["id", "kb_id"]
    )

    op.create_table(
        "terminology_concepts",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("canonical_term", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column("updated_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "char_length(code) > 0", name="ck_terminology_concepts_code_nonempty"
        ),
        sa.CheckConstraint(
            "char_length(btrim(canonical_term)) > 0",
            name="ck_terminology_concepts_canonical_term_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_bases.id"],
            name="fk_terminology_concepts_kb", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        # Required targets for composite term/binding FKs and per-KB code
        # uniqueness.  Neither one is a global concept namespace.
        sa.UniqueConstraint("id", "kb_id", name="uq_terminology_concepts_id_kb_id"),
        sa.UniqueConstraint("kb_id", "code", name="uq_terminology_concepts_kb_code"),
    )
    op.create_index(
        "ix_terminology_concepts_kb_active",
        "terminology_concepts", ["kb_id", "is_active"], unique=False,
    )

    op.create_table(
        "terminology_terms",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("concept_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column("term", sa.String(length=120), nullable=False),
        sa.Column("normalized_term", sa.String(length=120), nullable=False),
        sa.Column("match_mode", sa.String(length=24), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column("updated_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["concept_id", "kb_id"],
            ["terminology_concepts.id", "terminology_concepts.kb_id"],
            name="fk_terminology_terms_concept_kb", ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "match_mode IN ('strict_equivalent', 'retrieval_only')",
            name="ck_terminology_terms_match_mode",
        ),
        sa.CheckConstraint(
            "char_length(btrim(term)) > 0",
            name="ck_terminology_terms_term_nonempty",
        ),
        sa.CheckConstraint(
            "char_length(normalized_term) > 0",
            name="ck_terminology_terms_normalized_term_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "concept_id", "kb_id", "normalized_term",
            name="uq_terminology_terms_concept_kb_normalized",
        ),
    )
    op.create_index(
        "ix_terminology_terms_kb_normalized_active",
        "terminology_terms", ["kb_id", "normalized_term", "is_active"], unique=False,
    )

    op.create_table(
        "terminology_registry_state",
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "revision", sa.Integer(), nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("updated_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "revision >= 0", name="ck_terminology_registry_state_revision"
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("kb_id"),
    )
    # Backfill one state row per existing KB.  New KBs are initialized in the
    # same transaction by the KB creation domain path.
    op.execute(
        "INSERT INTO terminology_registry_state (kb_id, revision, updated_at) "
        "SELECT id, 0, CURRENT_TIMESTAMP FROM knowledge_bases "
        "ON CONFLICT (kb_id) DO NOTHING"
    )

    op.create_table(
        "terminology_registry_revisions",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("object_type", sa.String(length=48), nullable=False),
        sa.Column("object_id", sa.String(length=64), nullable=False),
        sa.Column("change_payload", JSONB(), nullable=False),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "revision > 0", name="ck_terminology_registry_revisions_revision"
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kb_id", "revision",
            name="uq_terminology_registry_revisions_kb_revision",
        ),
    )
    op.create_index(
        "ix_terminology_registry_revisions_kb_created_at",
        "terminology_registry_revisions", ["kb_id", "created_at"], unique=False,
    )

    op.create_table(
        "terminology_scope_bindings",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("concept_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=True),
        sa.Column("scope_product_key", sa.String(length=160), nullable=True),
        sa.Column("scope_version_key", sa.String(length=160), nullable=True),
        sa.Column("scope_project_key", sa.String(length=160), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column("updated_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["concept_id", "kb_id"],
            ["terminology_concepts.id", "terminology_concepts.kb_id"],
            name="fk_terminology_scope_bindings_concept_kb", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_bases.id"],
            name="fk_terminology_scope_bindings_kb", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "kb_id"], ["documents.id", "documents.kb_id"],
            name="fk_terminology_scope_bindings_document_kb", ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "scope_product_key IS NULL OR char_length(btrim(scope_product_key)) > 0",
            name="ck_terminology_scope_bindings_product_key_nonempty",
        ),
        sa.CheckConstraint(
            "scope_version_key IS NULL OR char_length(btrim(scope_version_key)) > 0",
            name="ck_terminology_scope_bindings_version_key_nonempty",
        ),
        sa.CheckConstraint(
            "scope_project_key IS NULL OR char_length(btrim(scope_project_key)) > 0",
            name="ck_terminology_scope_bindings_project_key_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # A normal UNIQUE constraint considers NULL selectors unequal.  This
    # expression index makes equivalent optional-scope bindings collide.
    op.execute(
        "CREATE UNIQUE INDEX uq_terminology_scope_bindings_identity "
        "ON terminology_scope_bindings ("
        "concept_id, kb_id, "
        f"COALESCE(document_id, '{_NULL_DOCUMENT_SENTINEL}'::uuid), "
        "COALESCE(scope_product_key, ''), "
        "COALESCE(scope_version_key, ''), "
        "COALESCE(scope_project_key, '')"
        ")"
    )
    op.create_index(
        "ix_terminology_scope_bindings_kb_active",
        "terminology_scope_bindings", ["kb_id", "is_active"], unique=False,
    )
    op.create_index(
        "ix_terminology_scope_bindings_document_active",
        "terminology_scope_bindings", ["document_id", "is_active"], unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_terminology_scope_bindings_document_active",
        table_name="terminology_scope_bindings",
    )
    op.drop_index(
        "ix_terminology_scope_bindings_kb_active",
        table_name="terminology_scope_bindings",
    )
    op.execute("DROP INDEX uq_terminology_scope_bindings_identity")
    op.drop_table("terminology_scope_bindings")

    op.drop_index(
        "ix_terminology_registry_revisions_kb_created_at",
        table_name="terminology_registry_revisions",
    )
    op.drop_table("terminology_registry_revisions")
    op.drop_table("terminology_registry_state")

    op.drop_index(
        "ix_terminology_terms_kb_normalized_active",
        table_name="terminology_terms",
    )
    op.drop_table("terminology_terms")

    op.drop_index(
        "ix_terminology_concepts_kb_active",
        table_name="terminology_concepts",
    )
    op.drop_table("terminology_concepts")
    op.drop_constraint("uq_documents_id_kb_id", "documents", type_="unique")
