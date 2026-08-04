"""remove the retired controlled terminology registry

Revision ID: 0035
Revises: 0034
Create Date: 2026-08-04

The controlled-terminology menu, API and RAG runtime have been retired.  This
 migration removes only the registry tables and the composite document key
 that existed solely to support their foreign key.  Knowledge bases,
documents, conversations and all other application data are untouched.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0035"
down_revision: Union[str, None] = "0034"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop dependents before the concept table they reference.
    op.execute("DROP INDEX IF EXISTS ix_terminology_scope_bindings_document_active")
    op.execute("DROP INDEX IF EXISTS ix_terminology_scope_bindings_kb_active")
    op.execute("DROP INDEX IF EXISTS uq_terminology_scope_bindings_identity")
    op.drop_table("terminology_scope_bindings")

    op.execute(
        "DROP INDEX IF EXISTS ix_terminology_registry_revisions_kb_created_at"
    )
    op.drop_table("terminology_registry_revisions")
    op.drop_table("terminology_registry_state")

    op.execute("DROP INDEX IF EXISTS ix_terminology_terms_kb_normalized_active")
    op.drop_table("terminology_terms")

    op.execute("DROP INDEX IF EXISTS ix_terminology_concepts_kb_active")
    op.drop_table("terminology_concepts")

    # This constraint was introduced only as the composite target for the
    # terminology document-scope foreign key.
    op.drop_constraint("uq_documents_id_kb_id", "documents", type_="unique")


def downgrade() -> None:
    raise RuntimeError(
        "0035 removes a retired feature and is intentionally irreversible; "
        "restore a database backup instead of recreating terminology tables."
    )
