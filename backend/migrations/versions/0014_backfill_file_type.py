"""backfill documents.file_type mislabeled 'md' by the edit-resets-type bug

Restores file_type from the filename extension for rows currently typed 'md'
whose filename carries a recognized upload extension. Hand-written markdown
documents (titles without these extensions) are left untouched.

Revision ID: 0014
Revises: 0013
Create Date: 2026-06-14
"""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        UPDATE documents
        SET file_type = lower(substring(filename from '\.([^.]+)$'))
        WHERE file_type = 'md'
          AND lower(substring(filename from '\.([^.]+)$')) IN
              ('pdf','docx','doc','pptx','ppt','xlsx','xls','txt',
               'png','jpg','jpeg','webp','gif','bmp')
    """)


def downgrade() -> None:
    # 不可逆：无法区分原本即为这些类型的文档与被回填修正的文档
    pass
