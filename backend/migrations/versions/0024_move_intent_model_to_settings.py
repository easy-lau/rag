"""move intent model into model management settings

Revision ID: 0024
Revises: 0023
Create Date: 2026-07-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 保留智能路由页已经配置的模型，但不覆盖 settings 中可能提前写入的新值。
    op.execute(
        sa.text(
            """
            INSERT INTO settings (key, value)
            SELECT 'intent_model', BTRIM(intent_model)
            FROM intent_router_configs
            WHERE id = 1
              AND intent_model IS NOT NULL
              AND BTRIM(intent_model) <> ''
            ON CONFLICT (key) DO NOTHING
            """
        )
    )
    op.drop_column("intent_router_configs", "intent_model")


def downgrade() -> None:
    op.add_column(
        "intent_router_configs",
        sa.Column("intent_model", sa.String(length=255), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE intent_router_configs
            SET intent_model = (
                SELECT NULLIF(BTRIM(value), '')
                FROM settings
                WHERE key = 'intent_model'
            )
            WHERE id = 1
            """
        )
    )
    op.execute(sa.text("DELETE FROM settings WHERE key = 'intent_model'"))
