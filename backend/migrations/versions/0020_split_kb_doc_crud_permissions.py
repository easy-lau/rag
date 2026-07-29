"""split knowledge-base and document write capabilities by CRUD action

Revision ID: 0020
Revises: 0019
Create Date: 2026-07-29

Historical ``doc:write`` covered create, update, and delete together.  Legacy
``kb:write`` covered all three only for all-scope roles; selected-scope roles
were already unable to create knowledge bases.  Upgrade preserves those exact
effective behaviors without creating an invalid ``selected + kb:create`` role.
"""

from alembic import op
import sqlalchemy as sa


revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            WITH replacements(legacy_key, action_key) AS (
                VALUES
                    ('kb:write', 'kb:create'),
                    ('kb:write', 'kb:update'),
                    ('kb:write', 'kb:delete'),
                    ('doc:write', 'doc:create'),
                    ('doc:write', 'doc:update'),
                    ('doc:write', 'doc:delete')
            )
            INSERT INTO role_permissions (id, role_id, permission_key)
            SELECT
                CAST(md5(CAST(permission.role_id AS text) || ':' || replacement.action_key) AS uuid),
                permission.role_id,
                replacement.action_key
            FROM role_permissions AS permission
            JOIN replacements AS replacement
              ON replacement.legacy_key = permission.permission_key
            JOIN roles AS role
              ON role.id = permission.role_id
            WHERE replacement.action_key != 'kb:create'
               OR role.scope_mode = 'all'
            ON CONFLICT (role_id, permission_key) DO NOTHING
            """
        )
    )
    conn.execute(
        sa.text(
            """
            DELETE FROM role_permissions
            WHERE permission_key IN ('kb:write', 'doc:write')
            """
        )
    )


def downgrade() -> None:
    conn = op.get_bind()

    # The coarse legacy model cannot express an arbitrary partial action set.
    # Refuse a lossy downgrade instead of silently dropping access or turning a
    # create/update-only role into full delete access.  Operators can restore a
    # pre-migration backup or first normalize roles into one of the exact legacy
    # sets described by the error message.
    conn.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM roles AS role
                    LEFT JOIN role_permissions AS permission
                      ON permission.role_id = role.id
                     AND permission.permission_key IN (
                         'kb:create', 'kb:update', 'kb:delete'
                     )
                    GROUP BY role.id, role.scope_mode
                    HAVING
                        (
                            role.scope_mode = 'all'
                            AND COUNT(DISTINCT permission.permission_key) NOT IN (0, 3)
                        )
                        OR (
                            role.scope_mode = 'selected'
                            AND NOT (
                                COUNT(DISTINCT permission.permission_key) = 0
                                OR (
                                    COUNT(DISTINCT permission.permission_key) = 2
                                    AND COUNT(*) FILTER (
                                        WHERE permission.permission_key IN ('kb:update', 'kb:delete')
                                    ) = 2
                                )
                            )
                        )
                        OR (
                            role.scope_mode = 'none'
                            AND COUNT(DISTINCT permission.permission_key) != 0
                        )
                ) THEN
                    RAISE EXCEPTION
                        'Cannot downgrade 0020: partial knowledge-base CRUD grants cannot be represented by kb:write';
                END IF;

                IF EXISTS (
                    SELECT 1
                    FROM role_permissions
                    WHERE permission_key IN ('doc:create', 'doc:update', 'doc:delete')
                    GROUP BY role_id
                    HAVING COUNT(DISTINCT permission_key) NOT IN (0, 3)
                ) THEN
                    RAISE EXCEPTION
                        'Cannot downgrade 0020: partial document CRUD grants cannot be represented by doc:write';
                END IF;
            END
            $$
            """
        )
    )

    # Exact reversible sets: all-scope KB roles own all three actions;
    # selected-scope KB roles own update+delete because legacy creation was
    # already blocked by scope; document roles own all three actions.
    conn.execute(
        sa.text(
            """
            WITH kb_complete AS (
                SELECT permission.role_id, 'kb:write' AS legacy_key
                FROM role_permissions AS permission
                JOIN roles AS role ON role.id = permission.role_id
                WHERE permission.permission_key IN ('kb:create', 'kb:update', 'kb:delete')
                GROUP BY permission.role_id, role.scope_mode
                HAVING
                    (
                        role.scope_mode = 'all'
                        AND COUNT(DISTINCT permission.permission_key) = 3
                    )
                    OR (
                        role.scope_mode = 'selected'
                        AND COUNT(DISTINCT permission.permission_key) = 2
                        AND COUNT(*) FILTER (
                            WHERE permission.permission_key IN ('kb:update', 'kb:delete')
                        ) = 2
                    )
            ), doc_complete AS (
                SELECT permission.role_id, 'doc:write' AS legacy_key
                FROM role_permissions AS permission
                WHERE permission.permission_key IN ('doc:create', 'doc:update', 'doc:delete')
                GROUP BY permission.role_id
                HAVING COUNT(DISTINCT permission.permission_key) = 3
            ), complete_grants AS (
                SELECT role_id, legacy_key FROM kb_complete
                UNION ALL
                SELECT role_id, legacy_key FROM doc_complete
            )
            INSERT INTO role_permissions (id, role_id, permission_key)
            SELECT
                CAST(md5(CAST(grant_row.role_id AS text) || ':' || grant_row.legacy_key) AS uuid),
                grant_row.role_id,
                grant_row.legacy_key
            FROM complete_grants AS grant_row
            ON CONFLICT (role_id, permission_key) DO NOTHING
            """
        )
    )
    conn.execute(
        sa.text(
            """
            DELETE FROM role_permissions
            WHERE permission_key IN (
                'kb:create', 'kb:update', 'kb:delete',
                'doc:create', 'doc:update', 'doc:delete'
            )
            """
        )
    )
