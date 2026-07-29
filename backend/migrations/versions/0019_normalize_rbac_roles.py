"""normalize role capabilities and knowledge-base scope

Revision ID: 0019
Revises: 0018
Create Date: 2026-07-29

``menu:*`` used to be persisted together with real API capabilities, and
``kb:access_all`` represented a role's data scope.  The normalized model keeps
only capabilities in ``role_permissions`` and stores the KB scope on the role.
"""

from alembic import op
import sqlalchemy as sa


revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep server defaults: roles may also be created by administration scripts,
    # not only through the application ORM.
    op.add_column("roles", sa.Column("code", sa.String(length=64), nullable=True))
    op.add_column(
        "roles",
        sa.Column(
            "scope_mode",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'none'"),
        ),
    )
    op.add_column(
        "roles",
        sa.Column(
            "is_assignable",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )

    conn = op.get_bind()

    # A normal installation has one superadmin-backed system role.  The legacy
    # role may no longer be attached to the original account, so the immutable
    # seed name is retained only as a migration-time fallback.  Runtime policy
    # uses the stable code assigned below and never relies on a display name.
    # Every matching role is made non-assignable; only one receives the unique
    # canonical code.
    conn.execute(
        sa.text(
            """
            UPDATE roles AS role
            SET is_assignable = false
            WHERE role.is_system IS TRUE
              AND (
                  role.name = '超级管理员'
                  OR EXISTS (
                      SELECT 1
                      FROM users AS usr
                      WHERE usr.is_superadmin IS TRUE
                        AND usr.role_id = role.id
                  )
              )
            """
        )
    )
    conn.execute(
        sa.text(
            """
            WITH canonical_superadmin AS (
                SELECT role.id
                FROM roles AS role
                WHERE role.is_system IS TRUE
                  AND (
                      role.name = '超级管理员'
                      OR EXISTS (
                          SELECT 1
                          FROM users AS usr
                          WHERE usr.is_superadmin IS TRUE
                            AND usr.role_id = role.id
                      )
                  )
                ORDER BY
                    CASE WHEN EXISTS (
                        SELECT 1
                        FROM users AS usr
                        WHERE usr.is_superadmin IS TRUE
                          AND usr.role_id = role.id
                    ) THEN 0 ELSE 1 END,
                    CASE WHEN role.name = '超级管理员' THEN 0 ELSE 1 END,
                    role.created_at NULLS LAST,
                    role.id
                LIMIT 1
            )
            UPDATE roles AS role
            SET code = 'superadmin', is_assignable = false
            FROM canonical_superadmin AS canonical
            WHERE role.id = canonical.id
            """
        )
    )

    # The legacy schema creates exactly one other system role (普通用户).  Prefer
    # that name, but fall back deterministically to another non-superadmin
    # system role so renamed legacy installs remain migratable.  Extra system
    # roles deliberately keep a NULL code: assigning all of them the same code
    # would violate the new unique constraint and silently collapse their roles.
    conn.execute(
        sa.text(
            """
            WITH canonical_standard_user AS (
                SELECT role.id
                FROM roles AS role
                WHERE role.is_system IS TRUE
                  AND role.code IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM users AS usr
                      WHERE usr.is_superadmin IS TRUE
                        AND usr.role_id = role.id
                  )
                ORDER BY
                    CASE WHEN role.name = '普通用户' THEN 0 ELSE 1 END,
                    role.created_at NULLS LAST,
                    role.id
                LIMIT 1
            )
            UPDATE roles AS role
            SET code = 'standard_user'
            FROM canonical_standard_user AS canonical
            WHERE role.id = canonical.id
            """
        )
    )

    # Scope precedence preserves the old effective access rule: access-all is
    # broader than any individually assigned KB.  Rows for all-scope roles are
    # then removed because ``scope_mode=all`` and explicit kb_ids are mutually
    # exclusive in the new model.
    conn.execute(
        sa.text(
            """
            UPDATE roles AS role
            SET scope_mode = CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM role_permissions AS permission
                    WHERE permission.role_id = role.id
                      AND permission.permission_key = 'kb:access_all'
                ) THEN 'all'
                WHEN EXISTS (
                    SELECT 1
                    FROM role_knowledge_bases AS role_kb
                    WHERE role_kb.role_id = role.id
                ) THEN 'selected'
                ELSE 'none'
            END
            """
        )
    )
    conn.execute(
        sa.text(
            """
            DELETE FROM role_knowledge_bases AS role_kb
            USING roles AS role
            WHERE role_kb.role_id = role.id
              AND role.scope_mode = 'all'
            """
        )
    )

    # Navigation is now derived from effective capabilities.  The old global
    # access key is represented by scope_mode, so neither is assignable any more.
    conn.execute(
        sa.text(
            """
            DELETE FROM role_permissions
            WHERE permission_key LIKE 'menu:%'
               OR permission_key = 'kb:access_all'
            """
        )
    )

    # The old built-in ordinary-user role inherited global KB search/listing.
    # Retain its sensible baseline capability (chat:use) while removing the two
    # KB-reading capabilities that contradict the new least-privilege template.
    conn.execute(
        sa.text(
            """
            DELETE FROM role_permissions AS permission
            USING roles AS role
            WHERE permission.role_id = role.id
              AND role.code = 'standard_user'
              AND permission.permission_key IN ('kb:read', 'search:use')
            """
        )
    )

    # Old custom roles could carry KB capabilities while having neither an
    # all-scope marker nor an explicit KB row.  Their effective legacy data
    # access was empty; remove those unusable grants so the migrated role is a
    # valid ``scope_mode=none`` configuration without broadening access.
    conn.execute(
        sa.text(
            """
            DELETE FROM role_permissions AS permission
            USING roles AS role
            WHERE permission.role_id = role.id
              AND role.scope_mode = 'none'
              AND permission.permission_key IN (
                  'search:use', 'kb:read', 'kb:write', 'doc:read', 'doc:write'
              )
            """
        )
    )

    op.create_unique_constraint("uq_roles_code", "roles", ["code"])
    op.create_check_constraint(
        "ck_roles_scope_mode",
        "roles",
        "scope_mode IN ('none', 'selected', 'all')",
    )
    op.create_check_constraint(
        "ck_roles_superadmin_not_assignable",
        "roles",
        "code IS DISTINCT FROM 'superadmin' OR is_assignable IS FALSE",
    )


def downgrade() -> None:
    # Rebuild the legacy all-scope marker from the normalized source of truth.
    # UUIDs are deterministic and need no database extension.
    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            INSERT INTO role_permissions (id, role_id, permission_key)
            SELECT
                CAST(md5(CAST(role.id AS text) || ':kb:access_all') AS uuid),
                role.id,
                'kb:access_all'
            FROM roles AS role
            WHERE role.scope_mode = 'all'
            ON CONFLICT (role_id, permission_key) DO NOTHING
            """
        )
    )

    # Reverse the explicit standard-user capability normalization while the
    # stable code is still available.  Its selected KB rows were retained by
    # upgrade; historical all-scope variants receive kb:access_all above.
    conn.execute(
        sa.text(
            """
            INSERT INTO role_permissions (id, role_id, permission_key)
            SELECT
                CAST(md5(CAST(role.id AS text) || ':' || capability.permission_key) AS uuid),
                role.id,
                capability.permission_key
            FROM roles AS role
            CROSS JOIN (VALUES ('kb:read'), ('search:use')) AS capability(permission_key)
            WHERE role.code = 'standard_user'
            ON CONFLICT (role_id, permission_key) DO NOTHING
            """
        )
    )

    # 0018 clients require persisted menu keys.  Recreate the menu implied by
    # each remaining capability (including dependency-equivalent write keys).
    conn.execute(
        sa.text(
            """
            WITH menu_sources(menu_key, capability_key) AS (
                VALUES
                    ('menu:chat', 'chat:use'),
                    ('menu:knowledge', 'kb:read'),
                    ('menu:knowledge', 'kb:write'),
                    ('menu:knowledge', 'search:use'),
                    ('menu:knowledge', 'doc:read'),
                    ('menu:knowledge', 'doc:write'),
                    ('menu:documents', 'doc:read'),
                    ('menu:documents', 'doc:write'),
                    ('menu:search_test', 'search:use'),
                    ('menu:intent_routing', 'intent:read'),
                    ('menu:intent_routing', 'intent:manage'),
                    ('menu:settings', 'settings:read'),
                    ('menu:settings', 'settings:write'),
                    ('menu:users', 'user:manage'),
                    ('menu:roles', 'role:manage'),
                    ('menu:login_logs', 'log:read')
            )
            INSERT INTO role_permissions (id, role_id, permission_key)
            SELECT DISTINCT
                CAST(md5(CAST(role.id AS text) || ':' || source.menu_key) AS uuid),
                role.id,
                source.menu_key
            FROM roles AS role
            JOIN role_permissions AS permission ON permission.role_id = role.id
            JOIN menu_sources AS source
              ON source.capability_key = permission.permission_key
            ON CONFLICT (role_id, permission_key) DO NOTHING
            """
        )
    )

    # Exact pre-upgrade menu rows and unusable scope-none grants cannot be
    # reconstructed, but the statements above restore the legacy effective
    # all-scope and navigation contracts for valid normalized roles.
    op.drop_constraint("ck_roles_superadmin_not_assignable", "roles", type_="check")
    op.drop_constraint("ck_roles_scope_mode", "roles", type_="check")
    op.drop_constraint("uq_roles_code", "roles", type_="unique")
    op.drop_column("roles", "is_assignable")
    op.drop_column("roles", "scope_mode")
    op.drop_column("roles", "code")
