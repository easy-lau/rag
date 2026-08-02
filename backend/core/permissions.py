"""RBAC capability catalog, dependency rules, and derived navigation metadata.

Only *capabilities* are stored in ``role_permissions``.  ``menu:*`` keys are
derived from effective capabilities for backward-compatible frontend routing;
they must never be persisted or assigned as security permissions.

The module exposes action-level capabilities (for example ``doc:create`` and
``doc:delete``) so roles can receive only the operations they actually need.
New code should consume the structured catalog instead of duplicating labels,
dependencies, or risk classifications in API and UI layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


# ── Derived menu keys (presentation/routing only; never assignable) ────────
MENU_CHAT = "menu:chat"
MENU_KNOWLEDGE = "menu:knowledge"
MENU_DOCUMENTS = "menu:documents"
MENU_SEARCH_TEST = "menu:search_test"
MENU_INTENT_ROUTING = "menu:intent_routing"
MENU_TERMINOLOGY = "menu:terminology"
MENU_SETTINGS = "menu:settings"
MENU_USERS = "menu:users"
MENU_ROLES = "menu:roles"
MENU_LOGIN_LOGS = "menu:login_logs"
MENU_RAG_TRACES = "menu:rag_traces"

DERIVED_MENU_KEYS: tuple[str, ...] = (
    MENU_CHAT,
    MENU_KNOWLEDGE,
    MENU_DOCUMENTS,
    MENU_SEARCH_TEST,
    MENU_INTENT_ROUTING,
    MENU_TERMINOLOGY,
    MENU_SETTINGS,
    MENU_USERS,
    MENU_ROLES,
    MENU_LOGIN_LOGS,
    MENU_RAG_TRACES,
)


# ── Persisted capabilities ────────────────────────────────────────────────
CHAT_USE = "chat:use"
SEARCH_USE = "search:use"
KB_READ = "kb:read"
KB_CREATE = "kb:create"
KB_UPDATE = "kb:update"
KB_DELETE = "kb:delete"
DOC_READ = "doc:read"
DOC_CREATE = "doc:create"
DOC_UPDATE = "doc:update"
DOC_DELETE = "doc:delete"
SETTINGS_READ = "settings:read"
SETTINGS_WRITE = "settings:write"
INTENT_READ = "intent:read"
INTENT_MANAGE = "intent:manage"
TERMINOLOGY_READ = "terminology:read"
TERMINOLOGY_MANAGE = "terminology:manage"
USER_MANAGE = "user:manage"
ROLE_MANAGE = "role:manage"
LOG_READ = "log:read"

# Historical key retained only while old databases are upgraded.  Scope is now
# represented by Role.scope_mode and role_knowledge_bases, not this key.
KB_ACCESS_ALL = "kb:access_all"
LEGACY_KB_WRITE = "kb:write"
LEGACY_DOC_WRITE = "doc:write"

KB_SCOPE_NONE = "none"
KB_SCOPE_SELECTED = "selected"
KB_SCOPE_ALL = "all"
KB_SCOPE_MODES: tuple[str, ...] = (KB_SCOPE_NONE, KB_SCOPE_SELECTED, KB_SCOPE_ALL)

# Stable built-in role identity.  Security-sensitive code must not infer this
# role from its editable/localized display name.
SUPERADMIN_ROLE_CODE = "superadmin"
STANDARD_USER_ROLE_CODE = "standard_user"


CAPABILITY_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": CHAT_USE,
        "group": "workspace",
        "module": "chat",
        "label": "使用问答",
        "description": "发送问答、查看和管理自己的会话。",
        "risk": "medium",
        "requires": (),
    },
    {
        "key": SEARCH_USE,
        "group": "knowledge",
        "module": "search_test",
        "label": "执行检索测试",
        "description": "对授权知识库运行检索测试并读取召回片段。",
        "risk": "medium",
        "requires": (KB_READ,),
    },
    {
        "key": KB_READ,
        "group": "knowledge",
        "module": "knowledge",
        "label": "查看知识库",
        "description": "列出授权知识库及其标签，用于问答、文档和检索选择。",
        "risk": "medium",
        "requires": (),
    },
    {
        "key": KB_CREATE,
        "group": "knowledge",
        "module": "knowledge",
        "label": "新建知识库",
        "description": "创建新的知识库；仅可与全部知识库范围一起授权。",
        "risk": "high",
        "superadmin_only": True,
        "requires": (KB_READ,),
    },
    {
        "key": KB_UPDATE,
        "group": "knowledge",
        "module": "knowledge",
        "label": "编辑知识库",
        "description": "修改授权知识库的名称、说明和外观。",
        "risk": "high",
        "superadmin_only": True,
        "requires": (KB_READ,),
    },
    {
        "key": KB_DELETE,
        "group": "knowledge",
        "module": "knowledge",
        "label": "删除知识库",
        "description": "删除授权范围内不包含文档的知识库。",
        "risk": "high",
        "superadmin_only": True,
        "requires": (KB_READ,),
    },
    {
        "key": DOC_READ,
        "group": "knowledge",
        "module": "documents",
        "label": "查看文档",
        "description": "查看授权知识库中的文档、内容和标签。",
        "risk": "medium",
        "requires": (KB_READ,),
    },
    {
        "key": DOC_CREATE,
        "group": "knowledge",
        "module": "documents",
        "label": "新建文档",
        "description": "在授权知识库中上传文件、图片或新建文本文档。",
        "risk": "high",
        "requires": (DOC_READ, KB_READ),
    },
    {
        "key": DOC_UPDATE,
        "group": "knowledge",
        "module": "documents",
        "label": "编辑文档",
        "description": "编辑授权知识库中的文档内容、标签和启停状态。",
        "risk": "high",
        "requires": (DOC_READ, KB_READ),
    },
    {
        "key": DOC_DELETE,
        "group": "knowledge",
        "module": "documents",
        "label": "删除文档",
        "description": "删除授权知识库中的文档及其检索分块。",
        "risk": "high",
        "requires": (DOC_READ, KB_READ),
    },
    {
        "key": SETTINGS_READ,
        "group": "system",
        "module": "settings",
        "label": "查看系统设置",
        "description": "查看模型、检索策略和站点配置（密钥仅掩码显示）。",
        "risk": "medium",
        "requires": (),
    },
    {
        "key": SETTINGS_WRITE,
        "group": "system",
        "module": "settings",
        "label": "修改系统设置",
        "description": "修改模型、密钥、检索策略和站点品牌配置。",
        "risk": "critical",
        "superadmin_only": True,
        "requires": (SETTINGS_READ,),
    },
    {
        "key": INTENT_READ,
        "group": "knowledge",
        "module": "intent_routing",
        "label": "查看智能路由",
        "description": "查看路由配置、分类、测试结果和运行日志。",
        "risk": "medium",
        "requires": (),
    },
    {
        "key": INTENT_MANAGE,
        "group": "knowledge",
        "module": "intent_routing",
        "label": "管理智能路由",
        "description": "修改路由策略、意图分类和路由反馈。",
        "risk": "high",
        "requires": (INTENT_READ,),
    },
    {
        "key": TERMINOLOGY_READ,
        "group": "knowledge",
        "module": "terminology",
        "label": "查看受控术语",
        "description": "查看授权知识库中已审核的术语、同义关系及其适用范围。",
        "risk": "medium",
        "requires": (KB_READ,),
    },
    {
        "key": TERMINOLOGY_MANAGE,
        "group": "knowledge",
        "module": "terminology",
        "label": "管理受控术语",
        "description": "维护会改变检索和证据语义的术语定义、别名与适用范围。",
        "risk": "critical",
        "superadmin_only": True,
        "requires": (TERMINOLOGY_READ, KB_READ),
    },
    {
        "key": USER_MANAGE,
        "group": "system",
        "module": "users",
        "label": "管理用户",
        "description": "创建、修改、禁用、删除用户及分配允许的角色。",
        "risk": "critical",
        "superadmin_only": True,
        "requires": (),
    },
    {
        "key": ROLE_MANAGE,
        "group": "system",
        "module": "roles",
        "label": "管理角色",
        "description": "创建和维护低风险、可委派角色及其知识库范围。",
        "risk": "critical",
        "superadmin_only": True,
        "requires": (),
    },
    {
        "key": LOG_READ,
        "group": "system",
        "module": "audit_logs",
        "label": "查看审计与调用链",
        "description": "查看登录日志、操作审计和 RAG 调用链诊断信息。",
        "risk": "high",
        "requires": (),
    },
)

CAPABILITY_BY_KEY: dict[str, dict[str, Any]] = {
    item["key"]: item for item in CAPABILITY_DEFINITIONS
}
ASSIGNABLE_CAPABILITIES: tuple[str, ...] = tuple(item["key"] for item in CAPABILITY_DEFINITIONS)
ASSIGNABLE_CAPABILITY_SET = frozenset(ASSIGNABLE_CAPABILITIES)

# Read-only compatibility for a database that has not yet applied migration
# 0020.  Legacy keys are expanded while loading effective permissions, but are
# deliberately absent from the assignable catalog and rejected by role writes.
LEGACY_CAPABILITY_EXPANSIONS: dict[str, frozenset[str]] = {
    LEGACY_KB_WRITE: frozenset({KB_CREATE, KB_UPDATE, KB_DELETE}),
    LEGACY_DOC_WRITE: frozenset({DOC_CREATE, DOC_UPDATE, DOC_DELETE}),
}

# A non-superadmin must not modify an existing role with any of these grants,
# nor create/delegate a role containing them.  All KB mutations retain the old
# high-impact delegation boundary; action-level document operations remain
# delegable within the actor's own capability and KB-scope ceiling.
HIGH_RISK_CAPABILITIES = frozenset({
    KB_CREATE,
    KB_UPDATE,
    KB_DELETE,
    SETTINGS_WRITE,
    TERMINOLOGY_MANAGE,
    USER_MANAGE,
    ROLE_MANAGE,
})
NON_SUPERADMIN_GRANT_FORBIDDEN = frozenset({
    USER_MANAGE,
    ROLE_MANAGE,
    SETTINGS_WRITE,
    TERMINOLOGY_MANAGE,
})

# These capabilities read or mutate KB-backed data.  Their authorization is
# meaningless without a data scope, so a role with ``scope_mode=none`` cannot
# save them.  Chat is deliberately not included: it remains useful as a
# general (non-RAG) conversation role, and intent/settings/audit capabilities
# are similarly independent from knowledge-base data access.
KB_SCOPE_REQUIRED_CAPABILITIES = frozenset({
    SEARCH_USE,
    KB_READ,
    KB_CREATE,
    KB_UPDATE,
    KB_DELETE,
    DOC_READ,
    DOC_CREATE,
    DOC_UPDATE,
    DOC_DELETE,
    TERMINOLOGY_READ,
    TERMINOLOGY_MANAGE,
})


# ``ALL_PERMISSIONS`` remains the effective-permission compatibility export.
# It includes read-only legacy aliases, derived menu keys, and the legacy
# all-scope marker so a true superadmin still behaves as unrestricted for
# cached clients.  Role write APIs accept only ASSIGNABLE_CAPABILITIES.
ALL_PERMISSIONS: list[str] = [
    *ASSIGNABLE_CAPABILITIES,
    *LEGACY_CAPABILITY_EXPANSIONS,
    KB_ACCESS_ALL,
    *DERIVED_MENU_KEYS,
]


# Derivation rules are deliberately capability-based rather than role-name
# based.  A menu is a presentation affordance, never an authorization grant.
MENU_REQUIRE_ANY: dict[str, frozenset[str]] = {
    MENU_CHAT: frozenset({CHAT_USE}),
    MENU_KNOWLEDGE: frozenset({KB_READ, KB_CREATE, KB_UPDATE, KB_DELETE}),
    MENU_DOCUMENTS: frozenset({DOC_READ, DOC_CREATE, DOC_UPDATE, DOC_DELETE}),
    MENU_SEARCH_TEST: frozenset({SEARCH_USE}),
    MENU_INTENT_ROUTING: frozenset({INTENT_READ, INTENT_MANAGE}),
    MENU_TERMINOLOGY: frozenset({TERMINOLOGY_READ, TERMINOLOGY_MANAGE}),
    MENU_SETTINGS: frozenset({SETTINGS_READ, SETTINGS_WRITE}),
    MENU_USERS: frozenset({USER_MANAGE}),
    MENU_ROLES: frozenset({ROLE_MANAGE}),
    MENU_LOGIN_LOGS: frozenset({LOG_READ}),
    MENU_RAG_TRACES: frozenset({LOG_READ}),
}

MENUS: list[dict[str, Any]] = [
    {"key": MENU_CHAT, "route": "chat", "title": "智能问答", "permission": MENU_CHAT, "derived": True},
    {"key": MENU_KNOWLEDGE, "route": "knowledge", "title": "知识库", "permission": MENU_KNOWLEDGE, "derived": True},
    {"key": MENU_DOCUMENTS, "route": "documents", "title": "文档管理", "permission": MENU_DOCUMENTS, "derived": True},
    {"key": MENU_SEARCH_TEST, "route": "search-test", "title": "检索测试", "permission": MENU_SEARCH_TEST, "derived": True},
    {"key": MENU_INTENT_ROUTING, "route": "intent-routing", "title": "智能路由", "permission": MENU_INTENT_ROUTING, "derived": True},
    {"key": MENU_TERMINOLOGY, "route": "terminology", "title": "受控术语", "permission": MENU_TERMINOLOGY, "derived": True},
    {"key": MENU_SETTINGS, "route": "settings", "title": "系统设置", "permission": MENU_SETTINGS, "derived": True},
    {"key": MENU_USERS, "route": "users", "title": "用户管理", "permission": MENU_USERS, "derived": True},
    {"key": MENU_ROLES, "route": "roles", "title": "角色管理", "permission": MENU_ROLES, "derived": True},
    {"key": MENU_LOGIN_LOGS, "route": "audit-logs", "title": "审计日志", "permission": MENU_LOGIN_LOGS, "derived": True},
    {"key": MENU_RAG_TRACES, "route": "rag-traces", "title": "调用链路", "permission": MENU_RAG_TRACES, "derived": True},
]


# Module metadata belongs to the backend catalog as well.  Keeping labels and
# membership here prevents the role editor from re-creating an out-of-date
# permission tree in each client.
MODULE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {"key": "chat", "group": "workspace", "label": "智能问答", "description": "问答会话与检索选择。", "menu": MENU_CHAT},
    {"key": "knowledge", "group": "knowledge", "label": "知识库", "description": "知识库列表和元数据。", "menu": MENU_KNOWLEDGE},
    {"key": "documents", "group": "knowledge", "label": "文档管理", "description": "知识库文档与标签。", "menu": MENU_DOCUMENTS},
    {"key": "search_test", "group": "knowledge", "label": "检索测试", "description": "检索召回与效果测试。", "menu": MENU_SEARCH_TEST},
    {"key": "intent_routing", "group": "knowledge", "label": "智能路由", "description": "意图分类、路由策略和反馈。", "menu": MENU_INTENT_ROUTING},
    {"key": "terminology", "group": "knowledge", "label": "受控术语", "description": "审核术语、同义关系及其知识库适用范围。", "menu": MENU_TERMINOLOGY},
    {"key": "settings", "group": "system", "label": "系统设置", "description": "模型、检索策略和站点配置。", "menu": MENU_SETTINGS},
    {"key": "users", "group": "system", "label": "用户管理", "description": "用户账号与角色分配。", "menu": MENU_USERS},
    {"key": "roles", "group": "system", "label": "角色管理", "description": "角色、能力和知识库范围。", "menu": MENU_ROLES},
    {"key": "audit_logs", "group": "system", "label": "审计与追踪", "description": "登录、操作审计记录与 RAG 调用链诊断。", "menu": MENU_LOGIN_LOGS},
)


ROLE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "code": "standard_user",
        "name": "普通问答用户",
        "description": "仅使用问答；知识库范围由角色分配决定。",
        "scope_mode": KB_SCOPE_SELECTED,
        "capabilities": (CHAT_USE,),
        "is_assignable": True,
    },
    {
        "code": "knowledge_viewer",
        "name": "知识库阅览者",
        "description": "查看授权知识库及文档，不可修改。",
        "scope_mode": KB_SCOPE_SELECTED,
        "capabilities": (KB_READ, DOC_READ),
        "is_assignable": True,
    },
    {
        "code": "knowledge_editor",
        "name": "知识编辑者",
        "description": "新建并维护授权知识库内的文档内容和标签，不含删除权限。",
        "scope_mode": KB_SCOPE_SELECTED,
        "capabilities": (KB_READ, DOC_READ, DOC_CREATE, DOC_UPDATE),
        "is_assignable": True,
    },
    {
        "code": "search_analyst",
        "name": "检索分析员",
        "description": "查看授权知识库并执行检索测试，不可修改文档。",
        "scope_mode": KB_SCOPE_SELECTED,
        "capabilities": (KB_READ, DOC_READ, SEARCH_USE),
        "is_assignable": True,
    },
    {
        "code": "intent_operator",
        "name": "智能路由运营员",
        "description": "查看并维护智能路由配置和分类。",
        "scope_mode": KB_SCOPE_NONE,
        "capabilities": (INTENT_READ, INTENT_MANAGE),
        "is_assignable": True,
    },
    {
        "code": "auditor",
        "name": "审计员",
        "description": "只读查看登录、操作审计日志和 RAG 调用链。",
        "scope_mode": KB_SCOPE_NONE,
        "capabilities": (LOG_READ,),
        "is_assignable": True,
    },
    {
        "code": "platform_operator",
        "name": "平台管理员",
        "description": "管理模型、检索和站点设置，并查看审计日志与调用链。",
        "scope_mode": KB_SCOPE_NONE,
        "capabilities": (SETTINGS_READ, SETTINGS_WRITE, LOG_READ),
        "is_assignable": False,
    },
)


PERMISSION_CATALOG: dict[str, Any] = {
    "version": 5,
    "capabilities": [dict(item) for item in CAPABILITY_DEFINITIONS],
    "groups": (
        {"key": "workspace", "label": "问答工作台"},
        {"key": "knowledge", "label": "知识运营"},
        {"key": "system", "label": "系统管理"},
    ),
    "scope_modes": (
        {"key": KB_SCOPE_NONE, "label": "不访问知识库"},
        {"key": KB_SCOPE_SELECTED, "label": "指定知识库"},
        {"key": KB_SCOPE_ALL, "label": "全部知识库", "risk": "high", "superadmin_only": True},
    ),
}


def _ordered(keys: Iterable[str], order: Iterable[str]) -> list[str]:
    key_set = set(keys)
    return [key for key in order if key in key_set]


def capability_closure(
    keys: Iterable[str],
    *,
    scope_mode: str | None = None,
) -> set[str]:
    """Return the transitive set of known capability dependencies.

    Unknown and non-assignable keys are ignored here so loading legacy role
    rows cannot make authentication fail.  API write paths use
    :func:`normalize_assignable_capabilities`, which rejects those keys.
    """

    supplied = set(keys)
    closure = {key for key in supplied if key in ASSIGNABLE_CAPABILITY_SET}
    for legacy_key, replacements in LEGACY_CAPABILITY_EXPANSIONS.items():
        if legacy_key in supplied:
            # Legacy kb:write was already unable to create a KB unless the
            # role had all scope.  Preserve that effective behavior instead of
            # inventing kb:create for a selected-scope role during rollout.
            if legacy_key == LEGACY_KB_WRITE and scope_mode != KB_SCOPE_ALL:
                closure.update(replacements - {KB_CREATE})
            else:
                closure.update(replacements)
    pending = list(closure)
    while pending:
        key = pending.pop()
        for required in CAPABILITY_BY_KEY[key]["requires"]:
            if required not in closure:
                closure.add(required)
                pending.append(required)
    return closure


def normalize_assignable_capabilities(keys: Iterable[str]) -> set[str]:
    """Validate a role payload and expand its required capabilities.

    Persisting the closure avoids configurations such as ``doc:update`` without
    ``doc:read``.  The caller can compare the returned set with the input for
    audit/detail output if desired.
    """

    supplied = {str(key).strip() for key in keys if str(key).strip()}
    menus = sorted(supplied & set(DERIVED_MENU_KEYS))
    if menus:
        raise ValueError(f"菜单入口为派生权限，不能分配：{', '.join(menus)}")
    if KB_ACCESS_ALL in supplied:
        raise ValueError("kb:access_all 已由知识库范围 scope_mode=all 替代，不能作为权限分配")
    legacy = sorted(supplied & set(LEGACY_CAPABILITY_EXPANSIONS))
    if legacy:
        raise ValueError(
            "旧写权限已拆分，不能继续分配：" + ", ".join(legacy)
        )
    unknown = sorted(supplied - ASSIGNABLE_CAPABILITY_SET)
    if unknown:
        raise ValueError(f"存在未知权限：{', '.join(unknown)}")
    return capability_closure(supplied)


def derive_menus(capabilities: Iterable[str]) -> list[str]:
    """Return menu keys implied by an effective capability set."""

    effective = set(capabilities)
    return [
        menu_key
        for menu_key in DERIVED_MENU_KEYS
        if effective & MENU_REQUIRE_ANY[menu_key]
    ]


def derive_legacy_write_aliases(
    capabilities: Iterable[str],
    *,
    scope_mode: str | None,
) -> list[str]:
    """Derive old frontend aliases only when they are behavior-equivalent.

    Aliases are response compatibility values, never persisted grants and never
    used by current API authorization.  Partial CRUD combinations intentionally
    receive no alias, so an outdated client cannot imply broader controls.
    """

    effective = set(capabilities)
    aliases: list[str] = []
    kb_legacy_actions = (
        {KB_CREATE, KB_UPDATE, KB_DELETE}
        if scope_mode == KB_SCOPE_ALL
        else {KB_UPDATE, KB_DELETE}
        if scope_mode == KB_SCOPE_SELECTED
        else set()
    )
    if kb_legacy_actions and kb_legacy_actions.issubset(effective):
        aliases.append(LEGACY_KB_WRITE)
    if {DOC_CREATE, DOC_UPDATE, DOC_DELETE}.issubset(effective):
        aliases.append(LEGACY_DOC_WRITE)
    return aliases


def effective_permissions(
    raw_keys: Iterable[str],
    *,
    is_superadmin: bool = False,
    scope_mode: str | None = None,
) -> list[str]:
    """Resolve stored grants into capability closure plus derived menu keys."""

    if is_superadmin:
        return list(ALL_PERMISSIONS)
    raw = set(raw_keys)
    capabilities = capability_closure(raw, scope_mode=scope_mode)
    result = _ordered(capabilities, ASSIGNABLE_CAPABILITIES)
    result.extend(derive_legacy_write_aliases(capabilities, scope_mode=scope_mode))
    # Retain the historical marker for a pre-0019 database during rolling
    # deployment.  New roles never persist it.
    if KB_ACCESS_ALL in raw:
        result.append(KB_ACCESS_ALL)
    result.extend(derive_menus(capabilities))
    return result


def normalize_scope_mode(scope_mode: str | None, kb_ids: Iterable[object]) -> tuple[str, set[object]]:
    """Validate the mutually exclusive KB scope representation.

    ``None`` is intentionally inferred for old clients: selected when IDs are
    supplied, none otherwise.  Explicit selected scope must contain at least
    one ID to prevent a role that appears configured but accesses nothing.
    """

    ids = set(kb_ids)
    mode = (scope_mode or (KB_SCOPE_SELECTED if ids else KB_SCOPE_NONE)).strip().lower()
    if mode not in KB_SCOPE_MODES:
        raise ValueError("知识库范围必须是 none、selected 或 all")
    if mode == KB_SCOPE_ALL and ids:
        raise ValueError("全部知识库范围不能同时配置 kb_ids")
    if mode == KB_SCOPE_NONE and ids:
        raise ValueError("不访问知识库范围不能配置 kb_ids")
    if mode == KB_SCOPE_SELECTED and not ids:
        raise ValueError("指定知识库范围至少需要选择一个知识库")
    return mode, ids


def validate_capability_scope(capabilities: Iterable[str], scope_mode: str) -> None:
    """Reject a role whose KB data capabilities have no KB data scope.

    This function is intentionally DB-free so all role write paths and tests
    share exactly the same policy.  Call it after capability closure and scope
    normalization; the latter handles malformed scope names and KB IDs.
    """

    capability_set = set(capabilities)
    if KB_CREATE in capability_set and scope_mode != KB_SCOPE_ALL:
        raise ValueError("kb:create 只能与全部知识库范围 all 一起授权")
    if scope_mode == KB_SCOPE_NONE:
        invalid = sorted(capability_set & KB_SCOPE_REQUIRED_CAPABILITIES)
    else:
        invalid = []
    if invalid:
        raise ValueError(
            "知识库范围为 none 时不能授予知识库数据权限：" + ", ".join(invalid)
        )


def is_high_risk_configuration(capabilities: Iterable[str], scope_mode: str) -> bool:
    return bool(set(capabilities) & HIGH_RISK_CAPABILITIES) or scope_mode == KB_SCOPE_ALL


def non_superadmin_delegation_error(
    actor_capabilities: Iterable[str],
    target_capabilities: Iterable[str],
    target_scope_mode: str,
    *,
    target_is_assignable: bool = True,
) -> str | None:
    """Return a deterministic reason when a non-superadmin cannot delegate.

    Kept pure for reuse by role and user APIs and unit tests.
    """

    target = set(target_capabilities)
    actor = set(actor_capabilities)
    if not target_is_assignable:
        return "目标角色不可分配"
    if target_scope_mode == KB_SCOPE_ALL:
        return "非超级管理员不能授予全部知识库范围"
    forbidden = sorted(target & NON_SUPERADMIN_GRANT_FORBIDDEN)
    if forbidden:
        return f"非超级管理员不能授予高风险权限：{', '.join(forbidden)}"
    if is_high_risk_configuration(target, target_scope_mode):
        return "非超级管理员不能授予或操作高风险角色"
    missing = sorted(target - actor)
    if missing:
        return f"不能授予自己未拥有的权限：{', '.join(missing)}"
    return None


def capability_catalog_payload() -> dict[str, Any]:
    """Return a copy-safe structured payload for role-management clients."""

    modules_by_group: dict[str, list[dict[str, Any]]] = {}
    for module in MODULE_DEFINITIONS:
        item = dict(module)
        item["permissions"] = [
            capability["key"]
            for capability in CAPABILITY_DEFINITIONS
            if capability["module"] == module["key"]
        ]
        modules_by_group.setdefault(module["group"], []).append(item)

    groups = []
    for group in PERMISSION_CATALOG["groups"]:
        item = dict(group)
        item["modules"] = modules_by_group.get(group["key"], [])
        groups.append(item)

    templates = []
    for template in ROLE_TEMPLATES:
        capabilities = list(template["capabilities"])
        templates.append({
            "key": template["code"],
            "code": template["code"],
            "label": template["name"],
            "name": template["name"],
            "description": template["description"],
            "scope_mode": template["scope_mode"],
            "permissions": capabilities,
            # Compatibility for clients that adopted the early internal name.
            "capabilities": capabilities,
            "is_assignable": template["is_assignable"],
        })

    return {
        "version": PERMISSION_CATALOG["version"],
        "capabilities": [dict(item) for item in CAPABILITY_DEFINITIONS],
        "groups": groups,
        "modules": [dict(item) for item in MODULE_DEFINITIONS],
        "scope_modes": [dict(item) for item in PERMISSION_CATALOG["scope_modes"]],
        "templates": templates,
    }
