"""权限目录：菜单权限 + 操作权限的字符串常量定义。

权限统一用字符串 key，存入 role_permissions 表；可配置角色只需勾选已有 key，
无需做权限项 CRUD。本模块供角色分配与前端菜单消费。
"""

# ── 菜单权限（控制侧边栏可见性 + 路由可访问）──────────────────────
MENU_CHAT = "menu:chat"
MENU_KNOWLEDGE = "menu:knowledge"
MENU_DOCUMENTS = "menu:documents"
MENU_SEARCH_TEST = "menu:search_test"
MENU_SETTINGS = "menu:settings"
MENU_USERS = "menu:users"
MENU_ROLES = "menu:roles"
MENU_LOGIN_LOGS = "menu:login_logs"

# ── 操作权限（控制接口/动作）──────────────────────────────────────
CHAT_USE = "chat:use"
SEARCH_USE = "search:use"
KB_READ = "kb:read"
KB_WRITE = "kb:write"
DOC_READ = "doc:read"
DOC_WRITE = "doc:write"
SETTINGS_READ = "settings:read"
SETTINGS_WRITE = "settings:write"
USER_MANAGE = "user:manage"
ROLE_MANAGE = "role:manage"
LOG_READ = "log:read"
KB_ACCESS_ALL = "kb:access_all"  # 可访问全部 KB，绕过授权表

# 全部权限 key 汇总
ALL_PERMISSIONS: list[str] = [
    # 菜单权限
    MENU_CHAT,
    MENU_KNOWLEDGE,
    MENU_DOCUMENTS,
    MENU_SEARCH_TEST,
    MENU_SETTINGS,
    MENU_USERS,
    MENU_ROLES,
    MENU_LOGIN_LOGS,
    # 操作权限
    CHAT_USE,
    SEARCH_USE,
    KB_READ,
    KB_WRITE,
    DOC_READ,
    DOC_WRITE,
    SETTINGS_READ,
    SETTINGS_WRITE,
    USER_MANAGE,
    ROLE_MANAGE,
    LOG_READ,
    KB_ACCESS_ALL,
]

# 菜单清单：供角色分配与前端菜单消费
# key=菜单权限 key，route=前端路由名，title=中文标题，permission=该菜单对应权限 key
MENUS: list[dict] = [
    {"key": MENU_CHAT, "route": "chat", "title": "智能问答", "permission": MENU_CHAT},
    {"key": MENU_KNOWLEDGE, "route": "knowledge", "title": "知识库", "permission": MENU_KNOWLEDGE},
    {"key": MENU_DOCUMENTS, "route": "documents", "title": "文档管理", "permission": MENU_DOCUMENTS},
    {"key": MENU_SEARCH_TEST, "route": "search-test", "title": "检索测试", "permission": MENU_SEARCH_TEST},
    {"key": MENU_SETTINGS, "route": "settings", "title": "系统设置", "permission": MENU_SETTINGS},
    {"key": MENU_USERS, "route": "users", "title": "用户管理", "permission": MENU_USERS},
    {"key": MENU_ROLES, "route": "roles", "title": "角色管理", "permission": MENU_ROLES},
    {"key": MENU_LOGIN_LOGS, "route": "login-logs", "title": "登录日志", "permission": MENU_LOGIN_LOGS},
]
