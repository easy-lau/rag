import {
  ChatbubbleEllipsesOutline, LibraryOutline, DocumentTextOutline, SearchOutline,
  SettingsOutline, PeopleOutline, ShieldOutline, TimeOutline, GitNetworkOutline,
  HardwareChipOutline, LanguageOutline, PulseOutline,
} from '@vicons/ionicons5'

// 问答工作台与管理后台使用两套独立菜单。权限仍以既有 menu:* 为准；
// 这里仅决定入口可见性和路由回退，不能替代后端接口的操作权限校验。
export const WORKSPACE_MENU_ITEMS = [
  { to: '/chat', label: '问答对话', icon: ChatbubbleEllipsesOutline, permission: 'menu:chat' },
]

export const ADMIN_MENU_GROUPS = [
  { key: 'knowledge', label: '知识运营' },
  { key: 'system', label: '系统管理' },
]

export const ADMIN_MENU_ITEMS = [
  { to: '/admin/knowledge', label: '知识库管理', icon: LibraryOutline, permission: 'menu:knowledge', group: 'knowledge', match: ['/admin/knowledge'] },
  { to: '/admin/documents', label: '文档管理', icon: DocumentTextOutline, permission: 'menu:documents', group: 'knowledge', match: ['/admin/documents'] },
  { to: '/admin/search-test', label: '检索测试', icon: SearchOutline, permission: 'menu:search_test', group: 'knowledge' },
  { to: '/admin/intent-routing', label: '智能路由', icon: GitNetworkOutline, permission: 'menu:intent_routing', group: 'knowledge' },
  { to: '/admin/terminology', label: '受控术语', icon: LanguageOutline, permission: 'menu:terminology', group: 'knowledge' },
  { to: '/admin/users', label: '用户管理', icon: PeopleOutline, permission: 'menu:users', group: 'system' },
  { to: '/admin/roles', label: '角色管理', icon: ShieldOutline, permission: 'menu:roles', group: 'system' },
  { to: '/admin/audit-logs', label: '审计日志', icon: TimeOutline, permission: 'menu:login_logs', group: 'system' },
  { to: '/admin/rag-traces', label: '调用链路', icon: PulseOutline, permission: 'menu:rag_traces', group: 'system' },
  { to: '/admin/model-management', label: '模型管理', icon: HardwareChipOutline, permission: 'menu:settings', group: 'system' },
  { to: '/admin/settings', label: '系统设置', icon: SettingsOutline, permission: 'menu:settings', group: 'system' },
]

// 兼容已有导入；新代码应按场景使用 workspace/admin 专属函数。
export const MENU_ITEMS = [...WORKSPACE_MENU_ITEMS, ...ADMIN_MENU_ITEMS]

function filterAccessible(items, authStore) {
  return items.filter(item => !item.permission || authStore.hasPerm(item.permission))
}

export function accessibleWorkspaceMenus(authStore) {
  return filterAccessible(WORKSPACE_MENU_ITEMS, authStore)
}

export function accessibleAdminMenus(authStore) {
  return filterAccessible(ADMIN_MENU_ITEMS, authStore)
}

export function accessibleMenus(authStore) {
  return filterAccessible(MENU_ITEMS, authStore)
}

export function hasAdminAccess(authStore) {
  return accessibleAdminMenus(authStore).length > 0
}

export function firstAccessibleAdminRoute(authStore) {
  return accessibleAdminMenus(authStore)[0]?.to || null
}

// 默认落到工作台，而不是依赖菜单数组的排列顺序。理论上每位可登录用户
// 都应具备 menu:chat；极端的仅后台角色才退回其首个后台页。
export function defaultWorkspaceRoute(authStore) {
  if (authStore.hasPerm('menu:chat')) return '/chat'
  return firstAccessibleAdminRoute(authStore) || '/forbidden'
}

// 兼容登录页等旧调用；后续应使用语义更明确的 defaultWorkspaceRoute。
export function firstAccessibleRoute(authStore) {
  return defaultWorkspaceRoute(authStore)
}
