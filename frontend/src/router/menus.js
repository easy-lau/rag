import {
  ChatbubbleEllipsesOutline, LibraryOutline, SearchOutline,
  SettingsOutline, PeopleOutline, ShieldOutline, TimeOutline,
} from '@vicons/ionicons5'

// 侧边栏菜单单一来源：to(路由路径)、label、icon、permission(为空表示公开)、
// match(高亮匹配的额外前缀)。守卫与侧边栏均消费此配置。
export const MENU_ITEMS = [
  { to: '/chat',        label: '问答对话',   icon: ChatbubbleEllipsesOutline, permission: 'menu:chat' },
  { to: '/knowledge',   label: '知识库管理', icon: LibraryOutline, permission: 'menu:knowledge', match: ['/knowledge', '/documents'] },
  { to: '/search-test', label: '检索测试',   icon: SearchOutline, permission: 'menu:search_test' },
  { to: '/users',       label: '用户管理',   icon: PeopleOutline, permission: 'menu:users' },
  { to: '/roles',       label: '角色管理',   icon: ShieldOutline, permission: 'menu:roles' },
  { to: '/login-logs',  label: '登录日志',   icon: TimeOutline, permission: 'menu:login_logs' },
  { to: '/settings',    label: '系统设置',   icon: SettingsOutline, permission: 'menu:settings' },
]

// 当前用户可见的菜单（permission 为空表示公开常显）
export function accessibleMenus(authStore) {
  return MENU_ITEMS.filter(item => !item.permission || authStore.hasPerm(item.permission))
}

// 登录/守卫跳转目标：第一个有权限的菜单；都没有则回退 /login
export function firstAccessibleRoute(authStore) {
  const first = accessibleMenus(authStore).find(item => item.permission)
  return first ? first.to : '/login'
}
