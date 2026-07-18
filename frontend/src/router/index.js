import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '@/layout/AppLayout.vue'
import { useAuthStore } from '@/stores/auth'
import { firstAccessibleRoute } from './menus'

const routes = [
  { path: '/login', component: () => import('@/views/LoginView.vue'), name: 'login' },
  {
    path: '/',
    component: AppLayout,
    redirect: '/chat',
    children: [
      { path: 'chat',        component: () => import('@/views/ChatView.vue'),             name: 'chat',        meta: { permission: 'menu:chat' } },
      { path: 'knowledge',   component: () => import('@/views/KnowledgeView.vue'),        name: 'knowledge',   meta: { permission: 'menu:knowledge' } },
      { path: 'documents',   component: () => import('@/views/DocumentView.vue'),         name: 'documents',   meta: { permission: 'menu:documents' } },
      { path: 'search-test', component: () => import('@/views/SearchTestView.vue'),       name: 'search-test', meta: { permission: 'menu:search_test' } },
      { path: 'users',       component: () => import('@/views/admin/UsersView.vue'),      name: 'users',       meta: { permission: 'menu:users' } },
      { path: 'roles',       component: () => import('@/views/admin/RolesView.vue'),      name: 'roles',       meta: { permission: 'menu:roles' } },
      { path: 'audit-logs',  component: () => import('@/views/admin/AuditLogsView.vue'),   name: 'audit-logs',  meta: { permission: 'menu:login_logs' } },
      { path: 'settings',    component: () => import('@/views/SettingsView.vue'),         name: 'settings',    meta: { permission: 'menu:settings' } },
    ]
  }
]

const router = createRouter({ history: createWebHistory(), routes })

router.beforeEach(async (to) => {
  const authStore = useAuthStore()

  // 未登录（无 token）：除 /login 外一律跳登录页
  if (!authStore.isAuthenticated) {
    return to.path === '/login' ? true : { path: '/login' }
  }

  // 已登录但 user 尚未加载（如刷新页面）：先拉取 /auth/me
  if (!authStore.user) {
    await authStore.fetchMe()
    // fetchMe 失败会清空 token，则视为未登录
    if (!authStore.isAuthenticated) {
      return to.path === '/login' ? true : { path: '/login' }
    }
  }

  // 已登录访问 /login：跳到第一个可见菜单
  if (to.path === '/login') {
    return firstAccessibleRoute(authStore)
  }

  // 路由设了 permission 且当前用户无权限：跳第一个可见菜单（无则 /login）
  if (to.meta?.permission && !authStore.hasPerm(to.meta.permission)) {
    return firstAccessibleRoute(authStore)
  }

  return true
})

export default router
