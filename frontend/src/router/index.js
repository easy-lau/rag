import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { defaultWorkspaceRoute, firstAccessibleAdminRoute } from './menus'

function legacyRedirect(name) {
  return (to) => ({ name, query: to.query, hash: to.hash })
}

function safeInternalRedirect(value) {
  if (typeof value !== 'string') return null
  if (!value.startsWith('/') || value.startsWith('//') || value.startsWith('/login')) return null
  return value
}

const routes = [
  { path: '/login', component: () => import('@/views/LoginView.vue'), name: 'login' },
  { path: '/forbidden', component: () => import('@/views/ForbiddenView.vue'), name: 'forbidden' },
  { path: '/', redirect: '/chat' },
  {
    path: '/',
    component: () => import('@/layout/ChatLayout.vue'),
    children: [
      { path: 'chat', component: () => import('@/views/ChatView.vue'), name: 'chat', meta: { permission: 'menu:chat' } },
    ],
  },
  {
    path: '/admin',
    component: () => import('@/layout/AdminLayout.vue'),
    children: [
      { path: '', redirect: '/admin/knowledge' },
      { path: 'knowledge', component: () => import('@/views/KnowledgeView.vue'), name: 'knowledge', meta: { permission: 'menu:knowledge' } },
      { path: 'documents', component: () => import('@/views/DocumentView.vue'), name: 'documents', meta: { permission: 'menu:documents' } },
      { path: 'search-test', component: () => import('@/views/SearchTestView.vue'), name: 'search-test', meta: { permission: 'menu:search_test' } },
      { path: 'intent-routing', component: () => import('@/views/admin/IntentRoutingView.vue'), name: 'intent-routing', meta: { permission: 'menu:intent_routing' } },
      { path: 'users', component: () => import('@/views/admin/UsersView.vue'), name: 'users', meta: { permission: 'menu:users' } },
      { path: 'roles', component: () => import('@/views/admin/RolesView.vue'), name: 'roles', meta: { permission: 'menu:roles' } },
      { path: 'audit-logs', component: () => import('@/views/admin/AuditLogsView.vue'), name: 'audit-logs', meta: { permission: 'menu:login_logs' } },
      { path: 'rag-traces', component: () => import('@/views/admin/RagTracesView.vue'), name: 'rag-traces', meta: { permission: 'menu:rag_traces' } },
      { path: 'model-management', component: () => import('@/views/admin/ModelManagementView.vue'), name: 'model-management', meta: { permission: 'menu:settings' } },
      { path: 'settings', component: () => import('@/views/SettingsView.vue'), name: 'settings', meta: { permission: 'menu:settings' } },
    ],
  },
  // 保留原路径，避免已有书签和页面内链接失效；查询参数（如 documents?kb=）一并保留。
  { path: '/knowledge', redirect: legacyRedirect('knowledge') },
  { path: '/documents', redirect: legacyRedirect('documents') },
  { path: '/search-test', redirect: legacyRedirect('search-test') },
  { path: '/intent-routing', redirect: legacyRedirect('intent-routing') },
  { path: '/users', redirect: legacyRedirect('users') },
  { path: '/roles', redirect: legacyRedirect('roles') },
  { path: '/audit-logs', redirect: legacyRedirect('audit-logs') },
  { path: '/rag-traces', redirect: legacyRedirect('rag-traces') },
  { path: '/model-management', redirect: legacyRedirect('model-management') },
  { path: '/settings', redirect: legacyRedirect('settings') },
]

const router = createRouter({ history: createWebHistory(), routes })

router.beforeEach(async (to) => {
  const authStore = useAuthStore()

  // 未登录：保留站内深链，登录成功后再由守卫校验最终权限。
  if (!authStore.isAuthenticated) {
    if (to.path === '/login') return true
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  // 已登录但 user 尚未加载（如刷新页面）：先拉取 /auth/me。
  if (!authStore.user) {
    await authStore.fetchMe()
    if (!authStore.isAuthenticated) {
      if (to.path === '/login') return true
      return { name: 'login', query: { redirect: to.fullPath } }
    }
  }

  // 已登录访问登录页：优先回到原本请求的站内页面，否则进入问答工作台。
  if (to.path === '/login') {
    return safeInternalRedirect(to.query.redirect) || defaultWorkspaceRoute(authStore)
  }

  // 单页权限仍按原 menu:* 校验。后台页无权限时跳到首个可用后台页；
  // 普通工作台页无权限时回到默认工作台，避免把两套导航混在一起。
  if (to.meta?.permission && !authStore.hasPerm(to.meta.permission)) {
    if (to.path.startsWith('/admin')) {
      return firstAccessibleAdminRoute(authStore) || defaultWorkspaceRoute(authStore)
    }
    return defaultWorkspaceRoute(authStore)
  }

  return true
})

export default router
