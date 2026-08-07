import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const adminLayout = readFileSync(new URL('../src/layout/AdminLayout.vue', import.meta.url), 'utf8')
const router = readFileSync(new URL('../src/router/index.js', import.meta.url), 'utf8')
const adminViews = [
  '../src/views/KnowledgeView.vue',
  '../src/views/DocumentView.vue',
  '../src/views/SearchTestView.vue',
  '../src/views/admin/IntentRoutingView.vue',
  '../src/views/admin/DashboardView.vue',
  '../src/views/admin/UsersView.vue',
  '../src/views/admin/RolesView.vue',
  '../src/views/admin/AuditLogsView.vue',
  '../src/views/admin/RagTracesView.vue',
  '../src/views/admin/ModelManagementView.vue',
  '../src/views/SettingsView.vue',
]

test('管理后台顶栏使用当前路由标题', () => {
  assert.match(adminLayout, /\{\{ pageTitle \}\}/)
  assert.match(adminLayout, /route\.meta\?\.title/)
  for (const title of [
    '知识库管理', '文档管理', '检索测试', '智能路由', '数据看板', '用户管理',
    '角色管理', '审计日志', '调用链路', '模型管理', '系统设置',
  ]) {
    assert.ok(router.includes(`title: '${title}'`), `路由缺少页面标题: ${title}`)
  }
})

test('后台页面不重复渲染说明标题并占满内容宽度', () => {
  for (const path of adminViews) {
    const source = readFileSync(new URL(path, import.meta.url), 'utf8')
    assert.doesNotMatch(source, /<PageHeader/)
    assert.doesNotMatch(source, /max-w-(?:5xl|6xl)/)
    assert.doesNotMatch(source, /admin-page-toolbar/)
  }
})
