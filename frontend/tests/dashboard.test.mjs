import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const dashboardView = readFileSync(
  new URL('../src/views/admin/DashboardView.vue', import.meta.url),
  'utf8',
)
const dashboardApi = readFileSync(
  new URL('../src/api/dashboard.js', import.meta.url),
  'utf8',
)
const menus = readFileSync(
  new URL('../src/router/menus.js', import.meta.url),
  'utf8',
)

test('数据看板页面覆盖核心运营维度', () => {
  for (const label of ['用户总数', '知识库', '文章总数', '问答次数', '知识分块', '回答质量', '每日问答趋势', '证据状态分布']) {
    assert.ok(dashboardView.includes(label), `缺少维度文案: ${label}`)
  }
})

test('核心指标使用可区分的知识资产强调色', () => {
  assert.match(dashboardView, /label="知识库"[\s\S]*?tone="violet"/)
  assert.match(dashboardView, /metric-card--violet/)
  assert.match(dashboardView, /--ui-accent-violet/)
})

test('用户排行榜独占整行且系统运营卡片在下一行响应式排列', () => {
  assert.match(dashboardView, /\.operations-grid \{ display: grid; grid-template-columns: minmax\(0, 1fr\);/)
  assert.match(dashboardView, /\.operations-side \{ display: grid; grid-template-columns: repeat\(2, minmax\(0, 1fr\)\); gap: 16px; align-items: stretch; \}/)
  assert.match(dashboardView, /@media \(max-width: 767px\)[\s\S]*?\.operations-side \{ grid-template-columns: 1fr; \}/)
})

test('登录安全使用四个真实聚合维度组成二乘二网格', () => {
  for (const label of ['登录成功', '登录失败', '登录账号', '失败来源 IP']) {
    assert.ok(dashboardView.includes(label), `缺少登录安全维度: ${label}`)
  }
  assert.match(dashboardView, /security\.login_users/)
  assert.match(dashboardView, /security\.failed_sources/)
  assert.match(dashboardView, /\.security-grid \{ display: grid; grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/)
})

test('用户问答 Top 10 同时展示质量、性能与最近活跃度', () => {
  for (const label of ['问答次数', '证据命中率', '平均响应', '最近问答']) {
    assert.ok(dashboardView.includes(label), `缺少用户排行维度: ${label}`)
  }
  assert.match(dashboardView, /row\.hit_rate/)
  assert.match(dashboardView, /row\.avg_duration_ms/)
  assert.match(dashboardView, /row\.last_active_at/)
  assert.match(dashboardView, /:scroll-x="760"/)
})

test('数据看板包含 AI 报告入口与新增运营维度', () => {
  for (const label of ['AI 分析报告', '下载 Markdown', '登录安全', '管理操作 Top', '平均命中片段']) {
    assert.ok(dashboardView.includes(label), `缺少维度文案: ${label}`)
  }
  assert.match(dashboardView, /AppModal/)
  assert.match(dashboardView, /downloadReport/)
  assert.match(dashboardView, /ai-dashboard-report-\$\{days\.value\}d\.md/)
})

test('数据看板按运营层级重组概览、问答、质量与系统运营', () => {
  for (const heading of ['核心概览', '问答运营', '模型资源消耗', '内容与回答质量', '用户与系统运营']) {
    assert.ok(dashboardView.includes(heading), `缺少看板分区: ${heading}`)
  }
  assert.doesNotMatch(dashboardView, /AI 运营洞察/)
  assert.doesNotMatch(dashboardView, /class="ai-report-card"/)
  assert.match(dashboardView, /activeRate/)
  assert.match(dashboardView, /readyRate/)
  assert.match(dashboardView, /dailyAverage/)
  assert.match(dashboardView, /avg_tokens_per_qa/)
})

test('每日趋势使用面积折线图，证据状态使用环形分布图', () => {
  assert.match(dashboardView, /import \{ BarChart, LineChart, PieChart \} from 'echarts\/charts'/)
  assert.match(dashboardView, /type: 'line'/)
  assert.match(dashboardView, /type: 'pie'/)
  assert.match(dashboardView, /evidence-legend/)
})

test('数据看板明确区分系统命中率与人工正确率', () => {
  assert.match(dashboardView, /代理指标/)
  assert.match(dashboardView, /证据命中/)
})

test('AI 报告只发送聚合数字，不包含正文', () => {
  assert.match(dashboardView, /只发送统计数字（不含问题正文、文档与会话内容）/u)
})

test('数据看板 API 调用聚合与 AI 报告接口', () => {
  assert.match(dashboardApi, /\/admin\/dashboard\/overview/)
  assert.match(dashboardApi, /\/admin\/dashboard\/report/)
  assert.match(dashboardApi, /params: \{ days \}/)
})

test('数据看板菜单挂在知识运营组并使用 menu:dashboard 权限', () => {
  assert.match(menus, /to: '\/admin\/dashboard'/)
  assert.match(menus, /permission: 'menu:dashboard'/)
  assert.match(menus, /group: 'knowledge'/)
})
