<template>
  <div class="space-y-4">
    <!-- 筛选条 -->
    <div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <p class="max-w-xl text-xs leading-5 text-slate-500 dark:text-slate-400">
        列表只保留可扫读的摘要；完整动作代码、对象 ID 与变更明细可在右侧详情查看。
      </p>
      <div class="flex flex-wrap items-center gap-2">
        <n-select
          v-model:value="moduleFilter" :options="moduleOptions"
          placeholder="全部模块" clearable size="small" class="w-40 max-w-full"
          @update:value="onFilterChange"
        />
        <n-input
          v-model:value="usernameFilter" placeholder="按操作人筛选" clearable size="small" class="w-44 max-w-full"
          @keyup.enter="onFilterChange" @clear="onFilterChange"
        />
        <n-button size="small" @click="onFilterChange">筛选</n-button>
      </div>
    </div>

    <n-data-table
      remote
      :columns="columns" :data="logs" :loading="loading"
      :pagination="pagination" :scroll-x="ui.isCompact ? 940 : 1060"
      class="admin-data-table"
    />

    <AuditDetailDrawer v-model:show="showDetail" title="操作日志详情" subtitle="完整事件信息仅用于审计和问题定位。">
      <div v-if="current" class="audit-detail space-y-4">
        <section class="audit-detail__surface">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="audit-detail__label">操作对象</div>
              <div class="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-100">{{ targetText(current) }}</div>
            </div>
            <n-tag :type="actionType(current.action)" size="small" round>{{ actionLabel(current.action) }}</n-tag>
          </div>
          <dl class="audit-detail__grid mt-5">
            <div><dt>操作人</dt><dd>{{ current.username || '—' }}</dd></div>
            <div><dt>操作时间</dt><dd class="tabular-nums">{{ fmtTime(current.created_at) }}</dd></div>
            <div><dt>IP 地址</dt><dd>{{ current.ip || '—' }}</dd></div>
            <div><dt>浏览器 / 系统</dt><dd>{{ parseUA(current.user_agent).browser }} · {{ parseUA(current.user_agent).os }}</dd></div>
          </dl>
        </section>

        <section class="audit-detail__surface">
          <h3 class="audit-detail__section-title">变更明细</h3>
          <dl v-if="detailRows(current.detail)" class="audit-detail__changes">
            <div v-for="row in detailRows(current.detail)" :key="row.label">
              <dt>{{ row.label }}</dt>
              <dd>{{ row.text }}</dd>
            </div>
          </dl>
          <pre v-else-if="current.detail" class="audit-detail__json">{{ prettyDetail(current.detail) }}</pre>
          <p v-else class="text-sm text-slate-400">本次操作没有可展示的字段变更。</p>
        </section>

        <section class="audit-detail__surface">
          <h3 class="audit-detail__section-title">审计标识</h3>
          <dl class="audit-detail__identifiers">
            <div><dt>事件代码</dt><dd>{{ current.action || '—' }}</dd></div>
            <div><dt>对象 ID</dt><dd>{{ current.target_id || '—' }}</dd></div>
            <div><dt>日志 ID</dt><dd>{{ current.id || '—' }}</dd></div>
          </dl>
        </section>
      </div>
    </AuditDetailDrawer>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, h } from 'vue'
import { NDataTable, NTag, NButton, NSelect, NInput } from 'naive-ui'
import { getOperationLogs } from '@/api/operationLogs'
import { parseUA } from '@/utils/ua'
import { useUiStore } from '@/stores/ui'
import AuditDetailDrawer from '@/components/ui/AuditDetailDrawer.vue'

const ui = useUiStore()
const logs = ref([])
const loading = ref(false)
const showDetail = ref(false)
const current = ref(null)

const moduleFilter = ref(null)
const usernameFilter = ref('')

const moduleOptions = [
  { label: '用户', value: 'user' },
  { label: '角色', value: 'role' },
  { label: '知识库', value: 'kb' },
  { label: '文档', value: 'doc' },
  { label: '智能路由', value: 'intent_router' },
  { label: '系统设置', value: 'settings' },
  { label: '账户安全', value: 'auth' },
]

// 动作码 → 中文
const ACTION_LABELS = {
  'user.create': '创建用户', 'user.update': '修改用户', 'user.delete': '删除用户',
  'role.create': '创建角色', 'role.update': '修改角色', 'role.delete': '删除角色',
  'kb.create': '创建知识库', 'kb.update': '修改知识库', 'kb.delete': '删除知识库',
  'doc.upload': '上传文档', 'doc.upload_image': '上传图片', 'doc.create_text': '新建文本文档',
  'doc.update': '编辑文档', 'doc.delete': '删除文档',
  'settings.update': '修改系统设置',
  'settings.connection_test': '测试模型连接',
  'settings.model_list': '获取模型列表',
  'auth.change_password': '修改密码',
  'intent_router.config.update': '更新路由策略',
  'intent_router.category.create': '新增意图分类',
  'intent_router.category.update': '修改意图分类',
  'intent_router.category.delete': '删除意图分类',
  'intent_router.log.feedback': '标注路由结果',
}
const TARGET_LABELS = {
  user: '用户', role: '角色', knowledge_base: '知识库',
  document: '文档', settings: '系统设置', conversation: '会话',
  intent_router_config: '智能路由策略', intent_category: '意图分类',
  intent_route_log: '路由记录',
}
// 变更字段 → 中文
const FIELD_LABELS = {
  display_name: '显示名', role: '角色', is_active: '状态', password: '密码',
  name: '名称', description: '描述', permissions: '权限', kb_ids: '可访问知识库',
  enabled: '启用状态', mode: '判定模式', intent_model: '意图模型',
  rerank_model: '检索重排模型',
  confidence_threshold: '置信度阈值', fallback_intent_code: '兜底意图',
  allow_general_chat: '通用问答开关', examples: '示例问题', action: '路由动作',
  priority: '优先级', code: '意图编码', feedback: '反馈结果', file_type: '文件类型',
  service: '模型类型', model: '模型名称', host: '服务地址', ok: '测试结果',
  latency_ms: '耗时（毫秒）', error_code: '错误标识', model_count: '模型数量',
}
const INTENT_ACTION_LABELS = {
  retrieve: '知识库检索', chat: '通用回答', writing: '写作 / 润色', system_help: '系统使用帮助',
}
const FEEDBACK_LABELS = { correct: '标注为正确', incorrect: '标注为有误' }

const dateTimeFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
})

const actionLabel = (a) => ACTION_LABELS[a] || a || '—'
const targetLabel = (t) => TARGET_LABELS[t] || t || '—'
const actionType = (a) => {
  if (!a) return 'default'
  if (a.includes('delete')) return 'error'
  if (a.includes('create')) return 'success'
  if (a.includes('update') || a.includes('upload') || a.includes('password')) return 'warning'
  return 'info'
}
const fmtTime = (v) => {
  if (!v) return '—'
  const date = new Date(v)
  if (Number.isNaN(date.getTime())) return String(v)
  const parts = Object.fromEntries(dateTimeFormatter.formatToParts(date)
    .filter(part => part.type !== 'literal')
    .map(part => [part.type, part.value]))
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`
}

const timeParts = (v) => {
  const formatted = fmtTime(v)
  const [date = formatted, time = ''] = formatted.split(' ')
  return { date, time }
}
const prettyDetail = (d) => {
  try { return JSON.stringify(d, null, 2) } catch { return String(d) }
}

// 单个字段值的友好显示
function fmtFieldVal(field, v) {
  if (field === 'is_active') return v ? '启用' : '禁用'
  if (field === 'enabled') return v ? '开启' : '关闭'
  if (field === 'action') return INTENT_ACTION_LABELS[v] || String(v)
  if (field === 'feedback') return FEEDBACK_LABELS[v] || String(v)
  if (field === 'ok') return v ? '成功' : '失败'
  if (v === null || v === '' || v === undefined) return '空'
  if (v === true) return '是'
  if (v === false) return '否'
  return String(v)
}

const fieldLabel = field => FIELD_LABELS[field] || field

// 把 detail.changes 解析成 [{label, text}] —— 展示"怎么改的"
function changeRows(detail) {
  const c = detail && detail.changes
  if (!c) return null
  const rows = []
  for (const [field, val] of Object.entries(c)) {
    const label = fieldLabel(field)
    if (val && typeof val === 'object' && ('added' in val || 'removed' in val)) {
      const parts = []
      if (val.added && val.added.length) parts.push('新增：' + val.added.join('、'))
      if (val.removed && val.removed.length) parts.push('移除：' + val.removed.join('、'))
      rows.push({ label, text: parts.join('；') || '—' })
    } else if (val && typeof val === 'object' && ('from' in val || 'to' in val)) {
      rows.push({ label, text: `${fmtFieldVal(field, val.from)} → ${fmtFieldVal(field, val.to)}` })
    } else {
      rows.push({ label, text: fmtFieldVal(field, val) })
    }
  }
  return rows
}

// 非 diff 类型的 detail 同样转成可读行：智能路由的配置、分类与反馈都属于这种结构。
function detailRows(detail) {
  if (!detail || typeof detail !== 'object') return null
  const changes = changeRows(detail)
  if (changes) return changes

  const rows = []
  if (Array.isArray(detail.changed)) {
    rows.push({ label: '修改项', text: detail.changed.map(fieldLabel).join('、') || '—' })
  }

  for (const [field, value] of Object.entries(detail)) {
    if (field === 'changed') continue
    if (Array.isArray(value)) {
      rows.push({ label: fieldLabel(field), text: value.map(item => String(item)).join('、') || '—' })
    } else if (value && typeof value === 'object') {
      rows.push({ label: fieldLabel(field), text: prettyDetail(value) })
    } else {
      rows.push({ label: fieldLabel(field), text: fmtFieldVal(field, value) })
    }
  }
  return rows.length ? rows : null
}

// 列表里的一行变更摘要（紧凑）
function summaryText(row) {
  const rows = detailRows(row.detail)
  if (!rows?.length) return ''
  const compact = rows.slice(0, 2)
    .map(item => `${item.label}：${compactText(item.text, 42)}`)
    .join(' · ')
  return rows.length > 2 ? `${compact} 等 ${rows.length} 项` : compact
}

function compactText(value, maxLength) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text
}

function targetText(row) {
  const type = targetLabel(row.target_type)
  const name = row.target_name || (row.target_type === 'intent_category' ? row.detail?.code : '')
  return name ? `${type} · ${name}` : type
}

const pagination = reactive({
  page: 1,
  pageSize: 10,
  itemCount: 0,
  showSizePicker: true,
  pageSizes: [10, 20, 30, 50],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (p) => { pagination.page = p; loadLogs() },
  onUpdatePageSize: (ps) => { pagination.pageSize = ps; pagination.page = 1; loadLogs() },
})

const columns = [
  {
    title: '时间', key: 'created_at', width: 158, align: 'center', titleAlign: 'center',
    render: row => {
      const time = timeParts(row.created_at)
      return h('div', { class: 'audit-time' }, [
        h('span', { class: 'audit-time__date' }, time.date),
        h('span', { class: 'audit-time__value' }, time.time),
      ])
    },
  },
  { title: '操作人', key: 'username', width: 112, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  {
    title: '动作', key: 'action', width: 126, align: 'center', titleAlign: 'center',
    render: r => h(NTag, { type: actionType(r.action), size: 'small', style: { whiteSpace: 'nowrap' } }, () => actionLabel(r.action))
  },
  {
    title: '对象', key: 'target', width: 178, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => targetText(r)
  },
  {
    title: '变更摘要', key: 'summary', minWidth: 280, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => summaryText(r) || h('span', { class: 'text-slate-300 dark:text-slate-600' }, '—')
  },
  { title: 'IP', key: 'ip', width: 130, align: 'center', titleAlign: 'center', render: r => r.ip || '—' },
  {
    title: '操作', key: 'actions', width: 88, align: 'center', titleAlign: 'center',
    render: row => h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openDetail(row) }, () => '详情')
  },
]

function openDetail(row) {
  current.value = row
  showDetail.value = true
}

function onFilterChange() {
  pagination.page = 1
  loadLogs()
}

onMounted(loadLogs)

async function loadLogs() {
  loading.value = true
  try {
    const params = { page: pagination.page, page_size: pagination.pageSize }
    if (moduleFilter.value) params.action = moduleFilter.value
    if (usernameFilter.value.trim()) params.username = usernameFilter.value.trim()
    const data = await getOperationLogs(params)
    logs.value = data.items
    pagination.itemCount = data.total
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.audit-time {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  color: var(--ui-text-secondary, #64748b);
  font-size: 12px;
  line-height: 1.45;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.audit-time__date { color: var(--ui-text-secondary, #64748b); }
.audit-time__value { color: var(--ui-text, #1e293b); }

.audit-detail__surface {
  padding: 16px;
  background: var(--ui-surface, #ffffff);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-card, 16px);
}

.audit-detail__label,
.audit-detail dt {
  color: var(--ui-text-tertiary, #94a3b8);
  font-size: 12px;
  line-height: 1.4;
}

.audit-detail__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.audit-detail dd {
  min-width: 0;
  margin: 5px 0 0;
  overflow-wrap: anywhere;
  color: var(--ui-text, #1e293b);
  font-size: 13px;
  line-height: 1.55;
}

.audit-detail__section-title {
  margin: 0 0 12px;
  color: var(--ui-text, #1e293b);
  font-size: 13px;
  font-weight: 650;
}

.audit-detail__changes {
  display: grid;
  gap: 10px;
}

.audit-detail__changes > div {
  display: grid;
  grid-template-columns: minmax(92px, 0.38fr) minmax(0, 1fr);
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--ui-border, #e2e8f0);
}

.audit-detail__changes > div:first-child {
  padding-top: 0;
  border-top: 0;
}

.audit-detail__changes dd { margin-top: 0; }

.audit-detail__json {
  max-height: 320px;
  margin: 0;
  padding: 12px;
  overflow: auto;
  color: var(--ui-text-secondary, #64748b);
  background: var(--ui-surface-muted, #f8fafc);
  border-radius: var(--ui-radius-control, 10px);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.audit-detail__identifiers {
  display: grid;
  gap: 10px;
}

.audit-detail__identifiers > div {
  display: grid;
  grid-template-columns: 84px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
}

.audit-detail__identifiers dd {
  margin: 0;
  color: var(--ui-text-secondary, #64748b);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}

@media (max-width: 639px) {
  .audit-detail__grid { grid-template-columns: 1fr; gap: 12px; }
  .audit-detail__changes > div { grid-template-columns: 1fr; gap: 4px; }
}
</style>
