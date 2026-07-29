<template>
  <div>
    <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200">管理操作审计</h3>
        <p class="mt-1 text-xs text-gray-400">列表展示可读摘要；动作码、对象 ID 与完整变更信息在详情中保留。</p>
      </div>

      <!-- 筛选条 -->
      <div class="flex flex-wrap items-center gap-2">
        <n-select
          v-model:value="moduleFilter" :options="moduleOptions"
          placeholder="全部模块" clearable size="small" class="w-40"
          @update:value="onFilterChange"
        />
        <n-input
          v-model:value="usernameFilter" placeholder="按操作人筛选" clearable size="small" class="w-44"
          @keyup.enter="onFilterChange" @clear="onFilterChange"
        />
        <n-button size="small" @click="onFilterChange">筛选</n-button>
      </div>
    </div>

    <n-data-table
      remote
      :columns="columns" :data="logs" :loading="loading"
      :pagination="pagination" :scroll-x="1040"
      class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden"
    />

    <!-- 详情弹窗 -->
    <n-modal v-model:show="showDetail" to="#app" preset="card" title="操作日志详情" style="width: 600px; max-width: 92vw">
      <n-descriptions v-if="current" :column="1" label-placement="left" bordered label-style="width: 96px">
        <n-descriptions-item label="操作人">{{ current.username || '—' }}</n-descriptions-item>
        <n-descriptions-item label="动作">
          <n-tag :type="actionType(current.action)" size="small">{{ actionLabel(current.action) }}</n-tag>
        </n-descriptions-item>
        <n-descriptions-item label="操作对象">{{ targetText(current) }}</n-descriptions-item>
        <n-descriptions-item label="变更明细">
          <div v-if="detailRows(current.detail)" class="space-y-1">
            <div v-for="row in detailRows(current.detail)" :key="row.label" class="text-sm break-all">
              <span class="text-gray-500 dark:text-gray-400">{{ row.label }}：</span>
              <span class="text-gray-700 dark:text-gray-200">{{ row.text }}</span>
            </div>
          </div>
          <pre v-else-if="current.detail" class="text-xs font-mono whitespace-pre-wrap break-all text-gray-600 dark:text-gray-300 m-0">{{ prettyDetail(current.detail) }}</pre>
          <span v-else class="text-gray-400">—</span>
        </n-descriptions-item>
        <n-descriptions-item label="IP 地址">{{ current.ip || '—' }}</n-descriptions-item>
        <n-descriptions-item label="浏览器">{{ parseUA(current.user_agent).browser }}</n-descriptions-item>
        <n-descriptions-item label="操作系统">{{ parseUA(current.user_agent).os }}</n-descriptions-item>
        <n-descriptions-item label="操作时间">{{ fmtTime(current.created_at) }}</n-descriptions-item>
        <n-descriptions-item label="对象 ID">
          <span class="text-xs font-mono break-all text-gray-500 dark:text-gray-400">{{ current.target_id || '—' }}</span>
        </n-descriptions-item>
        <n-descriptions-item label="日志 ID">
          <span class="text-xs font-mono break-all text-gray-500 dark:text-gray-400">{{ current.id }}</span>
        </n-descriptions-item>
        <n-descriptions-item label="事件代码">
          <span class="text-xs font-mono break-all text-gray-500 dark:text-gray-400">{{ current.action || '—' }}</span>
        </n-descriptions-item>
      </n-descriptions>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, h } from 'vue'
import { NDataTable, NTag, NButton, NModal, NSelect, NInput, NDescriptions, NDescriptionsItem } from 'naive-ui'
import { getOperationLogs } from '@/api/operationLogs'
import { parseUA } from '@/utils/ua'
import { useUiStore } from '@/stores/ui'

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
  confidence_threshold: '置信度阈值', fallback_intent_code: '兜底意图',
  allow_general_chat: '通用问答开关', examples: '示例问题', action: '路由动作',
  priority: '优先级', code: '意图编码', feedback: '反馈结果', file_type: '文件类型',
}
const INTENT_ACTION_LABELS = {
  retrieve: '知识库检索', chat: '通用回答', writing: '写作 / 润色', system_help: '系统使用帮助',
}
const FEEDBACK_LABELS = { correct: '标注为正确', incorrect: '标注为有误' }

const actionLabel = (a) => ACTION_LABELS[a] || a || '—'
const targetLabel = (t) => TARGET_LABELS[t] || t || '—'
const actionType = (a) => {
  if (!a) return 'default'
  if (a.includes('delete')) return 'error'
  if (a.includes('create')) return 'success'
  if (a.includes('update') || a.includes('upload') || a.includes('password')) return 'warning'
  return 'info'
}
const fmtTime = (v) => v ? new Date(v).toLocaleString('zh-CN') : '—'
const prettyDetail = (d) => {
  try { return JSON.stringify(d, null, 2) } catch { return String(d) }
}

// 单个字段值的友好显示
function fmtFieldVal(field, v) {
  if (field === 'is_active') return v ? '启用' : '禁用'
  if (field === 'enabled') return v ? '开启' : '关闭'
  if (field === 'action') return INTENT_ACTION_LABELS[v] || String(v)
  if (field === 'feedback') return FEEDBACK_LABELS[v] || String(v)
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
  return rows ? rows.map(item => `${item.label}：${item.text}`).join('；') : ''
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
  { title: '时间', key: 'created_at', width: 172, align: 'center', render: r => fmtTime(r.created_at) },
  { title: '操作人', key: 'username', width: 96, align: 'center', ellipsis: { tooltip: true } },
  {
    title: '动作', key: 'action', width: 122, align: 'center',
    render: r => h(NTag, { type: actionType(r.action), size: 'small', style: { whiteSpace: 'nowrap' } }, () => actionLabel(r.action))
  },
  {
    title: '对象', key: 'target', width: 180, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => targetText(r)
  },
  {
    title: '变更 / 结果', key: 'summary', minWidth: 240, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => summaryText(r) || h('span', { class: 'text-gray-300 dark:text-gray-600' }, '—')
  },
  { title: 'IP', key: 'ip', width: 130, align: 'center', render: r => r.ip || '—' },
  {
    title: '操作', key: 'actions', width: 96, align: 'center',
    render: row => h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openDetail(row) }, () => '查看详细')
  },
]
columns.forEach(c => {
  if (!c.titleAlign) c.titleAlign = 'center'
})

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
