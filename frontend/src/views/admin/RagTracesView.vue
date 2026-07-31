<template>
  <div class="trace-page">
    <div class="trace-page__inner">
      <PageHeader
        title="调用链路"
        description="按 Trace、会话和时间定位一次问答的意图判断、检索、重排、证据筛选与生成过程。"
      >
        <template #actions>
          <n-button :loading="loading" @click="loadRuns">刷新</n-button>
        </template>
      </PageHeader>

      <SurfaceCard padding="md">
        <div class="trace-filters">
          <n-input
            v-model:value="filters.traceId"
            clearable
            placeholder="Trace ID（至少 4 位）"
            aria-label="按 Trace ID 筛选"
            @keyup.enter="applyFilters"
          />
          <n-input
            v-model:value="filters.conversationId"
            clearable
            placeholder="会话 ID"
            aria-label="按会话 ID 筛选"
            @keyup.enter="applyFilters"
          />
          <n-select
            v-model:value="filters.requestKind"
            :options="requestOptions"
            clearable
            placeholder="全部类型"
            aria-label="按调用类型筛选"
          />
          <n-select
            v-model:value="filters.status"
            :options="statusOptions"
            clearable
            placeholder="全部状态"
            aria-label="按调用状态筛选"
          />
          <n-date-picker
            v-model:value="filters.timeRange"
            type="datetimerange"
            clearable
            :actions="['clear', 'confirm']"
            aria-label="按调用时间筛选"
          />
          <div class="trace-filters__actions">
            <n-button type="primary" @click="applyFilters">查询</n-button>
            <n-button @click="resetFilters">重置</n-button>
          </div>
        </div>
      </SurfaceCard>

      <SurfaceCard padding="none" class="trace-table-card">
        <n-data-table
          remote
          :columns="columns"
          :data="runs"
          :loading="loading"
          :pagination="pagination"
          :scroll-x="1160"
          class="admin-data-table"
        />
      </SurfaceCard>
    </div>

    <AuditDetailDrawer
      v-model:show="showDetail"
      title="调用链详情"
      subtitle="事件按真实执行顺序排列；生产环境默认仅保存指标、摘要和对象 ID。"
      :width="760"
    >
      <div v-if="detailLoading" class="trace-detail-loading">
        <n-spin size="small" />
        <span>正在读取调用链…</span>
      </div>
      <div v-else-if="detail" class="trace-detail">
        <section class="trace-detail__surface">
          <div class="trace-detail__heading">
            <div>
              <div class="trace-detail__label">Trace ID</div>
              <div class="trace-detail__trace-id">{{ detail.trace_id }}</div>
            </div>
            <div class="trace-detail__heading-actions">
              <n-tag :type="statusType(detail.status)" size="small" round>
                {{ statusLabel(detail.status) }}
              </n-tag>
              <n-button text type="primary" size="small" @click="copyTraceId">复制</n-button>
              <n-button
                secondary
                size="small"
                :loading="downloadLoading"
                aria-label="下载当前调用链的 JSON 分析文件"
                title="按安全上限导出当前已保存的调用链数据，可交给 AI 辅助分析"
                @click="showExportConfirm = true"
              >
                下载 AI 分析文件
              </n-button>
            </div>
          </div>
          <dl class="trace-detail__grid">
            <div><dt>调用类型</dt><dd>{{ requestLabel(detail.request_kind) }}</dd></div>
            <div><dt>开始时间</dt><dd>{{ fmtTime(detail.started_at) }}</dd></div>
            <div><dt>总耗时</dt><dd>{{ fmtDuration(detail.duration_ms) }}</dd></div>
            <div>
              <dt>事件数量</dt>
              <dd>{{ detail.event_count }} / {{ detail.observed_event_count }} 个（保存 / 观测）</dd>
            </div>
            <div><dt>操作用户</dt><dd>{{ detail.username || detail.user_id || '—' }}</dd></div>
            <div><dt>会话 ID</dt><dd class="trace-detail__break">{{ detail.conversation_id || '—' }}</dd></div>
            <div><dt>证据状态</dt><dd>{{ evidenceLabel(detail.evidence_status) }}</dd></div>
            <div><dt>知识库 / 命中</dt><dd>{{ detail.selected_kb_count ?? '—' }} / {{ detail.hit_count ?? '—' }}</dd></div>
          </dl>
          <div v-if="detail.input_preview" class="trace-detail__preview">
            <div class="trace-detail__label">问题摘要</div>
            <p>{{ detail.input_preview }}</p>
          </div>
          <div v-else class="trace-detail__privacy">
            当前环境未保存问题正文；可通过指标与对象 ID 完成生产排障。
          </div>
          <n-alert
            v-if="detail.storage_truncated"
            type="warning"
            :show-icon="false"
            class="trace-detail__storage-warning"
          >
            持久化阶段已按安全上限省略 {{ detail.storage_omitted_event_count }} 个事件；
            最终状态仍会优先保留，分析时请结合完整性标记。
          </n-alert>
        </section>

        <section class="trace-detail__surface">
          <div class="trace-detail__section-head">
            <div>
              <h3>执行阶段</h3>
              <p>点击阶段可展开该事件的结构化指标。</p>
            </div>
            <span>已加载 {{ detail.events.length }} / {{ detail.event_count }} 个</span>
          </div>
          <ol class="trace-event-list">
            <li v-for="event in detail.events" :key="event.id" class="trace-event">
              <span class="trace-event__line" aria-hidden="true"></span>
              <span class="trace-event__dot" :class="eventTone(event.event)" aria-hidden="true"></span>
              <button
                type="button"
                class="trace-event__trigger"
                :aria-expanded="expandedEvents.has(event.sequence)"
                @click="toggleEvent(event.sequence)"
              >
                <span class="trace-event__sequence">{{ event.sequence }}</span>
                <span class="trace-event__copy">
                  <strong>{{ eventLabel(event) }}</strong>
                  <small>{{ event.event }} · {{ fmtTime(event.created_at) }}</small>
                </span>
                <span v-if="eventDuration(event)" class="trace-event__duration">{{ eventDuration(event) }}</span>
                <span class="trace-event__chevron" :class="{ 'is-open': expandedEvents.has(event.sequence) }">⌄</span>
              </button>
              <pre v-if="expandedEvents.has(event.sequence)" class="trace-event__payload">{{ prettyPayload(event.payload) }}</pre>
            </li>
          </ol>
          <div v-if="detail.events.length < detail.event_count" class="trace-event-more">
            <n-button :loading="detailEventsLoading" @click="loadMoreEvents">
              加载更多（剩余 {{ detail.event_count - detail.events.length }} 个）
            </n-button>
          </div>
        </section>
      </div>
    </AuditDetailDrawer>

    <AppModal
      v-model:show="showExportConfirm"
      title="下载调用链分析文件"
      width="min(92vw, 520px)"
      :loading="downloadLoading"
    >
      <div class="trace-export-confirm">
        <n-alert
          :type="detail?.content_included ? 'warning' : 'info'"
          :show-icon="false"
        >
          {{ detail?.content_included
            ? '当前调用链包含开发环境已保存的问题、回答或候选正文。上传到外部 AI 前，请先确认符合业务数据安全要求。'
            : '当前环境未保存业务正文，文件仅包含已入库的摘要、哈希、指标和对象 ID。' }}
        </n-alert>
        <p>
          文件会导出已保存的事件时间线、路由判断、召回与重排分数、证据筛选、耗时和版本信息。
          超过安全上限时会优先保留核心阶段，并在 integrity 中标明省略数量；不会额外读取聊天消息、
          文档、模型设置或密钥。
        </p>
      </div>
      <template #footer>
        <n-button :disabled="downloadLoading" @click="showExportConfirm = false">取消</n-button>
        <n-button type="primary" :loading="downloadLoading" @click="downloadAnalysisFile">
          确认下载
        </n-button>
      </template>
    </AppModal>
  </div>
</template>

<script setup>
import { h, onMounted, reactive, ref } from 'vue'
import {
  NAlert,
  NButton,
  NDataTable,
  NDatePicker,
  NInput,
  NSelect,
  NSpin,
  NTag,
  useMessage,
} from 'naive-ui'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import AuditDetailDrawer from '@/components/ui/AuditDetailDrawer.vue'
import AppModal from '@/components/ui/AppModal.vue'
import { downloadRagTrace, getRagTraceDetail, getRagTraces } from '@/api/ragTraces'
import { useAuthStore } from '@/stores/auth'

const message = useMessage()
const authStore = useAuthStore()
const runs = ref([])
const loading = ref(false)
const showDetail = ref(false)
const detailLoading = ref(false)
const detailEventsLoading = ref(false)
const downloadLoading = ref(false)
const showExportConfirm = ref(false)
const detail = ref(null)
const expandedEvents = ref(new Set())

const filters = reactive({
  traceId: '',
  conversationId: '',
  requestKind: null,
  status: null,
  timeRange: null,
})

const requestOptions = [
  { label: '问答请求', value: 'chat' },
  { label: '检索测试', value: 'search_test' },
]
const statusOptions = [
  { label: '执行中', value: 'running' },
  { label: '成功', value: 'success' },
  { label: '失败', value: 'error' },
  { label: '已中断', value: 'interrupted' },
]

const EVENT_LABELS = {
  'chat.request': '接收问答请求',
  'conversation.context_candidates': '整理多轮候选',
  'conversation.context_resolved': '解析多轮上下文',
  'conversation.reference_unresolved': '等待补充追问对象',
  'intent.model_result': '评估意图模型结果',
  'intent.model_error': '意图模型调用失败',
  'intent.contract_compiled': '编译路由执行合同',
  'intent.routing_decision': '确定智能路由策略',
  'intent.clarification_created': '保存待澄清任务',
  'intent.clarification_resolved': '完成待澄清任务',
  'intent.clarification_expired': '待澄清任务已过期',
  'evidence.ambiguity_assessed': '检查证据适用范围',
  'evidence.clarification_required': '要求选择证据范围',
  'evidence.clarification_created': '保存证据范围选项',
  'evidence.clarification_repeated': '重复证据范围选项',
  'evidence.clarification_resolved': '完成证据范围选择',
  'evidence.scope_filter_applied': '应用证据范围过滤',
  'evidence.scope_filter_rejected_candidates': '拦截范围外候选',
  'retrieval.plan': '制定检索计划',
  'retrieval.candidate': '召回候选片段',
  'retrieval.completed': '完成知识库召回',
  'retrieval.channel_error': '召回通道降级',
  'retrieval.error': '知识库召回失败',
  'rerank.candidate': '评估候选证据',
  'rerank.completed': '完成重排',
  'evidence.selection': '筛选回答证据',
  'generation.context': '组装生成上下文',
  'generation.skipped': '跳过回答生成',
  'generation.completed': '完成模型生成',
  'chat.response': '保存回答摘要',
  'chat.cancelled': '问答流已中断',
  'chat.error': '问答执行失败',
  'chat.persistence_error': '回答保存失败',
  'search_test.request': '接收检索测试',
  'search_test.completed': '完成检索测试',
  'search_test.error': '检索测试失败',
}
const EVIDENCE_LABELS = {
  hit: '直接命中', partial: '部分支撑', version_mismatch: '版本不匹配',
  needs_clarification: '等待选择范围', no_hit: '无有效证据',
  unverified: '未验证', skipped: '未检索', error: '检索异常',
}

const dateTimeFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
})

const fmtTime = value => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return dateTimeFormatter.format(date).replaceAll('/', '-').replace(', ', ' ')
}
const fmtDuration = value => {
  if (value === null || value === undefined) return '—'
  if (value < 1000) return `${value} ms`
  return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)} s`
}
const statusLabel = value => ({ running: '执行中', success: '成功', error: '失败', interrupted: '已中断' }[value] || value || '—')
const statusType = value => ({ running: 'info', success: 'success', error: 'error', interrupted: 'warning' }[value] || 'default')
const requestLabel = value => ({ chat: '问答请求', search_test: '检索测试', unknown: '其他调用' }[value] || value || '—')
const requestType = value => ({ chat: 'info', search_test: 'warning', unknown: 'default' }[value] || 'default')
const evidenceLabel = value => EVIDENCE_LABELS[value] || value || '—'
const eventLabel = event => {
  const value = typeof event === 'string' ? event : event?.event
  if (value === 'intent.model_result' || value === 'intent.model_error') {
    const attempt = typeof event === 'string' ? null : event?.payload?.attempt
    if (attempt === 'primary') return `${EVENT_LABELS[value]}（主模型）`
    if (attempt === 'fallback_chat_model') return `${EVENT_LABELS[value]}（对话模型兜底）`
  }
  return EVENT_LABELS[value] || value
}
const canOpenTrace = row => row.content_accessible !== undefined
  ? row.content_accessible
  : (!row.content_included || Boolean(authStore.user?.is_superadmin))

const pagination = reactive({
  page: 1,
  pageSize: 20,
  itemCount: 0,
  showSizePicker: true,
  pageSizes: [10, 20, 50, 100],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: page => { pagination.page = page; loadRuns() },
  onUpdatePageSize: pageSize => { pagination.pageSize = pageSize; pagination.page = 1; loadRuns() },
})

const columns = [
  {
    title: 'Trace ID', key: 'trace_id', width: 260,
    render: row => h('span', { class: 'trace-id-cell' }, row.trace_id),
  },
  {
    title: '类型', key: 'request_kind', width: 102, align: 'center',
    render: row => h(NTag, { type: requestType(row.request_kind), size: 'small', round: true }, () => requestLabel(row.request_kind)),
  },
  {
    title: '状态', key: 'status', width: 92, align: 'center',
    render: row => h(NTag, { type: statusType(row.status), size: 'small', round: true }, () => statusLabel(row.status)),
  },
  { title: '用户', key: 'username', width: 140, ellipsis: { tooltip: true }, render: row => row.username || '—' },
  { title: '阶段', key: 'event_count', width: 78, align: 'center', render: row => row.event_count },
  { title: '证据', key: 'evidence_status', width: 108, align: 'center', render: row => evidenceLabel(row.evidence_status) },
  { title: '耗时', key: 'duration_ms', width: 104, align: 'center', render: row => fmtDuration(row.duration_ms) },
  { title: '开始时间', key: 'started_at', width: 168, align: 'center', render: row => fmtTime(row.started_at) },
  {
    title: '操作', key: 'actions', width: 80, align: 'center', fixed: 'right',
    render: row => {
      const allowed = canOpenTrace(row)
      const button = h(NButton, {
        text: true,
        type: 'primary',
        size: 'small',
        disabled: !allowed,
        onClick: allowed ? () => openDetail(row.trace_id) : undefined,
      }, () => '查看')
      return allowed
        ? button
        : h('span', {
            title: '该调用链包含开发环境业务正文，仅超级管理员可查看',
            class: 'trace-action-disabled',
          }, [button])
    },
  },
]

function queryParams() {
  const params = { page: pagination.page, page_size: pagination.pageSize }
  if (filters.traceId.trim().length >= 4) params.trace_id = filters.traceId.trim()
  if (filters.conversationId.trim()) params.conversation_id = filters.conversationId.trim()
  if (filters.requestKind) params.request_kind = filters.requestKind
  if (filters.status) params.status = filters.status
  if (Array.isArray(filters.timeRange) && filters.timeRange.length === 2) {
    params.started_from = new Date(filters.timeRange[0]).toISOString()
    params.started_to = new Date(filters.timeRange[1]).toISOString()
  }
  return params
}

async function loadRuns() {
  loading.value = true
  try {
    const response = await getRagTraces(queryParams())
    runs.value = response.items || []
    pagination.itemCount = response.total || 0
  } catch (error) {
    message.error(error?.response?.data?.detail || '调用链加载失败')
  } finally {
    loading.value = false
  }
}

function applyFilters() {
  if (filters.traceId.trim() && filters.traceId.trim().length < 4) {
    message.warning('Trace ID 至少输入 4 位')
    return
  }
  pagination.page = 1
  loadRuns()
}

function resetFilters() {
  Object.assign(filters, { traceId: '', conversationId: '', requestKind: null, status: null, timeRange: null })
  pagination.page = 1
  loadRuns()
}

async function openDetail(traceId) {
  detail.value = null
  expandedEvents.value = new Set()
  detailLoading.value = true
  showDetail.value = true
  try {
    detail.value = await getRagTraceDetail(traceId, { event_offset: 0, event_limit: 50 })
  } catch (error) {
    message.error(error?.response?.data?.detail || '调用链详情加载失败')
    showDetail.value = false
  } finally {
    detailLoading.value = false
  }
}

async function loadMoreEvents() {
  if (!detail.value || detailEventsLoading.value) return
  const traceId = detail.value.trace_id
  const eventOffset = detail.value.events.length
  detailEventsLoading.value = true
  try {
    const response = await getRagTraceDetail(traceId, {
      event_offset: eventOffset,
      event_limit: 50,
    })
    if (!detail.value || detail.value.trace_id !== traceId) return
    const knownIds = new Set(detail.value.events.map(event => event.id))
    const appended = (response.events || []).filter(event => !knownIds.has(event.id))
    detail.value = {
      ...response,
      events: [...detail.value.events, ...appended],
    }
  } catch (error) {
    message.error(error?.response?.data?.detail || '调用链事件加载失败')
  } finally {
    detailEventsLoading.value = false
  }
}

function toggleEvent(sequence) {
  const next = new Set(expandedEvents.value)
  if (next.has(sequence)) next.delete(sequence)
  else next.add(sequence)
  expandedEvents.value = next
}

function eventTone(event) {
  if (event.includes('error')) return 'is-error'
  if (
    event === 'chat.cancelled'
    || event === 'intent.clarification_expired'
    || event === 'evidence.clarification_required'
    || event === 'evidence.clarification_repeated'
  ) return 'is-warning'
  if (
    event.endsWith('completed')
    || event === 'chat.response'
    || event === 'intent.clarification_resolved'
    || event === 'evidence.clarification_resolved'
  ) return 'is-success'
  return 'is-info'
}

function eventDuration(event) {
  const value = event.payload?.total_ms ?? event.payload?.elapsed_ms
  return value === null || value === undefined ? '' : fmtDuration(value)
}

function prettyPayload(payload) {
  try { return JSON.stringify(payload, null, 2) } catch { return String(payload) }
}

async function copyTraceId() {
  try {
    await navigator.clipboard.writeText(detail.value.trace_id)
    message.success('Trace ID 已复制')
  } catch {
    message.error('复制失败，请手动选择')
  }
}

async function downloadAnalysisFile() {
  if (!detail.value || downloadLoading.value) return
  const traceId = detail.value.trace_id
  downloadLoading.value = true
  try {
    const response = await downloadRagTrace(traceId)
    const blob = response.data
    const exportTruncated = response.headers?.['x-rag-trace-truncated'] === 'true'
    const omittedEvents = Number(response.headers?.['x-rag-trace-omitted-events'] || 0)
    const downloadUrl = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = downloadUrl
    anchor.download = `rag-trace-${traceId}.json`
    anchor.style.display = 'none'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0)
    showExportConfirm.value = false
    if (exportTruncated) {
      message.warning(`分析文件已下载，按安全上限省略 ${omittedEvents} 个事件`)
    } else {
      message.success('调用链分析文件已下载，事件完整性信息已写入文件')
    }
  } catch (error) {
    let detailMessage = error?.response?.data?.detail
    if (error?.response?.data instanceof Blob) {
      try {
        detailMessage = JSON.parse(await error.response.data.text())?.detail
      } catch {
        detailMessage = null
      }
    }
    message.error(detailMessage || '调用链分析文件下载失败')
  } finally {
    downloadLoading.value = false
  }
}

onMounted(loadRuns)
</script>

<style scoped>
.trace-page {
  height: 100%;
  overflow-y: auto;
  padding: 16px;
  background: var(--ui-bg);
}

.trace-page__inner {
  max-width: 1440px;
  margin: 0 auto;
  display: grid;
  gap: 20px;
}

.trace-filters {
  display: grid;
  grid-template-columns: minmax(190px, 1.1fr) minmax(190px, 1.1fr) 130px 120px minmax(280px, 1.4fr) auto;
  align-items: center;
  gap: 10px;
}
.trace-id-cell {
  display: block;
  color: var(--ui-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  letter-spacing: .01em;
  white-space: nowrap;
}

.trace-filters__actions { display: flex; gap: 8px; }
.trace-table-card { overflow: hidden; }
.trace-action-disabled { display: inline-flex; cursor: not-allowed; }

.trace-detail-loading {
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--ui-text-secondary);
}

.trace-detail { display: grid; gap: 16px; }
.trace-detail__surface {
  padding: 18px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
}

.trace-detail__heading,
.trace-detail__heading-actions,
.trace-detail__section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.trace-detail__heading-actions { align-items: center; justify-content: flex-end; flex-wrap: wrap; }
.trace-detail__label {
  color: var(--ui-text-tertiary);
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.trace-detail__trace-id { margin-top: 5px; color: var(--ui-text); font-family: ui-monospace, monospace; font-size: 13px; overflow-wrap: anywhere; }
.trace-detail__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 24px; margin: 18px 0 0; }
.trace-detail__grid div { min-width: 0; }
.trace-detail__grid dt { color: var(--ui-text-tertiary); font-size: 11px; }
.trace-detail__grid dd { margin: 4px 0 0; color: var(--ui-text); font-size: 13px; line-height: 1.5; }
.trace-detail__break { overflow-wrap: anywhere; }
.trace-detail__preview { margin-top: 18px; padding: 14px; border-radius: var(--ui-radius-control); background: var(--ui-surface-muted); }
.trace-detail__preview p { margin: 6px 0 0; color: var(--ui-text); font-size: 13px; line-height: 1.7; }
.trace-detail__privacy { margin-top: 18px; padding: 12px 14px; border-radius: var(--ui-radius-control); color: var(--ui-text-secondary); background: var(--ui-surface-muted); font-size: 12px; line-height: 1.6; }
.trace-detail__storage-warning { margin-top: 14px; }
.trace-detail__section-head h3 { margin: 0; color: var(--ui-text); font-size: 15px; }
.trace-detail__section-head p { margin: 5px 0 0; color: var(--ui-text-secondary); font-size: 12px; }
.trace-detail__section-head > span { color: var(--ui-text-tertiary); font-size: 12px; white-space: nowrap; }

.trace-event-list { margin: 18px 0 0; padding: 0; list-style: none; }
.trace-event { position: relative; padding: 0 0 12px 24px; }
.trace-event:last-child { padding-bottom: 0; }
.trace-event__line { position: absolute; left: 6px; top: 15px; bottom: -5px; width: 1px; background: var(--ui-divider); }
.trace-event:last-child .trace-event__line { display: none; }
.trace-event__dot { position: absolute; left: 2px; top: 13px; width: 9px; height: 9px; border-radius: 50%; background: var(--ui-info); box-shadow: 0 0 0 3px var(--ui-primary-subtle); }
.trace-event__dot.is-success { background: var(--ui-success); }
.trace-event__dot.is-error { background: var(--ui-danger); }
.trace-event__dot.is-warning { background: var(--ui-warning); }
.trace-event__trigger {
  width: 100%;
  min-height: 42px;
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto 20px;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-control);
  color: var(--ui-text);
  background: transparent;
  text-align: left;
  cursor: pointer;
}
.trace-event__trigger:hover { border-color: var(--ui-border); background: var(--ui-surface-hover); }
.trace-event__trigger:focus-visible { outline: 2px solid var(--ui-focus-outline); outline-offset: 2px; }
.trace-event__sequence { color: var(--ui-text-tertiary); font-family: ui-monospace, monospace; font-size: 11px; }
.trace-event__copy { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.trace-event__copy strong { font-size: 13px; font-weight: 600; }
.trace-event__copy small { color: var(--ui-text-tertiary); font-family: ui-monospace, monospace; font-size: 10px; overflow-wrap: anywhere; }
.trace-event__duration { color: var(--ui-text-secondary); font-size: 11px; white-space: nowrap; }
.trace-event__chevron { color: var(--ui-icon); font-size: 16px; transition: transform .16s ease; }
.trace-event__chevron.is-open { transform: rotate(180deg); }
.trace-event__payload {
  max-height: 360px;
  margin: 6px 0 4px;
  padding: 14px;
  overflow: auto;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-popover);
  color: var(--ui-text-secondary);
  background: var(--ui-bg-subtle);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.trace-event-more { display: flex; justify-content: center; padding-top: 8px; }
.trace-export-confirm { display: grid; gap: 14px; }
.trace-export-confirm p { margin: 0; color: var(--ui-text-secondary); font-size: 13px; line-height: 1.7; }

@media (max-width: 1279px) {
  .trace-filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trace-filters__actions { grid-column: span 2; justify-content: flex-end; }
}

@media (max-width: 639px) {
  .trace-page { padding: 16px 12px; }
  .trace-filters { grid-template-columns: 1fr; }
  .trace-filters__actions { grid-column: auto; justify-content: stretch; }
  .trace-filters__actions :deep(.n-button) { flex: 1; min-height: 40px; }
  .trace-detail__grid { grid-template-columns: 1fr; }
  .trace-detail__heading { flex-direction: column; }
  .trace-detail__heading-actions { width: 100%; justify-content: flex-start; }
  .trace-event__trigger { grid-template-columns: 24px minmax(0, 1fr) 18px; }
  .trace-event__duration { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .trace-event__chevron { transition: none; }
}
</style>
