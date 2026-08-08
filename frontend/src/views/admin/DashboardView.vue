<template>
  <div class="dashboard-page">
    <div class="dashboard-container">
      <SurfaceCard v-if="!canRead" padding="lg" class="permission-empty">
        <div class="permission-empty__icon">
          <n-icon :size="26"><LockClosedOutline /></n-icon>
        </div>
        <div class="permission-empty__title">暂无查看数据看板的权限</div>
        <p>请联系管理员为当前角色授予 <code>dashboard:read</code> 权限。</p>
      </SurfaceCard>

      <n-spin v-else :show="loading">
        <div class="dashboard-content">
          <section aria-labelledby="overview-heading">
            <div class="section-heading section-heading--controls">
              <div>
                <h2 id="overview-heading">核心概览</h2>
                <p>从规模、活跃度和内容沉淀快速判断当前运营状态</p>
              </div>
              <div class="section-heading__actions">
                <span v-if="generatedAt" class="section-heading__meta">数据更新于 {{ generatedAt }}</span>
                <div class="dashboard-toolbar">
                  <n-radio-group v-model:value="days" size="small" @update:value="onDaysChange">
                    <n-radio-button :value="7">近 7 天</n-radio-button>
                    <n-radio-button :value="30">近 30 天</n-radio-button>
                  </n-radio-group>
                  <n-button secondary size="small" :loading="loading" @click="loadOverview">
                    <template #icon><n-icon><RefreshOutline /></n-icon></template>
                    刷新
                  </n-button>
                  <n-button type="primary" size="small" :loading="reportLoading" @click="openReport">
                    <template #icon><n-icon><SparklesOutline /></n-icon></template>
                    AI 分析报告
                  </n-button>
                </div>
              </div>
            </div>

            <div class="metric-grid">
              <MetricCard
                label="用户总数"
                :value="scale.users"
                related-label="活跃用户"
                :related-value="scale.active_users"
                :progress="activeRate"
                :hint="`近 ${days} 天活跃率 ${formatPercent(activeRate)}`"
                tone="primary"
                :icon="PeopleOutline"
              />
              <MetricCard
                label="知识库"
                :value="scale.knowledge_bases"
                related-label="文章总数"
                :related-value="scale.documents"
                :hint="`近 ${days} 天新增 ${formatNumber(scale.new_documents)} 篇文章`"
                tone="violet"
                :icon="LibraryOutline"
              />
              <MetricCard
                label="问答次数"
                :value="qa.total"
                related-label="日均问答"
                :related-value="dailyAverage"
                :hint="`覆盖 ${formatNumber(qa.daily.length)} 个有问答的日期`"
                tone="success"
                :icon="ChatbubbleEllipsesOutline"
              />
              <MetricCard
                label="知识分块"
                :value="scale.chunks"
                related-label="篇均分块"
                :related-value="chunksPerDocument"
                :progress="readyRate"
                :hint="`文章就绪率 ${formatPercent(readyRate)}`"
                tone="warning"
                :icon="LayersOutline"
              />
            </div>
          </section>

          <section aria-labelledby="activity-heading">
            <div class="section-heading">
              <div>
                <h2 id="activity-heading">问答运营</h2>
                <p>关注使用量变化与回答依据是否充分</p>
              </div>
            </div>

            <div class="analytics-grid">
              <SurfaceCard padding="lg" class="chart-card trend-card">
                <div class="card-heading">
                  <div>
                    <h3>每日问答趋势</h3>
                    <p>近 {{ days }} 天问答量与单日平均响应耗时</p>
                  </div>
                  <div class="trend-summary" aria-label="趋势摘要">
                    <div>
                      <span>日均</span>
                      <strong>{{ formatNumber(dailyAverage) }}</strong>
                    </div>
                    <div>
                      <span>峰值</span>
                      <strong>{{ formatNumber(peakDay.count) }}</strong>
                    </div>
                    <div>
                      <span>峰值日期</span>
                      <strong>{{ peakDay.label }}</strong>
                    </div>
                  </div>
                </div>
                <div class="chart-stage chart-stage--trend">
                  <div v-show="qa.daily.length" ref="trendEl" class="dashboard-chart" />
                  <div v-if="!qa.daily.length" class="chart-empty">
                    <n-icon :size="24"><TrendingUpOutline /></n-icon>
                    <span>当前周期暂无问答趋势数据</span>
                  </div>
                </div>
              </SurfaceCard>

              <SurfaceCard padding="lg" class="chart-card evidence-card">
                <div class="card-heading">
                  <div>
                    <h3>证据状态分布</h3>
                    <p>回答调用链的证据判定构成</p>
                  </div>
                </div>
                <div v-if="evidenceEntries.length" class="evidence-layout">
                  <div ref="evidenceEl" class="evidence-chart" />
                  <div class="evidence-legend" aria-label="证据状态明细">
                    <div v-for="item in evidenceEntries" :key="item.key" class="evidence-legend__item">
                      <span class="evidence-legend__dot" :style="{ background: item.color }" />
                      <span class="evidence-legend__label">{{ item.label }}</span>
                      <strong>{{ formatNumber(item.count) }}</strong>
                      <span class="evidence-legend__rate">{{ formatPercent(item.rate) }}</span>
                    </div>
                  </div>
                </div>
                <div v-else class="chart-empty chart-empty--evidence">
                  <n-icon :size="24"><AnalyticsOutline /></n-icon>
                  <span>当前周期暂无证据状态数据</span>
                </div>
              </SurfaceCard>
            </div>
          </section>

          <section aria-labelledby="token-heading">
            <div class="section-heading">
              <div>
                <h2 id="token-heading">模型资源消耗</h2>
                <p>跟踪模型调用的 Token 使用趋势与各阶段消耗构成</p>
              </div>
            </div>

            <SurfaceCard padding="lg" class="token-card">
              <div class="card-heading token-card__heading">
                <div>
                  <h3>Token 消耗趋势</h3>
                  <p>包含意图路由、查询分析和回答生成中已返回 usage 的模型调用</p>
                </div>
                <div class="token-chart-legend" aria-label="Token 图例">
                  <span><i class="token-chart-legend__input" />输入 Token</span>
                  <span><i class="token-chart-legend__output" />输出 Token</span>
                </div>
              </div>

              <div class="token-summary-grid">
                <TokenStat label="Token 总消耗" :value="tokens.total_tokens" />
                <TokenStat label="输入 Token" :value="tokens.prompt_tokens" />
                <TokenStat label="输出 Token" :value="tokens.completion_tokens" />
                <TokenStat label="每次问答均耗" :value="tokens.avg_tokens_per_qa" />
              </div>

              <div class="token-layout">
                <div class="token-chart-stage">
                  <div v-show="tokens.daily.length" ref="tokenEl" class="token-chart" />
                  <div v-if="!tokens.daily.length" class="chart-empty token-chart-empty">
                    <n-icon :size="24"><PulseOutline /></n-icon>
                    <span>当前周期暂无可计量的 Token 使用数据</span>
                  </div>
                </div>

                <div class="token-stage-panel">
                  <div class="token-stage-panel__heading">
                    <div>
                      <h4>调用阶段构成</h4>
                      <p>{{ formatNumber(tokens.measured_calls) }} 次已计量模型调用</p>
                    </div>
                  </div>
                  <div v-if="tokenStageEntries.length" class="token-stage-list">
                    <div v-for="item in tokenStageEntries" :key="item.stage" class="token-stage-item">
                      <div class="token-stage-item__meta">
                        <span>{{ item.label }}</span>
                        <strong>{{ formatNumber(item.total_tokens) }}</strong>
                      </div>
                      <div class="token-stage-item__track" aria-hidden="true">
                        <span :style="{ width: formatPercent(item.rate) }" />
                      </div>
                      <div class="token-stage-item__foot">
                        <span>{{ formatNumber(item.measured_calls) }} 次调用</span>
                        <span>{{ formatPercent(item.rate) }}</span>
                      </div>
                    </div>
                  </div>
                  <div v-else class="compact-empty token-stage-empty">暂无阶段构成数据</div>
                </div>
              </div>

              <p class="token-note">
                Token 数据来自模型供应商返回的 usage；未返回 usage 的调用不纳入总量，调用链数据最长保留 30 天。
              </p>
            </SurfaceCard>
          </section>

          <section aria-labelledby="quality-heading">
            <div class="section-heading">
              <div>
                <h2 id="quality-heading">内容与回答质量</h2>
                <p>把可用内容、回答结果和性能放在同一层级对照</p>
              </div>
            </div>

            <div class="quality-grid">
              <SurfaceCard padding="lg" class="quality-card">
                <div class="card-heading">
                  <div>
                    <h3>回答质量</h3>
                    <p>系统证据状态代理指标 · 近 {{ days }} 天</p>
                  </div>
                  <div class="quality-score">
                    <strong>{{ formatPercent(quality.hit_rate) }}</strong>
                    <span>证据命中</span>
                  </div>
                </div>
                <div class="status-list">
                  <StatusRow
                    label="证据命中"
                    :count="evidenceHitCount"
                    :total="qa.total"
                    :rate="quality.hit_rate"
                    tone="success"
                  />
                  <StatusRow
                    label="需要澄清"
                    :count="qualityByEvidence.needs_clarification || 0"
                    :total="qa.total"
                    :rate="quality.clarify_rate"
                    tone="warning"
                  />
                  <StatusRow
                    label="无答案"
                    :count="noAnswerCount"
                    :total="qa.total"
                    :rate="quality.no_answer_rate"
                    tone="error"
                  />
                  <div class="inline-note">
                    <span><span class="status-dot status-dot--error" />执行失败</span>
                    <strong>{{ formatNumber(quality.error_count) }} 次</strong>
                  </div>
                </div>
              </SurfaceCard>

              <SurfaceCard padding="lg" class="document-card">
                <div class="card-heading">
                  <div>
                    <h3>文章状态</h3>
                    <p>{{ formatNumber(scale.documents) }} 篇文章的处理状态</p>
                  </div>
                  <div class="document-ready">
                    <strong>{{ formatPercent(readyRate) }}</strong>
                    <span>就绪率</span>
                  </div>
                </div>
                <div v-if="statusRows.length" class="status-list">
                  <StatusRow
                    v-for="item in statusRows"
                    :key="item.status"
                    :label="item.label"
                    :count="item.count"
                    :total="scale.documents"
                    :tone="item.tone"
                  />
                </div>
                <div v-else class="compact-empty">暂无文章状态数据</div>
              </SurfaceCard>

              <SurfaceCard padding="lg" class="performance-card">
                <div class="card-heading">
                  <div>
                    <h3>响应性能</h3>
                    <p>聊天调用链 · 近 {{ days }} 天</p>
                  </div>
                  <div class="card-heading__icon"><n-icon :size="20"><SpeedometerOutline /></n-icon></div>
                </div>
                <div class="performance-grid">
                  <PerfMetric label="平均响应" :value="performance.avg_duration_ms" unit="ms" />
                  <PerfMetric label="P95 响应" :value="performance.p95_duration_ms" unit="ms" />
                  <PerfMetric label="平均命中片段" :value="performance.avg_hit_count" />
                  <PerfMetric label="平均选择知识库" :value="performance.avg_selected_kb_count" />
                </div>
                <p class="retention-note">调用链数据随保留期自动清理，最长展示 30 天。</p>
              </SurfaceCard>
            </div>
          </section>

          <section aria-labelledby="system-heading">
            <div class="section-heading">
              <div>
                <h2 id="system-heading">用户与系统运营</h2>
                <p>查看高频使用用户、登录安全与管理行为</p>
              </div>
            </div>

            <div class="operations-grid">
              <SurfaceCard padding="none" class="user-ranking-card">
                <div class="table-card-heading">
                  <div>
                    <h3>用户问答次数 Top 10</h3>
                    <p>近 {{ days }} 天的问答活跃用户</p>
                  </div>
                  <span>{{ formatNumber(scale.active_users) }} 位活跃用户</span>
                </div>
                <n-data-table
                  :columns="userColumns"
                  :data="qa.per_user"
                  :bordered="false"
                  :scroll-x="760"
                  size="small"
                >
                  <template #empty><div class="table-empty">当前周期暂无用户问答记录</div></template>
                </n-data-table>
              </SurfaceCard>

              <div class="operations-side">
                <SurfaceCard padding="lg">
                  <div class="card-heading">
                    <div>
                      <h3>登录安全</h3>
                      <p>近 {{ days }} 天 · 登录日志保留 90 天</p>
                    </div>
                    <div class="card-heading__icon card-heading__icon--success">
                      <n-icon :size="20"><ShieldCheckmarkOutline /></n-icon>
                    </div>
                  </div>
                  <div class="security-grid">
                    <MiniStat label="登录成功" :value="security.login_success" tone="success" />
                    <MiniStat label="登录失败" :value="security.login_failed" tone="error" />
                    <MiniStat label="登录账号" :value="security.login_users" tone="info" />
                    <MiniStat label="失败来源 IP" :value="security.failed_sources" tone="warning" />
                  </div>
                </SurfaceCard>

                <SurfaceCard padding="lg">
                  <div class="card-heading">
                    <div>
                      <h3>管理操作 Top</h3>
                      <p>操作合计 {{ formatNumber(operations.total) }} 次</p>
                    </div>
                  </div>
                  <div v-if="operations.top_actions.length" class="action-list">
                    <div v-for="(item, index) in operations.top_actions" :key="item.action" class="action-list__item">
                      <span class="action-list__rank">{{ index + 1 }}</span>
                      <code>{{ item.action }}</code>
                      <strong>{{ formatNumber(item.count) }} 次</strong>
                    </div>
                  </div>
                  <div v-else class="compact-empty">近 {{ days }} 天暂无管理操作</div>
                  <p class="retention-note">管理操作日志保留 180 天。</p>
                </SurfaceCard>
              </div>
            </div>
          </section>
        </div>
      </n-spin>

      <AppModal
        v-if="canRead"
        v-model:show="reportModalOpen"
        title="AI 分析报告"
        width="min(94vw, 900px)"
        :loading="reportLoading"
      >
        <n-spin :show="reportLoading">
          <p class="ai-report-modal-note">只发送统计数字（不含问题正文、文档与会话内容）。</p>
          <div v-if="report" class="markdown-body ai-report-modal-content" v-html="renderedReport" />
          <div v-else-if="reportError" class="ai-report-modal-error">{{ reportError }}</div>
          <div v-else class="ai-report-modal-empty">
            <n-icon :size="22"><SparklesOutline /></n-icon>
            <span>正在准备分析报告…</span>
          </div>
        </n-spin>

        <template #footer>
          <n-button :disabled="reportLoading" @click="reportModalOpen = false">关闭</n-button>
          <n-button secondary :disabled="!report || reportLoading" @click="downloadReport">
            下载 Markdown
          </n-button>
          <n-button type="primary" :loading="reportLoading" @click="regenerateReport">
            重新生成
          </n-button>
        </template>
      </AppModal>
    </div>
  </div>
</template>

<script setup>
import { computed, defineComponent, h, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  NButton, NDataTable, NIcon, NRadioButton, NRadioGroup, NSpin, useMessage,
} from 'naive-ui'
import {
  AnalyticsOutline, ChatbubbleEllipsesOutline, LayersOutline, LibraryOutline,
  LockClosedOutline, PeopleOutline, PulseOutline, RefreshOutline, ShieldCheckmarkOutline,
  SparklesOutline, SpeedometerOutline, TrendingUpOutline,
} from '@vicons/ionicons5'
import * as echarts from 'echarts/core'
import { BarChart, LineChart, PieChart } from 'echarts/charts'
import { GridComponent, TitleComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { getDashboardAiReport, getDashboardOverview } from '@/api/dashboard'
import { renderMarkdown } from '@/utils/markdown'
import { useAuthStore } from '@/stores/auth'
import AppModal from '@/components/ui/AppModal.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'

echarts.use([BarChart, LineChart, PieChart, GridComponent, TitleComponent, TooltipComponent, CanvasRenderer])

const msg = useMessage()
const authStore = useAuthStore()
const canRead = computed(() => authStore.hasPerm('dashboard:read'))

const days = ref(7)
const loading = ref(false)
const overview = ref(null)
const report = ref('')
const reportError = ref('')
const reportLoading = ref(false)
const reportModalOpen = ref(false)

const numberFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 })
const dateTimeFormatter = new Intl.DateTimeFormat('zh-CN', {
  month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
})

const scale = computed(() => overview.value?.scale || {
  users: 0, active_users: 0, knowledge_bases: 0, documents: 0, chunks: 0, new_documents: 0,
  documents_by_status: {},
})
const qa = computed(() => overview.value?.qa || { total: 0, daily: [], per_user: [] })
const quality = computed(() => overview.value?.quality || {
  by_evidence: {}, hit_rate: 0, clarify_rate: 0, no_answer_rate: 0, error_count: 0,
})
const performance = computed(() => overview.value?.performance || {
  avg_duration_ms: null, p95_duration_ms: null, avg_hit_count: null, avg_selected_kb_count: null,
})
const tokens = computed(() => overview.value?.tokens || {
  prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, measured_calls: 0,
  avg_tokens_per_qa: 0, daily: [], by_stage: [],
})
const security = computed(() => overview.value?.security || {
  login_success: 0, login_failed: 0, login_users: 0, failed_sources: 0,
})
const operations = computed(() => overview.value?.operations || { total: 0, top_actions: [] })
const qualityByEvidence = computed(() => quality.value.by_evidence || {})
const renderedReport = computed(() => (report.value ? renderMarkdown(report.value) : ''))

const generatedAt = computed(() => {
  if (!overview.value?.generated_at) return ''
  const date = new Date(overview.value.generated_at)
  return Number.isNaN(date.getTime()) ? '' : dateTimeFormatter.format(date)
})
const activeRate = computed(() => (scale.value.users ? scale.value.active_users / scale.value.users : 0))
const readyRate = computed(() => (
  scale.value.documents ? (scale.value.documents_by_status?.ready || 0) / scale.value.documents : 0
))
const dailyAverage = computed(() => (
  qa.value.total && days.value ? Math.round((qa.value.total / days.value) * 10) / 10 : 0
))
const chunksPerDocument = computed(() => (
  scale.value.documents ? Math.round((scale.value.chunks / scale.value.documents) * 10) / 10 : 0
))
const peakDay = computed(() => {
  const peak = [...qa.value.daily].sort((a, b) => b.count - a.count)[0]
  return peak
    ? { count: peak.count, label: peak.date?.slice(5).replace('-', '/') || '—' }
    : { count: 0, label: '—' }
})
const evidenceHitCount = computed(() => (
  (qualityByEvidence.value.hit || 0) + (qualityByEvidence.value.partial || 0)
))
const noAnswerCount = computed(() => (
  (qualityByEvidence.value.no_hit || 0) + (qualityByEvidence.value.insufficient_evidence || 0)
))

const TOKEN_STAGE_LABELS = {
  'generation.completed': '回答生成',
  'intent.model_result': '意图路由',
}
const tokenStageEntries = computed(() => (tokens.value.by_stage || []).map(item => ({
  ...item,
  label: TOKEN_STAGE_LABELS[item.stage] || item.stage,
  rate: tokens.value.total_tokens ? item.total_tokens / tokens.value.total_tokens : 0,
})))

const STATUS_META = {
  ready: { label: '已就绪', tone: 'success' },
  draft: { label: '草稿', tone: 'warning' },
  processing: { label: '处理中', tone: 'info' },
  failed: { label: '处理失败', tone: 'error' },
  inactive: { label: '停用', tone: 'default' },
}
const statusRows = computed(() => Object.entries(scale.value.documents_by_status || {})
  .map(([status, count]) => ({
    status,
    count,
    ...(STATUS_META[status] || { label: status, tone: 'default' }),
  }))
  .sort((a, b) => b.count - a.count))

const EVIDENCE_META = {
  hit: { label: '命中', color: '--ui-success' },
  partial: { label: '部分命中', color: '--ui-info' },
  needs_clarification: { label: '需澄清', color: '--ui-warning' },
  no_hit: { label: '无命中', color: '--ui-danger' },
  insufficient_evidence: { label: '证据不足', color: '--ui-danger' },
  scope_mismatch: { label: '范围不符', color: '--ui-text-tertiary' },
  version_mismatch: { label: '版本不符', color: '--ui-text-tertiary' },
  unverified: { label: '未验证', color: '--ui-text-tertiary' },
  skipped: { label: '跳过检索', color: '--ui-text-tertiary' },
  error: { label: '失败', color: '--ui-danger' },
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

const evidenceEntries = computed(() => {
  const total = Object.values(qualityByEvidence.value).reduce((sum, count) => sum + Number(count || 0), 0)
  return Object.entries(qualityByEvidence.value)
    .filter(([, count]) => Number(count) > 0)
    .map(([key, count]) => {
      const meta = EVIDENCE_META[key] || { label: key, color: '--ui-text-tertiary' }
      return {
        key,
        label: meta.label,
        count: Number(count),
        rate: total ? Number(count) / total : 0,
        color: cssVar(meta.color, '#8190a5'),
      }
    })
    .sort((a, b) => b.count - a.count)
})

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return numberFormatter.format(Number(value))
}

function formatCompactNumber(value) {
  const number = Number(value) || 0
  if (number >= 100000000) return `${Math.round(number / 10000000) / 10}亿`
  if (number >= 10000) return `${Math.round(number / 1000) / 10}万`
  return numberFormatter.format(number)
}

function formatPercent(value) {
  const normalized = Math.max(0, Math.min(1, Number(value) || 0))
  return `${Math.round(normalized * 100)}%`
}

function formatOptionalPercent(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : formatPercent(value)
}

function formatDuration(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${formatNumber(value)} ms`
}

function formatDateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : dateTimeFormatter.format(date)
}

const userColumns = [
  {
    title: '排名', key: 'rank', width: 72, align: 'center',
    render: (_, index) => h('span', { class: `ranking-badge ranking-badge--${Math.min(index + 1, 4)}` }, index + 1),
  },
  {
    title: '用户', key: 'username',
    render: row => h('div', { class: 'user-cell' }, [
      h('span', { class: 'user-cell__avatar' }, (row.username || '—').slice(0, 1).toUpperCase()),
      h('span', { class: 'user-cell__name' }, row.username || '—'),
    ]),
  },
  {
    title: '问答次数', key: 'count', width: 120, align: 'center',
    render: row => h('strong', { class: 'count-cell' }, formatNumber(row.count)),
  },
  {
    title: '证据命中率', key: 'hit_rate', width: 120, align: 'center',
    render: row => h('span', { class: 'ranking-metric' }, formatOptionalPercent(row.hit_rate)),
  },
  {
    title: '平均响应', key: 'avg_duration_ms', width: 130, align: 'center',
    render: row => h('span', { class: 'ranking-metric' }, formatDuration(row.avg_duration_ms)),
  },
  {
    title: '最近问答', key: 'last_active_at', width: 140, align: 'center',
    render: row => h('span', { class: 'ranking-metric ranking-metric--time' }, formatDateTime(row.last_active_at)),
  },
]

const MetricCard = defineComponent({
  name: 'MetricCard',
  props: {
    label: String,
    value: [Number, String],
    relatedLabel: String,
    relatedValue: [Number, String],
    hint: String,
    progress: { type: Number, default: null },
    tone: { type: String, default: 'primary' },
    icon: { type: [Object, Function], required: true },
  },
  setup(props) {
    return () => h('article', { class: `metric-card metric-card--${props.tone}` }, [
      h('div', { class: 'metric-card__header' }, [
        h('div', { class: 'metric-card__icon' }, [h(NIcon, { size: 20 }, { default: () => h(props.icon) })]),
        h('span', { class: 'metric-card__label' }, props.label),
      ]),
      h('div', { class: 'metric-card__body' }, [
        h('strong', { class: 'metric-card__value' }, formatNumber(props.value)),
        h('div', { class: 'metric-card__related' }, [
          h('span', null, props.relatedLabel),
          h('strong', null, formatNumber(props.relatedValue)),
        ]),
      ]),
      props.progress != null
        ? h('div', { class: 'metric-card__progress', 'aria-hidden': 'true' }, [
            h('span', { style: { width: formatPercent(props.progress) } }),
          ])
        : h('div', { class: 'metric-card__divider' }),
      h('p', { class: 'metric-card__hint' }, props.hint),
    ])
  },
})

const MiniStat = defineComponent({
  name: 'MiniStat',
  props: { label: String, value: Number, tone: { type: String, default: 'default' } },
  setup(props) {
    return () => h('div', { class: `mini-stat mini-stat--${props.tone}` }, [
      h('span', { class: 'mini-stat__label' }, props.label),
      h('strong', { class: 'mini-stat__value' }, formatNumber(props.value)),
    ])
  },
})

const StatusRow = defineComponent({
  name: 'StatusRow',
  props: {
    label: String,
    count: Number,
    total: Number,
    rate: { type: Number, default: null },
    tone: { type: String, default: 'default' },
  },
  setup(props) {
    return () => {
      const rate = props.rate ?? (props.total ? props.count / props.total : 0)
      return h('div', { class: 'status-row' }, [
        h('div', { class: 'status-row__meta' }, [
          h('span', null, [h('span', { class: `status-dot status-dot--${props.tone}` }), props.label]),
          h('strong', null, `${formatNumber(props.count)} · ${formatPercent(rate)}`),
        ]),
        h('div', { class: 'status-row__track', 'aria-hidden': 'true' }, [
          h('span', { class: `status-row__fill status-row__fill--${props.tone}`, style: { width: formatPercent(rate) } }),
        ]),
      ])
    }
  },
})

const PerfMetric = defineComponent({
  name: 'PerfMetric',
  props: { label: String, value: { type: Number, default: null }, unit: { type: String, default: '' } },
  setup(props) {
    return () => h('div', { class: 'perf-metric' }, [
      h('span', null, props.label),
      h('div', null, [
        h('strong', null, formatNumber(props.value)),
        props.value != null && props.unit ? h('small', null, props.unit) : null,
      ]),
    ])
  },
})

const TokenStat = defineComponent({
  name: 'TokenStat',
  props: { label: String, value: { type: Number, default: 0 } },
  setup(props) {
    return () => h('div', { class: 'token-stat' }, [
      h('span', null, props.label),
      h('strong', null, formatCompactNumber(props.value)),
    ])
  },
})

const trendEl = ref(null)
const evidenceEl = ref(null)
const tokenEl = ref(null)
let trendChart = null
let evidenceChart = null
let tokenChart = null
let themeObserver = null

function chartTheme() {
  return {
    primary: cssVar('--ui-primary', '#3b82f6'),
    info: cssVar('--ui-info', '#3b82f6'),
    success: cssVar('--ui-success', '#209a6b'),
    text: cssVar('--ui-text', '#1b2a42'),
    textSecondary: cssVar('--ui-text-secondary', '#5e708b'),
    textTertiary: cssVar('--ui-text-tertiary', '#8190a5'),
    divider: cssVar('--ui-divider', '#e8eef5'),
    surface: cssVar('--ui-surface', '#ffffff'),
    border: cssVar('--ui-border', '#dce5f0'),
  }
}

function renderTrend() {
  if (!trendEl.value || !overview.value || !qa.value.daily.length) return
  if (!trendChart) trendChart = echarts.init(trendEl.value)
  const daily = qa.value.daily || []
  const theme = chartTheme()
  trendChart.setOption({
    animationDuration: 500,
    grid: { left: 10, right: 16, top: 34, bottom: 4, containLabel: true },
    tooltip: {
      trigger: 'axis',
      backgroundColor: theme.surface,
      borderColor: theme.border,
      borderWidth: 1,
      textStyle: { color: theme.text, fontSize: 12 },
      extraCssText: 'border-radius: 10px; box-shadow: 0 12px 28px rgba(18, 25, 51, 0.12);',
      formatter: params => {
        const item = params[0]
        const point = daily[item.dataIndex]
        return `<strong>${point?.date || item.axisValue}</strong><br/>问答次数&nbsp;&nbsp;${formatNumber(item.value)} 次<br/>平均响应&nbsp;&nbsp;${formatNumber(point?.avg_duration_ms)} ms`
      },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: daily.map(item => item.date.slice(5).replace('-', '/')),
      axisLine: { lineStyle: { color: theme.divider } },
      axisTick: { show: false },
      axisLabel: { color: theme.textTertiary, fontSize: 11, margin: 12, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: theme.textTertiary, fontSize: 11 },
      splitLine: { lineStyle: { color: theme.divider, type: 'dashed' } },
    },
    series: [{
      type: 'line',
      data: daily.map(item => item.count),
      smooth: 0.35,
      symbol: 'circle',
      symbolSize: daily.length <= 10 ? 7 : 5,
      showSymbol: daily.length <= 10,
      lineStyle: { color: theme.primary, width: 3 },
      itemStyle: { color: theme.surface, borderColor: theme.primary, borderWidth: 2 },
      emphasis: { focus: 'series', scale: 1.25 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: `${theme.primary}38` },
          { offset: 1, color: `${theme.primary}03` },
        ]),
      },
    }],
  }, true)
}

function renderEvidence() {
  if (!evidenceEl.value || !overview.value || !evidenceEntries.value.length) return
  if (!evidenceChart) evidenceChart = echarts.init(evidenceEl.value)
  const theme = chartTheme()
  evidenceChart.setOption({
    animationDuration: 500,
    title: {
      text: formatPercent(quality.value.hit_rate),
      subtext: '证据命中',
      left: 'center',
      top: '35%',
      textStyle: { color: theme.text, fontSize: 24, fontWeight: 650 },
      subtextStyle: { color: theme.textTertiary, fontSize: 11, lineHeight: 22 },
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: theme.surface,
      borderColor: theme.border,
      borderWidth: 1,
      textStyle: { color: theme.text, fontSize: 12 },
      formatter: '{b}<br/>{c} 次 · {d}%',
    },
    series: [{
      type: 'pie',
      radius: ['64%', '84%'],
      center: ['50%', '50%'],
      avoidLabelOverlap: true,
      label: { show: false },
      emphasis: { scale: true, scaleSize: 5 },
      itemStyle: { borderColor: theme.surface, borderWidth: 3, borderRadius: 5 },
      data: evidenceEntries.value.map(item => ({
        name: item.label,
        value: item.count,
        itemStyle: { color: item.color },
      })),
    }],
  }, true)
}

function renderTokens() {
  if (!tokenEl.value || !overview.value || !tokens.value.daily.length) return
  if (!tokenChart) tokenChart = echarts.init(tokenEl.value)
  const daily = tokens.value.daily || []
  const theme = chartTheme()
  tokenChart.setOption({
    animationDuration: 500,
    grid: { left: 10, right: 16, top: 22, bottom: 4, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: theme.surface,
      borderColor: theme.border,
      borderWidth: 1,
      textStyle: { color: theme.text, fontSize: 12 },
      extraCssText: 'border-radius: 10px; box-shadow: 0 12px 28px rgba(18, 25, 51, 0.12);',
      formatter: params => {
        const point = daily[params[0]?.dataIndex]
        return `<strong>${point?.date || ''}</strong><br/>输入 Token&nbsp;&nbsp;${formatNumber(point?.prompt_tokens)}<br/>输出 Token&nbsp;&nbsp;${formatNumber(point?.completion_tokens)}<br/>总消耗&nbsp;&nbsp;${formatNumber(point?.total_tokens)}`
      },
    },
    xAxis: {
      type: 'category',
      data: daily.map(item => item.date.slice(5).replace('-', '/')),
      axisLine: { lineStyle: { color: theme.divider } },
      axisTick: { show: false },
      axisLabel: { color: theme.textTertiary, fontSize: 11, margin: 12, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: theme.textTertiary, fontSize: 11, formatter: value => formatCompactNumber(value) },
      splitLine: { lineStyle: { color: theme.divider, type: 'dashed' } },
    },
    series: [
      {
        name: '输入 Token',
        type: 'bar',
        stack: 'tokens',
        barMaxWidth: 28,
        data: daily.map(item => item.prompt_tokens),
        itemStyle: { color: theme.primary, borderRadius: [0, 0, 4, 4] },
        emphasis: { focus: 'series' },
      },
      {
        name: '输出 Token',
        type: 'bar',
        stack: 'tokens',
        barMaxWidth: 28,
        data: daily.map(item => item.completion_tokens),
        itemStyle: { color: theme.success, borderRadius: [5, 5, 0, 0] },
        emphasis: { focus: 'series' },
      },
    ],
  }, true)
}

function renderCharts() {
  renderTrend()
  renderEvidence()
  renderTokens()
}

function resizeCharts() {
  trendChart?.resize()
  evidenceChart?.resize()
  tokenChart?.resize()
}

function onDaysChange() {
  report.value = ''
  reportError.value = ''
  reportModalOpen.value = false
  loadOverview()
}

async function loadOverview() {
  if (!canRead.value) return
  loading.value = true
  try {
    overview.value = await getDashboardOverview(days.value)
    await nextTick()
    renderCharts()
  } catch (error) {
    overview.value = null
    msg.error(error?.response?.data?.detail || '加载数据看板失败')
  } finally {
    loading.value = false
  }
}

async function generateReport() {
  if (!canRead.value || reportLoading.value) return
  reportLoading.value = true
  reportError.value = ''
  try {
    const data = await getDashboardAiReport(days.value)
    report.value = data?.report || ''
    if (!report.value) reportError.value = '模型未返回报告内容'
  } catch (error) {
    reportError.value = error?.response?.data?.detail || 'AI 分析报告生成失败，请稍后重试'
  } finally {
    reportLoading.value = false
  }
}

async function openReport() {
  if (!canRead.value || reportLoading.value) return
  if (report.value) {
    reportModalOpen.value = true
    return
  }

  reportModalOpen.value = true
  await generateReport()
}

async function regenerateReport() {
  if (!canRead.value || reportLoading.value) return
  reportModalOpen.value = true
  await generateReport()
}

function downloadReport() {
  if (!report.value) return
  const blob = new Blob([report.value], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `ai-dashboard-report-${days.value}d.md`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

onMounted(() => {
  loadOverview()
  window.addEventListener('resize', resizeCharts)
  themeObserver = new MutationObserver(() => requestAnimationFrame(renderCharts))
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts)
  themeObserver?.disconnect()
  trendChart?.dispose()
  evidenceChart?.dispose()
  tokenChart?.dispose()
})
</script>

<style scoped>
.dashboard-page {
  height: 100%;
  overflow-y: auto;
  padding: 20px 24px 32px;
}

.dashboard-container {
  width: 100%;
}

.dashboard-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
  margin-top: 0;
}

.dashboard-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}

.section-heading h2,
.card-heading h3,
.table-card-heading h3 {
  margin: 0;
  color: var(--ui-text);
  font-weight: 650;
}

.section-heading h2 { font-size: 15px; line-height: 1.5; }
.section-heading p,
.card-heading p,
.table-card-heading p {
  margin: 3px 0 0;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  line-height: 1.55;
}

.section-heading__meta {
  color: var(--ui-text-tertiary);
  font-size: 11px;
}

.section-heading__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 12px;
}

.section-heading--with-action { align-items: center; }

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

:deep(.metric-card) {
  position: relative;
  min-width: 0;
  overflow: hidden;
  padding: 18px;
  background: var(--ui-surface);
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  box-shadow: var(--ui-shadow-card);
}

:deep(.metric-card::before) {
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  height: 3px;
  background: var(--metric-accent);
  content: '';
  opacity: 0.9;
}

:deep(.metric-card--primary) { --metric-accent: var(--ui-primary); --metric-soft: var(--ui-primary-subtle); }
:deep(.metric-card--info) { --metric-accent: var(--ui-info); --metric-soft: color-mix(in srgb, var(--ui-info) 12%, var(--ui-surface)); }
:deep(.metric-card--violet) { --metric-accent: var(--ui-accent-violet); --metric-soft: color-mix(in srgb, var(--ui-accent-violet) 12%, var(--ui-surface)); }
:deep(.metric-card--success) { --metric-accent: var(--ui-success); --metric-soft: color-mix(in srgb, var(--ui-success) 12%, var(--ui-surface)); }
:deep(.metric-card--warning) { --metric-accent: var(--ui-warning); --metric-soft: color-mix(in srgb, var(--ui-warning) 12%, var(--ui-surface)); }

:deep(.metric-card__header) { display: flex; align-items: center; gap: 9px; }
:deep(.metric-card__icon) {
  display: inline-flex;
  width: 34px;
  height: 34px;
  align-items: center;
  justify-content: center;
  color: var(--metric-accent);
  background: var(--metric-soft);
  border-radius: var(--ui-radius-control);
}
:deep(.metric-card__label) { color: var(--ui-text-secondary); font-size: 12px; font-weight: 600; }
:deep(.metric-card__body) {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  margin-top: 15px;
}
:deep(.metric-card__value) {
  overflow: hidden;
  color: var(--ui-text);
  font-size: clamp(25px, 2.2vw, 32px);
  font-variant-numeric: tabular-nums;
  font-weight: 680;
  letter-spacing: -0.035em;
  line-height: 1.15;
  text-overflow: ellipsis;
}
:deep(.metric-card__related) { flex: 0 0 auto; text-align: right; }
:deep(.metric-card__related span) { display: block; color: var(--ui-text-tertiary); font-size: 10px; }
:deep(.metric-card__related strong) { display: block; margin-top: 2px; color: var(--ui-text-secondary); font-size: 14px; }
:deep(.metric-card__progress),
:deep(.metric-card__divider) {
  height: 3px;
  margin-top: 15px;
  overflow: hidden;
  background: var(--ui-surface-muted);
  border-radius: var(--ui-radius-pill);
}
:deep(.metric-card__progress span) {
  display: block;
  height: 100%;
  background: var(--metric-accent);
  border-radius: inherit;
}
:deep(.metric-card__hint) {
  overflow: hidden;
  margin: 8px 0 0;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.analytics-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(340px, 1fr);
  gap: 16px;
}

.chart-card { min-width: 0; }
.card-heading,
.table-card-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
.card-heading h3,
.table-card-heading h3 { font-size: 14px; line-height: 1.5; }

.trend-summary {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 18px;
}
.trend-summary > div { min-width: 48px; }
.trend-summary span { display: block; color: var(--ui-text-tertiary); font-size: 10px; }
.trend-summary strong { display: block; margin-top: 3px; color: var(--ui-text); font-size: 13px; font-variant-numeric: tabular-nums; }

.chart-stage { position: relative; min-height: 270px; margin-top: 8px; }
.dashboard-chart { width: 100%; height: 270px; }
.chart-empty {
  display: flex;
  min-height: 270px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
}
.chart-empty--evidence { min-height: 286px; }

.evidence-layout { display: grid; grid-template-columns: 156px minmax(0, 1fr); align-items: center; gap: 16px; min-height: 286px; }
.evidence-chart { width: 156px; height: 180px; }
.evidence-legend { display: flex; min-width: 0; max-height: 238px; flex-direction: column; gap: 9px; overflow-y: auto; padding-right: 2px; }
.evidence-legend__item { display: grid; grid-template-columns: 8px minmax(0, 1fr) auto 40px; align-items: center; gap: 7px; font-size: 11px; }
.evidence-legend__dot { width: 7px; height: 7px; border-radius: 50%; }
.evidence-legend__label { overflow: hidden; color: var(--ui-text-secondary); text-overflow: ellipsis; white-space: nowrap; }
.evidence-legend__item strong { color: var(--ui-text); font-size: 11px; font-variant-numeric: tabular-nums; }
.evidence-legend__rate { color: var(--ui-text-tertiary); text-align: right; font-variant-numeric: tabular-nums; }

.token-card { min-width: 0; }
.token-card__heading { align-items: center; }
.token-chart-legend { display: flex; flex: 0 0 auto; align-items: center; gap: 14px; color: var(--ui-text-tertiary); font-size: 11px; }
.token-chart-legend span { display: inline-flex; align-items: center; gap: 6px; }
.token-chart-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 3px; }
.token-chart-legend__input { background: var(--ui-primary); }
.token-chart-legend__output { background: var(--ui-success); }
.token-summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 20px; }
:deep(.token-stat) { min-width: 0; padding: 12px 14px; background: var(--ui-surface-muted); border-radius: var(--ui-radius-control); }
:deep(.token-stat span) { display: block; overflow: hidden; color: var(--ui-text-tertiary); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
:deep(.token-stat strong) { display: block; margin-top: 4px; overflow: hidden; color: var(--ui-text); font-size: 19px; font-variant-numeric: tabular-nums; text-overflow: ellipsis; }
.token-layout { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(260px, 0.8fr); gap: 24px; align-items: stretch; margin-top: 18px; }
.token-chart-stage { position: relative; min-height: 245px; }
.token-chart { width: 100%; height: 245px; }
.token-chart-empty { min-height: 245px; height: 245px; }
.token-stage-panel { min-width: 0; padding: 15px; background: var(--ui-surface-muted); border-radius: var(--ui-radius-control); }
.token-stage-panel__heading h4 { margin: 0; color: var(--ui-text); font-size: 12px; font-weight: 650; }
.token-stage-panel__heading p { margin: 3px 0 0; color: var(--ui-text-tertiary); font-size: 10px; }
.token-stage-list { display: flex; flex-direction: column; gap: 15px; margin-top: 18px; }
.token-stage-item__meta { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--ui-text-secondary); font-size: 11px; }
.token-stage-item__meta strong { color: var(--ui-text); font-size: 11px; font-variant-numeric: tabular-nums; }
.token-stage-item__track { height: 5px; margin-top: 7px; overflow: hidden; background: var(--ui-surface); border-radius: var(--ui-radius-pill); }
.token-stage-item__track span { display: block; height: 100%; background: var(--ui-primary); border-radius: inherit; }
.token-stage-item:nth-child(2) .token-stage-item__track span { background: var(--ui-info); }
.token-stage-item:nth-child(3) .token-stage-item__track span { background: var(--ui-warning); }
.token-stage-item:nth-child(4) .token-stage-item__track span { background: var(--ui-success); }
.token-stage-item__foot { display: flex; align-items: center; justify-content: space-between; margin-top: 4px; color: var(--ui-text-tertiary); font-size: 10px; }
.token-stage-empty { min-height: 150px; }
.token-note { margin: 16px 0 0; color: var(--ui-text-tertiary); font-size: 10px; line-height: 1.55; }

.quality-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.quality-score,
.document-ready { flex: 0 0 auto; text-align: right; }
.quality-score strong,
.document-ready strong { display: block; color: var(--ui-text); font-size: 20px; font-weight: 680; line-height: 1.15; }
.quality-score span,
.document-ready span { display: block; margin-top: 3px; color: var(--ui-text-tertiary); font-size: 10px; }
.status-list { display: flex; flex-direction: column; gap: 15px; margin-top: 22px; }
:deep(.status-row__meta) { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--ui-text-secondary); font-size: 11px; }
:deep(.status-row__meta > span) { display: inline-flex; align-items: center; gap: 7px; }
:deep(.status-row__meta strong) { color: var(--ui-text); font-size: 11px; font-variant-numeric: tabular-nums; }
:deep(.status-row__track) { height: 5px; margin-top: 7px; overflow: hidden; background: var(--ui-surface-muted); border-radius: var(--ui-radius-pill); }
:deep(.status-row__fill) { display: block; height: 100%; border-radius: inherit; }
:deep(.status-row__fill--success), .status-dot--success { background: var(--ui-success); }
:deep(.status-row__fill--warning), .status-dot--warning { background: var(--ui-warning); }
:deep(.status-row__fill--error), .status-dot--error { background: var(--ui-danger); }
:deep(.status-row__fill--info), .status-dot--info { background: var(--ui-info); }
:deep(.status-row__fill--default), .status-dot--default { background: var(--ui-text-tertiary); }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; }
.inline-note { display: flex; align-items: center; justify-content: space-between; color: var(--ui-text-secondary); font-size: 11px; }
.inline-note > span { display: inline-flex; align-items: center; gap: 7px; }
.inline-note strong { color: var(--ui-text); font-size: 11px; }

.card-heading__icon {
  display: inline-flex;
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  color: var(--ui-primary);
  background: var(--ui-primary-subtle);
  border-radius: var(--ui-radius-control);
}
.card-heading__icon--success { color: var(--ui-success); background: color-mix(in srgb, var(--ui-success) 12%, var(--ui-surface)); }
.performance-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 20px; }
:deep(.perf-metric) { min-width: 0; padding: 12px; background: var(--ui-surface-muted); border-radius: var(--ui-radius-control); }
:deep(.perf-metric > span) { display: block; color: var(--ui-text-tertiary); font-size: 10px; }
:deep(.perf-metric > div) { display: flex; align-items: baseline; gap: 4px; margin-top: 5px; }
:deep(.perf-metric strong) { overflow: hidden; color: var(--ui-text); font-size: 18px; font-variant-numeric: tabular-nums; text-overflow: ellipsis; }
:deep(.perf-metric small) { color: var(--ui-text-tertiary); font-size: 10px; }
.retention-note { margin: 13px 0 0; color: var(--ui-text-tertiary); font-size: 10px; line-height: 1.55; }

.ai-report-modal-content {
  max-height: min(68vh, 720px);
  overflow-y: auto;
  padding: 4px 4px 8px 0;
  color: var(--ui-text-secondary);
  font-size: 13px;
  line-height: 1.75;
}
.ai-report-modal-note { margin: 0 0 12px; color: var(--ui-text-tertiary); font-size: 11px; line-height: 1.55; }
.ai-report-modal-empty {
  display: flex;
  min-height: 180px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 7px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
}
.ai-report-modal-error { min-height: 110px; padding: 12px 14px; color: var(--ui-danger); background: var(--ui-danger-subtle); border-radius: var(--ui-radius-control); font-size: 12px; }

.operations-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
.operations-side { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: stretch; }
.table-card-heading { padding: 20px 24px 15px; border-bottom: 1px solid var(--ui-divider); }
.table-card-heading > span { flex: 0 0 auto; padding: 5px 9px; color: var(--ui-primary); background: var(--ui-primary-subtle); border-radius: var(--ui-radius-pill); font-size: 10px; font-weight: 600; }
.user-ranking-card { overflow: hidden; }
:deep(.ranking-badge) {
  display: inline-flex;
  width: 24px;
  height: 24px;
  align-items: center;
  justify-content: center;
  color: var(--ui-text-tertiary);
  background: var(--ui-surface-muted);
  border-radius: 8px;
  font-size: 11px;
  font-weight: 650;
}
:deep(.ranking-badge--1) { color: var(--ui-primary); background: var(--ui-primary-subtle); }
:deep(.ranking-badge--2),
:deep(.ranking-badge--3) { color: var(--ui-info); background: color-mix(in srgb, var(--ui-info) 10%, var(--ui-surface)); }
:deep(.user-cell) { display: flex; min-width: 0; align-items: center; gap: 9px; }
:deep(.user-cell__avatar) { display: inline-flex; width: 28px; height: 28px; flex: 0 0 auto; align-items: center; justify-content: center; color: var(--ui-primary); background: var(--ui-primary-subtle); border-radius: 50%; font-size: 11px; font-weight: 650; }
:deep(.user-cell__name) { overflow: hidden; color: var(--ui-text); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
:deep(.count-cell) { color: var(--ui-text); font-size: 12px; font-variant-numeric: tabular-nums; }
:deep(.ranking-metric) { color: var(--ui-text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; white-space: nowrap; }
:deep(.ranking-metric--time) { color: var(--ui-text-tertiary); }
.table-empty { padding: 26px; color: var(--ui-text-tertiary); font-size: 12px; }

.security-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 18px; }
:deep(.mini-stat) { padding: 12px 14px; background: var(--ui-surface-muted); border-radius: var(--ui-radius-control); }
:deep(.mini-stat__label) { display: block; color: var(--ui-text-tertiary); font-size: 10px; }
:deep(.mini-stat__value) { display: block; margin-top: 4px; color: var(--ui-text); font-size: 20px; font-variant-numeric: tabular-nums; }
:deep(.mini-stat--success .mini-stat__value) { color: var(--ui-success); }
:deep(.mini-stat--error .mini-stat__value) { color: var(--ui-danger); }
:deep(.mini-stat--info .mini-stat__value) { color: var(--ui-info); }
:deep(.mini-stat--warning .mini-stat__value) { color: var(--ui-warning); }
.action-list { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
.action-list__item { display: grid; grid-template-columns: 24px minmax(0, 1fr) auto; align-items: center; gap: 9px; font-size: 11px; }
.action-list__rank { display: inline-flex; width: 22px; height: 22px; align-items: center; justify-content: center; color: var(--ui-text-tertiary); background: var(--ui-surface-muted); border-radius: 7px; font-weight: 650; }
.action-list__item code { overflow: hidden; color: var(--ui-text-secondary); text-overflow: ellipsis; white-space: nowrap; }
.action-list__item strong { color: var(--ui-text); font-size: 11px; }
.compact-empty { display: flex; min-height: 110px; align-items: center; justify-content: center; color: var(--ui-text-tertiary); font-size: 12px; }

.permission-empty { display: flex; min-height: 260px; align-items: center; justify-content: center; flex-direction: column; text-align: center; }
.permission-empty__icon { display: flex; width: 52px; height: 52px; align-items: center; justify-content: center; color: var(--ui-text-tertiary); background: var(--ui-surface-muted); border-radius: 16px; }
.permission-empty__title { margin-top: 14px; color: var(--ui-text); font-size: 14px; font-weight: 650; }
.permission-empty p { margin: 6px 0 0; color: var(--ui-text-tertiary); font-size: 12px; }

@media (max-width: 1179px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .analytics-grid { grid-template-columns: 1fr; }
  .quality-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .performance-card { grid-column: 1 / -1; }
  .token-layout { grid-template-columns: 1fr; }
}

@media (max-width: 767px) {
  .dashboard-page { padding: 16px; }
  .dashboard-content { gap: 26px; margin-top: 18px; }
  .quality-grid { grid-template-columns: 1fr; }
  .performance-card { grid-column: auto; }
  .operations-side { grid-template-columns: 1fr; }
  .evidence-layout { grid-template-columns: 170px minmax(0, 1fr); justify-content: center; }
  .evidence-chart { width: 170px; }
}

@media (max-width: 639px) {
  .dashboard-toolbar { width: 100%; justify-content: space-between; }
  .dashboard-toolbar :deep(.n-radio-group) { flex: 1; }
  .dashboard-toolbar :deep(.n-radio-button) { flex: 1; }
  .dashboard-toolbar :deep(.n-radio-button__main) { justify-content: center; min-height: 40px; }
  .metric-grid { grid-template-columns: 1fr; }
  .section-heading { align-items: flex-start; }
  .section-heading--controls { flex-direction: column; }
  .section-heading__actions { width: 100%; align-items: flex-start; flex-direction: column; gap: 8px; }
  .section-heading__meta { display: none; }
  .card-heading { flex-direction: column; }
  .trend-summary { width: 100%; justify-content: space-between; padding: 10px 12px; background: var(--ui-surface-muted); border-radius: var(--ui-radius-control); }
  .token-card__heading { align-items: flex-start; }
  .token-chart-legend { width: 100%; }
  .token-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .chart-stage,
  .dashboard-chart,
  .chart-empty { min-height: 235px; height: 235px; }
  .token-chart-stage,
  .token-chart,
  .token-chart-empty { min-height: 220px; height: 220px; }
  .evidence-layout { grid-template-columns: 1fr; gap: 4px; }
  .evidence-chart { width: 100%; height: 190px; }
  .evidence-legend { max-height: none; }
  .table-card-heading { padding: 16px; }
  .table-card-heading > span { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  :deep(.metric-card__progress span),
  :deep(.status-row__fill) { transition: none; }
}
</style>
