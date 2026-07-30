<template>
  <div class="space-y-4">
    <n-data-table
      remote
      :columns="columns" :data="logs" :loading="loading"
      :pagination="pagination" :scroll-x="ui.isCompact ? 820 : 920"
      class="admin-data-table"
    />

    <AuditDetailDrawer v-model:show="showDetail" title="登录日志详情" subtitle="用于定位账号访问和安全问题。">
      <div v-if="current" class="login-detail space-y-4">
        <section class="login-detail__surface">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="login-detail__label">登录账号</div>
              <div class="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-100">{{ current.username || '—' }}</div>
            </div>
            <n-tag :type="current.success ? 'success' : 'error'" size="small" round>
              {{ current.success ? '登录成功' : '登录失败' }}
            </n-tag>
          </div>
          <p v-if="!current.success" class="login-detail__failure">{{ current.fail_reason || '未提供失败原因' }}</p>
          <dl class="login-detail__grid mt-5">
            <div><dt>首次尝试</dt><dd class="tabular-nums">{{ fmtTime(current.created_at) }}</dd></div>
            <div><dt>最近尝试</dt><dd class="tabular-nums">{{ fmtTime(current.last_attempt_at || current.created_at) }}</dd></div>
            <div><dt>尝试次数</dt><dd class="tabular-nums">{{ current.attempt_count || 1 }}</dd></div>
            <div><dt>IP 地址</dt><dd>{{ current.ip || '—' }}</dd></div>
            <div><dt>浏览器</dt><dd>{{ parseUA(current.user_agent).browser }}</dd></div>
            <div><dt>操作系统</dt><dd>{{ parseUA(current.user_agent).os }}</dd></div>
            <div><dt>设备类型</dt><dd>{{ parseUA(current.user_agent).device }}</dd></div>
          </dl>
        </section>

        <section class="login-detail__surface">
          <h3 class="login-detail__section-title">审计标识</h3>
          <dl class="login-detail__identifiers">
            <div><dt>用户 ID</dt><dd>{{ current.user_id || '—' }}</dd></div>
            <div><dt>日志 ID</dt><dd>{{ current.id || '—' }}</dd></div>
            <div><dt>User-Agent</dt><dd>{{ current.user_agent || '—' }}</dd></div>
          </dl>
        </section>
      </div>
    </AuditDetailDrawer>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, h } from 'vue'
import { NDataTable, NTag, NButton } from 'naive-ui'
import { getLoginLogs } from '@/api/loginLogs'
import { parseUA } from '@/utils/ua'
import { useUiStore } from '@/stores/ui'
import AuditDetailDrawer from '@/components/ui/AuditDetailDrawer.vue'

const ui = useUiStore()
const logs = ref([])
const loading = ref(false)

const showDetail = ref(false)
const current = ref(null)

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

const dateTimeFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
})

const fmtTime = (v) => {
  if (!v) return '—'
  const date = new Date(v)
  if (Number.isNaN(date.getTime())) return String(v)
  const parts = Object.fromEntries(dateTimeFormatter.formatToParts(date)
    .filter(part => part.type !== 'literal')
    .map(part => [part.type, part.value]))
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`
}

const timeParts = value => {
  const formatted = fmtTime(value)
  const [date = formatted, time = ''] = formatted.split(' ')
  return { date, time }
}

const columns = [
  { title: '用户名', key: 'username', width: 150, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  {
    title: '结果', key: 'success', width: 105, align: 'center', titleAlign: 'center',
    render: r => h(
      NTag,
      { type: r.success ? 'success' : 'error', size: 'small' },
      () => r.success ? '成功' : (r.attempt_count || 1) > 1 ? `失败 ×${r.attempt_count}` : '失败'
    )
  },
  {
    title: '失败原因', key: 'fail_reason', minWidth: 160, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => r.success
      ? h('span', { class: 'text-gray-300 dark:text-gray-600' }, '—')
      : h('span', { class: 'text-xs text-red-500 dark:text-red-400' }, r.fail_reason || '—')
  },
  { title: 'IP', key: 'ip', width: 132, align: 'center', titleAlign: 'center', render: r => r.ip || '—' },
  {
    title: '浏览器 / 系统', key: 'ua', minWidth: 170, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: r => {
      const { browser, os } = parseUA(r.user_agent)
      return `${browser} · ${os}`
    }
  },
  {
    title: '最近时间', key: 'last_attempt_at', width: 158, align: 'center', titleAlign: 'center',
    render: row => {
      const time = timeParts(row.last_attempt_at || row.created_at)
      return h('div', { class: 'login-time' }, [
        h('span', { class: 'login-time__date' }, time.date),
        h('span', { class: 'login-time__value' }, time.time),
      ])
    },
  },
  {
    title: '操作', key: 'actions', width: 88, align: 'center', titleAlign: 'center',
    render: row => h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openDetail(row) }, () => '详情')
  },
]

function openDetail(row) {
  current.value = row
  showDetail.value = true
}

onMounted(loadLogs)

async function loadLogs() {
  loading.value = true
  try {
    const data = await getLoginLogs({ page: pagination.page, page_size: pagination.pageSize })
    logs.value = data.items
    pagination.itemCount = data.total
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-time {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  color: var(--ui-text-secondary, #64748b);
  font-size: 12px;
  line-height: 1.45;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.login-time__date { color: var(--ui-text-secondary, #64748b); }
.login-time__value { color: var(--ui-text, #1e293b); }

.login-detail__surface {
  padding: 16px;
  background: var(--ui-surface, #ffffff);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-card, 16px);
}

.login-detail__label,
.login-detail dt {
  color: var(--ui-text-tertiary, #94a3b8);
  font-size: 12px;
  line-height: 1.4;
}

.login-detail__failure {
  margin: 16px 0 0;
  padding: 10px 12px;
  color: var(--ui-danger, #dc2626);
  background: var(--ui-danger-subtle, #fef2f2);
  border-radius: var(--ui-radius-control, 10px);
  font-size: 13px;
  line-height: 1.55;
}

.login-detail__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.login-detail dd {
  min-width: 0;
  margin: 5px 0 0;
  overflow-wrap: anywhere;
  color: var(--ui-text, #1e293b);
  font-size: 13px;
  line-height: 1.55;
}

.login-detail__section-title {
  margin: 0 0 12px;
  color: var(--ui-text, #1e293b);
  font-size: 13px;
  font-weight: 650;
}

.login-detail__identifiers {
  display: grid;
  gap: 10px;
}

.login-detail__identifiers > div {
  display: grid;
  grid-template-columns: 80px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
}

.login-detail__identifiers dd {
  margin: 0;
  color: var(--ui-text-secondary, #64748b);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}

@media (max-width: 639px) {
  .login-detail__grid { grid-template-columns: 1fr; gap: 12px; }
}
</style>
