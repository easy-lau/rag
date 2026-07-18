<template>
  <div>
    <n-data-table
      remote
      :columns="columns" :data="logs" :loading="loading"
      :pagination="pagination" :scroll-x="ui.isMobile ? 1000 : undefined"
      class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden"
    />

    <!-- 详情弹窗：展示单条登录日志的完整信息 -->
    <n-modal v-model:show="showDetail" to="#app" preset="card" title="登录日志详情" style="width: 560px; max-width: 92vw">
      <n-descriptions v-if="current" :column="1" label-placement="left" bordered label-style="width: 110px">
        <n-descriptions-item label="用户名">{{ current.username || '—' }}</n-descriptions-item>
        <n-descriptions-item label="登录结果">
          <n-tag :type="current.success ? 'success' : 'error'" size="small">
            {{ current.success ? '成功' : '失败' }}
          </n-tag>
        </n-descriptions-item>
        <n-descriptions-item v-if="!current.success" label="失败原因">
          <span class="text-red-500 dark:text-red-400">{{ current.fail_reason || '—' }}</span>
        </n-descriptions-item>
        <n-descriptions-item label="IP 地址">{{ current.ip || '—' }}</n-descriptions-item>
        <n-descriptions-item label="浏览器">{{ parseUA(current.user_agent).browser }}</n-descriptions-item>
        <n-descriptions-item label="操作系统">{{ parseUA(current.user_agent).os }}</n-descriptions-item>
        <n-descriptions-item label="设备类型">{{ parseUA(current.user_agent).device }}</n-descriptions-item>
        <n-descriptions-item label="登录时间">{{ fmtTime(current.created_at) }}</n-descriptions-item>
        <n-descriptions-item label="用户 ID">
          <span class="text-xs font-mono break-all text-gray-500 dark:text-gray-400">{{ current.user_id || '—' }}</span>
        </n-descriptions-item>
        <n-descriptions-item label="日志 ID">
          <span class="text-xs font-mono break-all text-gray-500 dark:text-gray-400">{{ current.id }}</span>
        </n-descriptions-item>
        <n-descriptions-item label="User-Agent">
          <span class="text-xs break-all text-gray-500 dark:text-gray-400 leading-relaxed">{{ current.user_agent || '—' }}</span>
        </n-descriptions-item>
      </n-descriptions>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, h } from 'vue'
import { NDataTable, NTag, NButton, NModal, NDescriptions, NDescriptionsItem } from 'naive-ui'
import { getLoginLogs } from '@/api/loginLogs'
import { parseUA } from '@/utils/ua'
import { useUiStore } from '@/stores/ui'

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

const fmtTime = (v) => v ? new Date(v).toLocaleString('zh-CN') : '—'

const columns = [
  { title: '用户名', key: 'username', width: 150, ellipsis: { tooltip: true } },
  {
    title: '结果', key: 'success', width: 90,
    render: r => h(NTag, { type: r.success ? 'success' : 'error', size: 'small' }, () => r.success ? '成功' : '失败')
  },
  {
    title: '失败原因', key: 'fail_reason', width: 130,
    render: r => r.success
      ? h('span', { class: 'text-gray-300 dark:text-gray-600' }, '—')
      : h('span', { class: 'text-xs text-red-500 dark:text-red-400' }, r.fail_reason || '—')
  },
  { title: 'IP', key: 'ip', width: 150, render: r => r.ip || '—' },
  {
    title: '浏览器 / 系统', key: 'ua', ellipsis: { tooltip: true },
    render: r => {
      const { browser, os } = parseUA(r.user_agent)
      return `${browser} · ${os}`
    }
  },
  { title: '时间', key: 'created_at', width: 180, render: r => fmtTime(r.created_at) },
  {
    title: '操作', key: 'actions', width: 100, align: 'center',
    render: row => h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openDetail(row) }, () => '查看详细')
  },
]
columns.forEach(c => { c.titleAlign = 'center'; c.align = 'center' })  // 表头 + 内容统一居中

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
