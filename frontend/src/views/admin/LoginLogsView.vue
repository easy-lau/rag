<template>
  <div class="p-6 h-full overflow-y-auto">
    <n-data-table
      remote
      :columns="columns" :data="logs" :loading="loading"
      :pagination="pagination"
      class="bg-white dark:bg-gray-800 rounded-xl"
    />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, h } from 'vue'
import { NDataTable, NTag } from 'naive-ui'
import { getLoginLogs } from '@/api/loginLogs'

const logs = ref([])
const loading = ref(false)

const pagination = reactive({
  page: 1,
  pageSize: 10,                      // 默认每页 10 条
  itemCount: 0,
  showSizePicker: true,
  pageSizes: [10, 20, 30, 50],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (p) => { pagination.page = p; loadLogs() },
  onUpdatePageSize: (ps) => { pagination.pageSize = ps; pagination.page = 1; loadLogs() },
})

const fmtTime = (v) => v ? new Date(v).toLocaleString() : '—'

const columns = [
  { title: '用户名', key: 'username', ellipsis: { tooltip: true } },
  {
    title: '状态', key: 'success', width: 100,
    render: r => h(NTag, { type: r.success ? 'success' : 'error', size: 'small' }, () => r.success ? '成功' : '失败')
  },
  { title: 'IP', key: 'ip', width: 160, render: r => r.ip || '—' },
  { title: 'User-Agent', key: 'user_agent', ellipsis: { tooltip: true }, render: r => r.user_agent || '—' },
  { title: '时间', key: 'created_at', width: 200, render: r => fmtTime(r.created_at) },
]

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
