<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-5">
      <PageHeader title="审计日志" description="查看后台管理操作与账号访问记录，详情保留完整的审计元数据。" />
      <SurfaceCard padding="none" class="overflow-hidden">
        <n-tabs v-model:value="tab" type="line" animated class="audit-log-tabs">
          <n-tab-pane name="operation" tab="操作日志">
            <div class="p-4 sm:p-5"><OperationLogsPanel /></div>
          </n-tab-pane>
          <n-tab-pane name="login" tab="登录日志">
            <div class="p-4 sm:p-5"><LoginLogsPanel /></div>
          </n-tab-pane>
        </n-tabs>
      </SurfaceCard>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { NTabs, NTabPane } from 'naive-ui'
import OperationLogsPanel from '@/components/admin/OperationLogsPanel.vue'
import LoginLogsPanel from '@/components/admin/LoginLogsPanel.vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'

const tab = ref('operation')
</script>

<style scoped>
.audit-log-tabs :deep(.n-tabs-nav) {
  padding: 0 20px;
  border-bottom: 1px solid var(--ui-border, #e2e8f0);
}

@media (max-width: 639px) {
  .audit-log-tabs :deep(.n-tabs-nav) { padding: 0 16px; }
}
</style>
