<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <n-tabs v-model:value="tab" type="line" animated>
      <n-tab-pane name="operation" tab="操作日志">
        <OperationLogsPanel />
      </n-tab-pane>
      <n-tab-pane name="login" tab="登录日志">
        <LoginLogsPanel />
      </n-tab-pane>
    </n-tabs>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { NTabs, NTabPane } from 'naive-ui'
import OperationLogsPanel from '@/components/admin/OperationLogsPanel.vue'
import LoginLogsPanel from '@/components/admin/LoginLogsPanel.vue'

const tab = ref('operation')
</script>
