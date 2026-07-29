<template>
  <n-drawer
    :show="show"
    :width="drawerWidth"
    placement="right"
    to="#app"
    @update:show="emit('update:show', $event)"
  >
    <n-drawer-content
      :title="title"
      closable
      :native-scrollbar="false"
      :header-style="headerStyle"
      :body-content-style="bodyStyle"
    >
      <div class="ui-audit-detail-drawer">
        <p v-if="subtitle" class="ui-audit-detail-drawer__subtitle">{{ subtitle }}</p>
        <slot />
      </div>
    </n-drawer-content>
  </n-drawer>
</template>

<script setup>
import { computed } from 'vue'
import { NDrawer, NDrawerContent } from 'naive-ui'
import { useUiStore } from '@/stores/ui'

defineProps({
  show: { type: Boolean, default: false },
  title: { type: String, required: true },
  subtitle: { type: String, default: '' },
})

const emit = defineEmits(['update:show'])
const ui = useUiStore()
const drawerWidth = computed(() => (ui.isMobile ? '94vw' : ui.isCompact ? 'min(88vw, 560px)' : 560))
const headerStyle = { borderBottom: '1px solid var(--ui-border)', padding: '18px 20px' }
const bodyStyle = { background: 'var(--ui-surface-muted)', padding: '20px' }
</script>

<style scoped>
.ui-audit-detail-drawer {
  min-height: 100%;
  color: var(--ui-text, #1e293b);
}

.ui-audit-detail-drawer__subtitle {
  margin: -2px 0 16px;
  color: var(--ui-text-secondary, #64748b);
  font-size: 12px;
  line-height: 1.55;
}

@media (max-width: 639px) {
  .ui-audit-detail-drawer__subtitle { margin-bottom: 12px; }
}
</style>
