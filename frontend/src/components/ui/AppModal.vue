<template>
  <n-modal
    :show="show"
    preset="card"
    class="ui-app-modal"
    :title="title"
    :style="modalStyle"
    :to="to"
    :mask-closable="maskClosable && !loading"
    :close-on-esc="closeOnEsc && !loading"
    :closable="closable && !loading"
    @update:show="handleShowUpdate"
  >
    <slot />
    <template v-if="$slots.footer" #footer>
      <div class="ui-app-modal__footer"><slot name="footer" /></div>
    </template>
  </n-modal>
</template>

<script setup>
import { computed } from 'vue'
import { NModal } from 'naive-ui'

const props = defineProps({
  show: { type: Boolean, default: false },
  title: { type: String, required: true },
  width: { type: String, default: 'min(92vw, 520px)' },
  to: { type: String, default: '#app' },
  loading: { type: Boolean, default: false },
  maskClosable: { type: Boolean, default: true },
  closeOnEsc: { type: Boolean, default: true },
  closable: { type: Boolean, default: true },
})

const emit = defineEmits(['update:show', 'close'])
const modalStyle = computed(() => ({ width: props.width }))

function handleShowUpdate(next) {
  if (!next && props.loading) return
  emit('update:show', next)
  if (!next) emit('close')
}
</script>

<style scoped>
:deep(.ui-app-modal.n-card) {
  overflow: hidden;
  border: 1px solid var(--ui-border) !important;
  border-radius: var(--ui-radius-card) !important;
  box-shadow: var(--ui-shadow-dialog) !important;
}

:deep(.ui-app-modal .n-card-header) {
  min-height: 0;
  padding: 20px 22px 14px;
}

:deep(.ui-app-modal .n-card-header__main) {
  color: var(--ui-text);
  font-size: 17px;
  font-weight: 650;
}

:deep(.ui-app-modal .n-card__content) { padding: 0 22px 20px; }

:deep(.ui-app-modal .n-card__footer) {
  margin: 0;
  padding: 14px 22px 20px;
  border-top: 1px solid var(--ui-divider);
  background: var(--ui-surface-muted);
}

.ui-app-modal__footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

@media (max-width: 639px) {
  :deep(.ui-app-modal.n-card) { border-radius: 17px !important; }
  :deep(.ui-app-modal .n-card-header) { padding: 18px 18px 12px; }
  :deep(.ui-app-modal .n-card__content) { padding: 0 18px 18px; }
  :deep(.ui-app-modal .n-card__footer) { padding: 13px 18px 18px; }
}
</style>
