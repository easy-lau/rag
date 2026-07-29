<template>
  <n-modal
    :show="show"
    to="#app"
    :mask-closable="!loading"
    :close-on-esc="!loading"
    @update:show="handleShowUpdate"
  >
    <section class="ui-danger-confirm" role="alertdialog" aria-modal="true" aria-labelledby="danger-confirm-title">
      <div class="ui-danger-confirm__icon" aria-hidden="true">
        <n-icon :size="22"><TrashOutline /></n-icon>
      </div>
      <div class="ui-danger-confirm__content">
        <h3 id="danger-confirm-title" class="ui-danger-confirm__title">{{ title }}</h3>
        <p v-if="subject" class="ui-danger-confirm__subject">{{ subject }}</p>
        <div v-if="description || $slots.default" class="ui-danger-confirm__description">
          <slot>{{ description }}</slot>
        </div>
      </div>
      <footer class="ui-danger-confirm__footer">
        <n-button :disabled="loading" @click="requestClose">取消</n-button>
        <n-button type="error" :loading="loading" @click="emit('confirm')">{{ confirmText }}</n-button>
      </footer>
    </section>
  </n-modal>
</template>

<script setup>
import { NButton, NIcon, NModal } from 'naive-ui'
import { TrashOutline } from '@vicons/ionicons5'

const props = defineProps({
  show: { type: Boolean, default: false },
  title: { type: String, default: '确认删除？' },
  subject: { type: String, default: '' },
  description: { type: String, default: '删除后不可恢复，请确认是否继续。' },
  confirmText: { type: String, default: '永久删除' },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['update:show', 'confirm', 'cancel'])

function handleShowUpdate(next) {
  if (!next && props.loading) return
  emit('update:show', next)
  if (!next) emit('cancel')
}

function requestClose() {
  if (props.loading) return
  emit('update:show', false)
  emit('cancel')
}
</script>

<style scoped>
.ui-danger-confirm {
  box-sizing: border-box;
  width: min(440px, calc(100vw - 32px));
  padding: 24px;
  color: var(--ui-text, #1e293b);
  background: var(--ui-surface-raised, #ffffff);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-dialog, 20px);
  box-shadow: var(--ui-shadow-dialog, 0 24px 48px rgb(15 23 42 / 0.18));
}

.ui-danger-confirm__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  color: var(--ui-danger, #dc2626);
  background: var(--ui-danger-subtle, #fef2f2);
  border-radius: 14px;
}

.ui-danger-confirm__content { margin-top: 16px; }

.ui-danger-confirm__title {
  margin: 0;
  color: var(--ui-text, #1e293b);
  font-size: 17px;
  font-weight: 650;
  line-height: 1.45;
}

.ui-danger-confirm__subject {
  margin: 8px 0 0;
  overflow: hidden;
  color: var(--ui-text-secondary, #64748b);
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ui-danger-confirm__description {
  margin-top: 8px;
  color: var(--ui-text-secondary, #64748b);
  font-size: 13px;
  line-height: 1.65;
}

.ui-danger-confirm__footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 24px;
}

.ui-danger-confirm__footer :deep(.n-button) {
  min-height: 36px;
  border-radius: var(--ui-radius-control, 10px);
}

@media (max-width: 639px) {
  .ui-danger-confirm { padding: 20px; }
  .ui-danger-confirm__footer { margin-top: 20px; }
  .ui-danger-confirm__footer :deep(.n-button) { min-height: 40px; }
}
</style>
