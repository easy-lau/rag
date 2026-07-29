<template>
  <n-modal
    :show="show"
    to="#app"
    :mask-closable="!loading"
    :close-on-esc="!loading"
    @update:show="handleShowUpdate"
  >
    <section
      class="ui-danger-confirm"
      role="alertdialog"
      aria-modal="true"
      :aria-labelledby="titleId"
      :aria-describedby="description || $slots.default ? descriptionId : undefined"
      :aria-busy="loading"
    >
      <header class="ui-danger-confirm__header">
        <div class="ui-danger-confirm__icon" aria-hidden="true">
          <n-icon :size="21"><TrashOutline /></n-icon>
        </div>
        <div class="ui-danger-confirm__heading">
          <span class="ui-danger-confirm__eyebrow">不可撤销操作</span>
          <h3 :id="titleId" class="ui-danger-confirm__title">{{ title }}</h3>
        </div>
      </header>

      <div v-if="subject || description || $slots.default" class="ui-danger-confirm__body">
        <div v-if="subject" class="ui-danger-confirm__target">
          <span class="ui-danger-confirm__target-label">即将删除</span>
          <strong class="ui-danger-confirm__target-value">{{ subject }}</strong>
        </div>

        <div
          v-if="description || $slots.default"
          :id="descriptionId"
          class="ui-danger-confirm__description"
        >
          <n-icon :size="17" aria-hidden="true"><WarningOutline /></n-icon>
          <div><slot>{{ description }}</slot></div>
        </div>
      </div>

      <footer class="ui-danger-confirm__footer">
        <span class="ui-danger-confirm__footer-hint">请确认对象和影响范围后再继续</span>
        <div class="ui-danger-confirm__actions">
          <n-button secondary :disabled="loading" autofocus @click="requestClose">取消</n-button>
          <n-button type="error" :loading="loading" :disabled="loading" @click="emit('confirm')">
            <template #icon><n-icon><TrashOutline /></n-icon></template>
            {{ confirmText }}
          </n-button>
        </div>
      </footer>
    </section>
  </n-modal>
</template>

<script setup>
import { useId } from 'vue'
import { NButton, NIcon, NModal } from 'naive-ui'
import { TrashOutline, WarningOutline } from '@vicons/ionicons5'

const props = defineProps({
  show: { type: Boolean, default: false },
  title: { type: String, default: '确认删除？' },
  subject: { type: String, default: '' },
  description: { type: String, default: '删除后不可恢复，请确认是否继续。' },
  confirmText: { type: String, default: '永久删除' },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['update:show', 'confirm', 'cancel'])
const componentId = useId()
const titleId = `${componentId}-title`
const descriptionId = `${componentId}-description`

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
  display: flex;
  flex-direction: column;
  width: min(468px, calc(100vw - 32px));
  max-height: calc(100vh - 32px);
  max-height: calc(100dvh - 32px);
  overflow: hidden;
  color: var(--ui-text, #1e293b);
  background: var(--ui-surface-raised, #ffffff);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-dialog, 20px);
  box-shadow: var(--ui-shadow-dialog, 0 24px 48px rgb(15 23 42 / 0.18));
}

.ui-danger-confirm__header {
  display: flex;
  flex: 0 0 auto;
  align-items: flex-start;
  gap: 14px;
  padding: 24px 24px 18px;
}

.ui-danger-confirm__icon {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  color: var(--ui-danger, #dc2626);
  background: var(--ui-danger-subtle, #fef2f2);
  border: 1px solid color-mix(in srgb, var(--ui-danger, #dc2626) 20%, transparent);
  border-radius: 13px;
}

.ui-danger-confirm__heading {
  min-width: 0;
  padding-top: 1px;
}

.ui-danger-confirm__eyebrow {
  display: block;
  margin-bottom: 3px;
  color: var(--ui-danger, #dc2626);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.4;
  letter-spacing: .12em;
}

.ui-danger-confirm__title {
  margin: 0;
  color: var(--ui-text, #1e293b);
  font-size: 18px;
  font-weight: 680;
  line-height: 1.4;
}

.ui-danger-confirm__body {
  display: grid;
  min-height: 0;
  overflow-y: auto;
  gap: 10px;
  padding: 0 24px 22px;
}

.ui-danger-confirm__target {
  min-width: 0;
  padding: 12px 14px;
  background: var(--ui-surface-muted, #f4f7fb);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-popover, 12px);
}

.ui-danger-confirm__target-label {
  display: block;
  margin-bottom: 4px;
  color: var(--ui-text-tertiary, #8190a5);
  font-size: 11px;
  font-weight: 600;
  line-height: 1.4;
}

.ui-danger-confirm__target-value {
  display: block;
  color: var(--ui-text, #1e293b);
  font-size: 14px;
  font-weight: 650;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.ui-danger-confirm__description {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 11px 13px;
  color: var(--ui-text-secondary, #64748b);
  background: var(--ui-danger-subtle, #fef2f2);
  border-radius: var(--ui-radius-popover, 12px);
  font-size: 13px;
  line-height: 1.6;
}

.ui-danger-confirm__description > .n-icon {
  flex: 0 0 auto;
  margin-top: 2px;
  color: var(--ui-danger, #dc2626);
}

.ui-danger-confirm__footer {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 16px;
  padding: 15px 24px 20px;
  background: var(--ui-surface-muted, #f4f7fb);
  border-top: 1px solid var(--ui-divider, #e8eef5);
}

.ui-danger-confirm__footer-hint {
  margin-right: auto;
  color: var(--ui-text-tertiary, #8190a5);
  font-size: 11px;
  line-height: 1.45;
}

.ui-danger-confirm__actions {
  display: flex;
  flex: 0 0 auto;
  gap: 8px;
}

.ui-danger-confirm__actions :deep(.n-button) {
  min-height: 36px;
  border-radius: var(--ui-radius-control, 10px);
}

.ui-danger-confirm__actions :deep(.n-button:first-child) { min-width: 88px; }
.ui-danger-confirm__actions :deep(.n-button:last-child) { min-width: 112px; }

@media (max-width: 639px) {
  .ui-danger-confirm {
    width: min(100vw - 24px, 468px);
    max-height: calc(100vh - 24px);
    max-height: calc(100dvh - 24px);
    border-radius: 18px;
  }

  .ui-danger-confirm__header { padding: 20px 18px 16px; }
  .ui-danger-confirm__body { padding: 0 18px 18px; }

  .ui-danger-confirm__footer {
    display: block;
    padding: 14px 18px 18px;
  }

  .ui-danger-confirm__footer-hint { display: none; }
  .ui-danger-confirm__actions { width: 100%; }
  .ui-danger-confirm__actions :deep(.n-button) {
    flex: 1 1 0;
    min-width: 0;
    min-height: 40px;
  }
}
</style>
