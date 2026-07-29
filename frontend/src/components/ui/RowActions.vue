<template>
  <div class="ui-row-actions" :class="{ 'ui-row-actions--center': center }" role="group" :aria-label="label">
    <slot />
  </div>
</template>

<script setup>
defineProps({
  label: { type: String, default: '行操作' },
  center: { type: Boolean, default: true },
})
</script>

<style scoped>
.ui-row-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  min-width: max-content;
}

.ui-row-actions--center {
  justify-content: center;
}

.ui-row-actions :deep(.n-button) {
  min-width: 34px;
  min-height: 30px;
  padding: 0 8px;
  border-radius: var(--ui-radius-control, 10px);
}

@media (max-width: 639px) {
  .ui-row-actions :deep(.n-button) {
    min-width: 40px;
    min-height: 36px;
  }
}
</style>
