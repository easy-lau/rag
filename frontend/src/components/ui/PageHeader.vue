<template>
  <header class="ui-page-header">
    <div class="ui-page-header__copy">
      <div v-if="eyebrow || $slots.eyebrow" class="ui-page-header__eyebrow">
        <slot name="eyebrow">{{ eyebrow }}</slot>
      </div>
      <h1 class="ui-page-header__title">{{ title }}</h1>
      <p v-if="description || $slots.description" class="ui-page-header__description">
        <slot name="description">{{ description }}</slot>
      </p>
    </div>

    <div v-if="$slots.actions || $slots.meta" class="ui-page-header__side">
      <div v-if="$slots.meta" class="ui-page-header__meta"><slot name="meta" /></div>
      <div v-if="$slots.actions" class="ui-page-header__actions"><slot name="actions" /></div>
    </div>
  </header>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  description: { type: String, default: '' },
  eyebrow: { type: String, default: '' },
})
</script>

<style scoped>
.ui-page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
}

.ui-page-header__copy {
  min-width: 0;
}

.ui-page-header__eyebrow {
  margin-bottom: 4px;
  color: var(--ui-primary, #3b82f6);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  line-height: 1.3;
  text-transform: uppercase;
}

.ui-page-header__title {
  margin: 0;
  color: var(--ui-text, #1e293b);
  font-size: 20px;
  font-weight: 650;
  letter-spacing: -0.018em;
  line-height: 1.35;
}

.ui-page-header__description {
  max-width: 720px;
  margin: 6px 0 0;
  color: var(--ui-text-secondary, #64748b);
  font-size: 13px;
  line-height: 1.6;
}

.ui-page-header__side {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
}

.ui-page-header__meta,
.ui-page-header__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

@media (max-width: 639px) {
  .ui-page-header {
    align-items: flex-start;
    flex-direction: column;
    gap: 12px;
  }

  .ui-page-header__side,
  .ui-page-header__meta,
  .ui-page-header__actions {
    width: 100%;
    justify-content: flex-start;
  }

  .ui-page-header__actions :deep(.n-button) {
    min-height: 40px;
  }
}
</style>
