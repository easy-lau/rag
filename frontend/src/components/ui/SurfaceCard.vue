<template>
  <component :is="as" class="ui-surface-card" :class="`ui-surface-card--${padding}`">
    <slot />
  </component>
</template>

<script setup>
defineProps({
  as: { type: String, default: 'section' },
  padding: {
    type: String,
    default: 'md',
    validator: value => ['none', 'sm', 'md', 'lg'].includes(value),
  },
})
</script>

<style scoped>
.ui-surface-card {
  background: var(--ui-surface, #ffffff);
  border: 1px solid var(--ui-border, #e2e8f0);
  border-radius: var(--ui-radius-card, 16px);
  box-shadow: var(--ui-shadow-card, 0 1px 2px rgb(15 23 42 / 0.04));
}

.ui-surface-card--none { padding: 0; }
.ui-surface-card--sm { padding: 16px; }
.ui-surface-card--md { padding: 20px; }
.ui-surface-card--lg { padding: 24px; }

@media (max-width: 639px) {
  .ui-surface-card--md,
  .ui-surface-card--lg { padding: 16px; }
}
</style>
