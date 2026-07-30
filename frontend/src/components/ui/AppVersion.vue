<template>
  <div class="app-version" :title="detailText" :aria-label="detailText">
    <span>版本</span>
    <span class="app-version__number">{{ versionLabel }}</span>
  </div>
</template>

<script setup>
const rawVersion = String(import.meta.env.VITE_APP_VERSION || '').trim()
const rawRevision = String(import.meta.env.VITE_APP_REVISION || '').trim()
const isDevelopment = !rawVersion || /^(dev|development|local)$/i.test(rawVersion)

const versionLabel = isDevelopment
  ? '开发版'
  : rawVersion.startsWith('v') ? rawVersion : `v${rawVersion}`

const shortRevision = rawRevision ? rawRevision.slice(0, 7) : ''
const detailText = isDevelopment
  ? '系统版本 · 本地开发版本'
  : shortRevision
    ? `系统版本 · ${versionLabel} · ${shortRevision}`
    : `系统版本 · ${versionLabel}`
</script>

<style scoped>
.app-version {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: flex-start;
  gap: 6px;
  padding: 7px 8px 0;
  color: var(--ui-text-tertiary);
  font-size: 10px;
  line-height: 1.2;
}

.app-version__number {
  flex: 0 0 auto;
  padding: 3px 7px;
  border: 1px solid var(--ui-divider);
  border-radius: var(--ui-radius-pill);
  color: var(--ui-text-tertiary);
  background: var(--ui-surface-muted);
  font-weight: 650;
  font-variant-numeric: tabular-nums;
  letter-spacing: .025em;
}
</style>
