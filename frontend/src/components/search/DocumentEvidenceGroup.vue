<template>
  <article class="evidence-document">
    <div class="evidence-document__header">
      <button
        type="button"
        class="evidence-document__toggle"
        :aria-expanded="expanded"
        :aria-controls="panelId"
        :aria-label="`${expanded ? '收起' : '展开'}${filename}的${fragmentCount}个${fragmentLabel}`"
        @click="expanded = !expanded"
      >
        <span class="evidence-document__rank">{{ rank }}</span>
        <FileTypeIcon :type="group.file_type" class="evidence-document__file-icon" />
        <span class="evidence-document__identity">
          <span class="evidence-document__title" :title="filename">{{ filename }}</span>
          <span class="evidence-document__summary">
            {{ fragmentCount }} 个{{ fragmentLabel }}
            <template v-if="sectionSummary"> · {{ sectionSummary }}</template>
          </span>
        </span>
        <n-icon
          :size="15"
          class="evidence-document__chevron"
          :class="{ 'evidence-document__chevron--open': expanded }"
          aria-hidden="true"
        >
          <ChevronDownOutline />
        </n-icon>
      </button>

      <n-button
        v-if="canPreviewDocument"
        quaternary
        size="tiny"
        class="evidence-document__full-button"
        :aria-label="`查看${filename}全文`"
        @click="previewDocument"
      >
        全文
      </n-button>
    </div>

    <div
      v-show="expanded"
      :id="panelId"
      class="evidence-document__fragments"
      role="region"
      :aria-label="`${filename}命中的${fragmentCount}个${fragmentLabel}`"
    >
      <DocumentResultItem
        v-for="item in group.items"
        :key="item._chunkKey"
        :item="item"
        :rank="item._resultRank"
        :preview-enabled="previewEnabled"
        fragment-mode
        @preview="previewFragment"
      />
    </div>
  </article>
</template>

<script setup>
import { computed, getCurrentInstance, ref, watch } from 'vue'
import { NButton, NIcon } from 'naive-ui'
import { ChevronDownOutline } from '@vicons/ionicons5'
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import DocumentResultItem from './DocumentResultItem.vue'
import { evidenceSectionLabel } from '@/utils/evidenceDocuments'

const props = defineProps({
  group: { type: Object, required: true },
  rank: { type: Number, required: true },
  previewEnabled: { type: Boolean, default: true },
  defaultExpanded: { type: Boolean, default: false },
  fragmentLabel: { type: String, default: '命中片段' },
  idPrefix: { type: String, default: 'evidence' },
})

const emit = defineEmits(['preview'])
const instanceUid = getCurrentInstance()?.uid ?? props.rank
const expanded = ref(props.defaultExpanded)
watch(() => props.defaultExpanded, value => {
  expanded.value = value
})
const filename = computed(() => props.group?.filename || '未命名文档')
const fragmentCount = computed(() => props.group?.items?.length || 0)
const canPreviewDocument = computed(() => Boolean(
  props.previewEnabled && props.group?.kb_id && props.group?.doc_id,
))
const panelId = computed(() => {
  const key = `${props.idPrefix}-${instanceUid}-${props.group?.key || `group-${props.rank}`}`
    .replace(/[^a-zA-Z0-9_-]/g, '-')
  return `evidence-document-${key}`
})
const sectionSummary = computed(() => {
  const labels = [...new Set(
    (props.group?.items || []).map(evidenceSectionLabel).filter(Boolean),
  )]
  if (!labels.length) return ''
  if (labels.length === 1) return labels[0]
  return `${labels[0]}等 ${labels.length} 个位置`
})

function previewDocument() {
  const representative = props.group?.items?.[0]
  if (!representative || !canPreviewDocument.value) return
  emit('preview', { ...representative, _previewMode: 'document' })
}

function previewFragment(source) {
  emit('preview', { ...source, _previewMode: 'fragment' })
}
</script>

<style scoped>
.evidence-document {
  overflow: hidden;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-popover);
  background: var(--ui-surface);
}

.evidence-document + .evidence-document {
  margin-top: 8px;
}

.evidence-document__header {
  display: flex;
  min-width: 0;
  align-items: stretch;
  background: var(--ui-bg-subtle);
}

.evidence-document__toggle {
  display: flex;
  min-width: 0;
  min-height: 46px;
  flex: 1 1 auto;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  padding: 9px 8px 9px 10px;
  color: inherit;
  text-align: left;
  transition: background-color 150ms ease;
}

.evidence-document__toggle:hover {
  background: var(--ui-surface-hover);
}

.evidence-document__toggle:focus-visible,
.evidence-document__full-button:focus-visible {
  position: relative;
  z-index: 1;
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: -2px;
  box-shadow: var(--ui-focus-ring);
}

.evidence-document__rank {
  width: 16px;
  flex: 0 0 16px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  font-weight: 700;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.evidence-document__file-icon {
  flex: 0 0 auto;
}

.evidence-document__identity {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  gap: 2px;
}

.evidence-document__title {
  overflow: hidden;
  color: var(--ui-text);
  font-size: 13px;
  font-weight: 650;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-document__summary {
  overflow: hidden;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-document__chevron {
  flex: 0 0 auto;
  color: var(--ui-icon);
  transition: transform 150ms ease;
}

.evidence-document__chevron--open {
  transform: rotate(180deg);
}

.evidence-document__full-button {
  min-width: 46px;
  align-self: stretch;
  border-left: 1px solid var(--ui-divider);
  border-radius: 0;
  color: var(--ui-primary);
}

.evidence-document__fragments {
  border-top: 1px solid var(--ui-divider);
  padding: 4px;
}

@media (max-width: 639px) {
  .evidence-document__toggle {
    min-height: 50px;
  }

  .evidence-document__full-button {
    min-width: 54px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .evidence-document__toggle,
  .evidence-document__chevron {
    transition: none;
  }
}
</style>
