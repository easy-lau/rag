<template>
  <button
    type="button"
    class="document-result"
    :class="{ 'document-result--interactive': canPreview }"
    :disabled="!canPreview"
    :aria-label="canPreview ? `${evidenceMeta.label}，预览检索文档：${item.filename || '未命名文档'}` : undefined"
    :title="canPreview ? '预览文档' : undefined"
    @click="openPreview"
  >
    <span class="document-result__rank">{{ rank }}</span>
    <FileTypeIcon :type="item.file_type" class="shrink-0" />
    <div class="min-w-0 flex-1">
      <div class="document-result__heading">
        <p class="document-result__title">{{ item.filename }}</p>
        <n-tag
          size="small"
          round
          :bordered="false"
          :type="evidenceMeta.type"
          :title="evidenceMeta.hint"
          class="shrink-0"
        >
          {{ evidenceMeta.label }}
        </n-tag>
      </div>
      <p class="document-result__content">{{ item.content }}</p>
      <div class="document-result__meta">
        <ScoreTag
          :score="item.score"
          :retrieval-score="item.retrieval_score"
          :rerank-score="item.rerank_score"
          :topic-relevance="item.topic_relevance"
          :answer-support="item.answer_support"
        />
        <span v-if="answerSupportDisplay" class="document-result__metric">
          回答支持 {{ answerSupportDisplay }}
        </span>
        <span v-if="constraintMeta" class="document-result__constraint" :title="constraintMeta.hint">
          {{ constraintMeta.label }}
        </span>
      </div>
    </div>
  </button>
</template>

<script setup>
import { computed } from 'vue'
import { NTag } from 'naive-ui'
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import ScoreTag from '@/components/common/ScoreTag.vue'

const props = defineProps({
  item: Object,
  rank: Number,
  previewEnabled: { type: Boolean, default: true },
})
const emit = defineEmits(['preview'])

const canPreview = computed(() => Boolean(
  props.previewEnabled && props.item?.kb_id && props.item?.doc_id
))
const evidenceMeta = computed(() => ({
  direct: {
    label: '回答依据',
    type: 'success',
    hint: '该资料通过了当前问题的关键约束判定，可作为回答依据。',
  },
  related: {
    label: '相近资料',
    type: 'warning',
    hint: '该资料与问题主题相关，但不能直接支撑目标版本或其他关键约束下的回答。',
  },
  irrelevant: {
    label: '非回答依据',
    type: 'default',
    hint: '该资料不应作为当前回答的依据。',
  },
})[props.item?.evidence_role] || {
  label: '待验证',
  type: 'default',
  hint: '旧版结果或服务端尚未完成证据角色判定，相关度不等于答案可信度。',
})

function scoreDisplay(value) {
  if (value === undefined || value === null || value === '') return ''
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed.toFixed(2) : ''
}

const answerSupportDisplay = computed(() => scoreDisplay(props.item?.answer_support))
const constraintMeta = computed(() => {
  const raw = typeof props.item?.constraint_status === 'string'
    ? props.item.constraint_status.trim().toLowerCase()
    : ''
  const reason = typeof props.item?.constraint_reason === 'string'
    ? props.item.constraint_reason.trim()
    : ''
  if (!raw) return null
  if (
    (raw.includes('mismatch') || raw.includes('conflict'))
    && (raw.includes('version') || reason.includes('版本'))
  ) {
    return { label: '版本不符', hint: reason || '资料版本与问题指定版本不一致。' }
  }
  if (
    (raw.includes('mismatch') || raw.includes('conflict'))
    && (raw.includes('product') || reason.includes('产品'))
  ) {
    return { label: '产品不符', hint: reason || '资料产品与问题指定产品不一致。' }
  }
  if (raw.includes('mismatch') || raw.includes('conflict')) {
    return { label: '关键约束不符', hint: reason || '资料未满足问题中的关键约束。' }
  }
  if (raw.includes('compatible')) {
    return { label: '版本兼容', hint: reason || '资料声明的适用版本范围覆盖当前问题版本。' }
  }
  if (raw.includes('exact')) {
    return { label: '版本精确匹配', hint: reason || '资料版本与问题指定版本精确一致。' }
  }
  if (raw.includes('match') || raw.includes('satisfied')) {
    return { label: '关键约束匹配', hint: reason || '资料满足当前问题中的关键约束。' }
  }
  if (raw.includes('unknown') || raw.includes('unverified') || raw.includes('unchecked')) {
    return { label: '约束待验证', hint: reason || '服务端尚未确认资料是否满足当前问题的关键约束。' }
  }
  return null
})

function openPreview() {
  if (!canPreview.value) return
  emit('preview', props.item)
}
</script>

<style scoped>
.document-result {
  display: flex;
  width: 100%;
  align-items: flex-start;
  gap: 8px;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-popover);
  background: transparent;
  padding: 12px;
  color: inherit;
  text-align: left;
  transition: border-color 150ms ease, background-color 150ms ease;
}

.document-result:disabled {
  cursor: default;
}

.document-result--interactive {
  cursor: pointer;
}

.document-result--interactive:hover {
  border-color: var(--ui-border);
  background: var(--ui-surface-hover);
}

.document-result--interactive:focus-visible {
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: -2px;
  box-shadow: var(--ui-focus-ring);
}

.document-result__rank {
  width: 16px;
  flex: 0 0 16px;
  margin-top: 3px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.document-result__heading {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.document-result__title {
  min-width: 96px;
  flex: 1 1 120px;
  overflow: hidden;
  color: var(--ui-text);
  font-size: 14px;
  font-weight: 600;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-result--interactive:hover .document-result__title {
  color: var(--ui-primary);
}

.document-result__content {
  display: -webkit-box;
  margin-top: 4px;
  overflow: hidden;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.6;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.document-result__meta {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
}

.document-result__metric,
.document-result__constraint {
  color: var(--ui-text-tertiary);
  font-size: 11px;
  line-height: 1.3;
  white-space: nowrap;
}

.document-result__constraint::before {
  margin-right: 6px;
  color: var(--ui-divider);
  content: '·';
}

@media (prefers-reduced-motion: reduce) {
  .document-result {
    transition: none;
  }
}
</style>
