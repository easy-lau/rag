<template>
  <span class="score-tag" :title="details" :aria-label="details">
    <span>{{ metricLabel }}</span>
    <span class="score-tag__value">{{ display }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  score: { type: [Number, String], default: null },
  retrievalScore: { type: [Number, String], default: null },
  rerankScore: { type: [Number, String], default: null },
  topicRelevance: { type: [Number, String], default: null },
  answerSupport: { type: [Number, String], default: null },
})

function finiteNumber(value) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function formatScore(value) {
  const parsed = finiteNumber(value)
  return parsed === null ? '--' : parsed.toFixed(2)
}

const primaryScore = computed(() => (
  finiteNumber(props.topicRelevance)
  ?? finiteNumber(props.rerankScore)
  ?? finiteNumber(props.score)
  ?? finiteNumber(props.retrievalScore)
))
const hasTopicScore = computed(() => (
  finiteNumber(props.topicRelevance) !== null || finiteNumber(props.rerankScore) !== null
))
const metricLabel = computed(() => (hasTopicScore.value ? '主题相关' : '召回排序'))
const display = computed(() => formatScore(primaryScore.value))
const details = computed(() => {
  const values = [`${metricLabel.value}：${display.value}`]
  const answerSupport = finiteNumber(props.answerSupport)
  const retrievalScore = finiteNumber(props.retrievalScore)
  if (answerSupport !== null) values.push(`回答支持：${formatScore(answerSupport)}`)
  if (retrievalScore !== null && retrievalScore !== primaryScore.value) {
    values.push(`原始召回：${formatScore(retrievalScore)}`)
  }
  const explanation = hasTopicScore.value
    ? '主题相关度不代表版本适用或答案可信度。'
    : '召回排序分只用于当前检索方式内部排序，不是概率，也不代表答案可信度。'
  return `${explanation}${values.join('；')}`
})
</script>

<style scoped>
.score-tag {
  display: inline-flex;
  max-width: 100%;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-pill);
  background: var(--ui-surface-muted);
  padding: 3px 7px;
  color: var(--ui-text-secondary);
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
  white-space: nowrap;
}

.score-tag__value {
  color: var(--ui-text);
  font-variant-numeric: tabular-nums;
}
</style>
