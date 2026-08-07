<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="space-y-5">
      <SurfaceCard class="space-y-4">
        <n-input v-model:value="query" type="textarea" :rows="3" placeholder="输入测试查询语句..." />
        <div class="flex flex-wrap items-center gap-3">
          <n-select v-model:value="config.method" :options="methodOptions" class="w-36" />
          <n-input-number v-model:value="config.top_k" :min="1" :max="20" class="w-40">
            <template #prefix>Top K:</template>
          </n-input-number>
          <div class="flex items-center gap-1.5">
            <span class="text-sm text-gray-600 dark:text-gray-400">重排</span>
            <n-tooltip trigger="hover" placement="top">
              <template #trigger>
                <n-icon :size="15" class="text-gray-400 cursor-help"><HelpCircleOutline /></n-icon>
              </template>
              <div class="max-w-xs text-xs leading-relaxed">
                重排（Rerank）：先快速召回候选，再分别评估主题相关度、回答支持度和产品/版本约束。<br>
                · 开启：结果更精准，但多一次模型调用、略慢。<br>
                · 关闭：直接用初步检索结果，更快但可能掺入不相关内容。
              </div>
            </n-tooltip>
            <n-switch v-model:value="config.rerank" />
          </div>
          <n-select v-model:value="config.knowledge_base_ids" :options="kbOptions" multiple placeholder="选择知识库" class="w-52" />
          <n-button type="primary" :loading="loading" @click="runSearch">开始检索</n-button>
        </div>
      </SurfaceCard>

      <SurfaceCard v-if="results.length || searchMeta.evidence_status" class="space-y-4">
        <div class="flex flex-wrap items-center gap-2 text-sm text-[var(--ui-text-secondary)]">
          <span>共 {{ results.length }} 条结果，耗时 {{ elapsed }}ms</span>
          <n-tag v-if="searchMeta.evidence_status" size="small" round :bordered="false" :type="evidenceStatusMeta.type">
            {{ evidenceStatusMeta.label }}
          </n-tag>
          <span v-if="searchMeta.trace_id" class="break-all text-xs text-[var(--ui-text-tertiary)]" :title="searchMeta.trace_id">
            Trace {{ searchMeta.trace_id }}
          </span>
        </div>
        <div class="space-y-3">
          <article v-for="(r, i) in results" :key="r.id"
            class="rounded-[var(--ui-radius-card)] border border-[var(--ui-border)] bg-[var(--ui-surface-muted)] p-4">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div class="flex min-w-0 flex-wrap items-center gap-2">
                <span class="text-xs font-bold text-gray-400">#{{ i + 1 }}</span>
                <FileTypeIcon :type="r.file_type" />
                <span class="text-sm font-medium text-gray-700 dark:text-gray-300">{{ r.filename }}</span>
                <n-tag
                  v-if="evidenceRoleMeta(r).label"
                  :type="evidenceRoleMeta(r).type"
                  size="small"
                  round
                  :bordered="false"
                >
                  {{ evidenceRoleMeta(r).label }}
                </n-tag>
              </div>
              <ScoreTag
                :score="r.score"
                :retrieval-score="r.retrieval_score"
                :topic-relevance="r.topic_relevance"
                :answer-support="r.answer_support"
              />
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{{ r.content }}</p>
            <div v-if="(r.answer_support !== null && r.answer_support !== undefined) || r.constraint_status" class="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-[var(--ui-text-tertiary)]">
              <span v-if="r.answer_support !== null && r.answer_support !== undefined">回答支持 {{ scoreDisplay(r.answer_support) }}</span>
              <span v-if="constraintLabel(r)" :title="r.constraint_reason || ''">{{ constraintLabel(r) }}</span>
            </div>
          </article>
          <div v-if="!results.length" class="rounded-[var(--ui-radius-card)] border border-dashed border-[var(--ui-border)] px-4 py-10 text-center text-sm text-[var(--ui-text-tertiary)]">
            本次没有返回可展示的检索候选。
          </div>
        </div>
      </SurfaceCard>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { NInput, NInputNumber, NSelect, NSwitch, NButton, NTooltip, NIcon, NTag } from 'naive-ui'
import { HelpCircleOutline } from '@vicons/ionicons5'
import { searchTest } from '@/api/search'
import { useKnowledgeStore } from '@/stores/knowledge'
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import ScoreTag from '@/components/common/ScoreTag.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import { evidenceStatusMeta as getEvidenceStatusMeta } from '@/utils/evidenceStatus'

const kbStore = useKnowledgeStore()
const query = ref('')
const loading = ref(false)
const results = ref([])
const elapsed = ref(0)
const searchMeta = ref({})
const config = ref({ method: 'hybrid', top_k: 5, rerank: true, knowledge_base_ids: [] })

const kbOptions = computed(() => kbStore.list.map(kb => ({ label: kb.name, value: kb.id })))
const methodOptions = [
  { label: '混合检索', value: 'hybrid' },
  { label: '向量检索', value: 'vector' },
  { label: '关键词检索（全文 + 词面）', value: 'keyword' },
]

const evidenceStatusMeta = computed(() => {
  const meta = getEvidenceStatusMeta(searchMeta.value.evidence_status)
  return meta
    ? { label: meta.label, type: meta.tagType }
    : { label: searchMeta.value.evidence_status, type: 'default' }
})

function evidenceRoleMeta(result) {
  return ({
    direct: { label: '回答依据', type: 'success' },
    related: { label: '相近资料', type: 'warning' },
    irrelevant: { label: '非回答依据', type: 'default' },
  })[result?.evidence_role] || { label: result?.rerank_status === 'unverified' ? '待验证' : '', type: 'default' }
}

function scoreDisplay(value) {
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(2) : '--'
}

function constraintLabel(result) {
  return ({
    exact: '版本精确匹配',
    compatible: '版本明确兼容',
    unknown: '版本适用性待确认',
    mismatch: '版本或产品不匹配',
    neutral: '无显式版本约束',
  })[result?.constraint_status] || ''
}

onMounted(() => kbStore.fetchList())

async function runSearch() {
  if (!query.value.trim()) return
  loading.value = true
  searchMeta.value = {}
  const t0 = Date.now()
  try {
    const res = await searchTest({ query: query.value, ...config.value })
    results.value = res.results
    searchMeta.value = res.search_meta || {}
    elapsed.value = Date.now() - t0
  } finally {
    loading.value = false
  }
}
</script>
