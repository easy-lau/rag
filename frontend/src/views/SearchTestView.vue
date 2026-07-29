<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-5xl mx-auto space-y-5">
      <PageHeader title="检索测试" description="使用指定知识库、检索方式和标签，快速验证召回与重排效果。" />

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
                重排（Rerank）：先快速召回一批候选片段，再用大模型逐条评估它们与查询的相关度并重新排序，把最相关的排前、剔除不相关的。<br>
                · 开启：结果更精准，但多一次模型调用、略慢。<br>
                · 关闭：直接用初步检索结果，更快但可能掺入不相关内容。
              </div>
            </n-tooltip>
            <n-switch v-model:value="config.rerank" />
          </div>
          <n-select v-model:value="config.knowledge_base_ids" :options="kbOptions" multiple placeholder="选择知识库" class="w-52" />
          <n-select
            v-if="tagOptions.length"
            v-model:value="config.tags" :options="tagOptions"
            multiple clearable placeholder="标签（软加权，可选）" class="w-52"
          />
          <n-button type="primary" :loading="loading" @click="runSearch">开始检索</n-button>
        </div>
        <p v-if="tagOptions.length" class="text-xs leading-5 text-gray-400">
          勾选标签后再检索，可对比同一查询「选标签 vs 不选」的排序差异：命中标签的文档片段会被上浮（软加权，不排除其他结果）。
        </p>
      </SurfaceCard>

      <SurfaceCard v-if="results.length" class="space-y-4">
        <div class="text-sm text-gray-500 mb-2">共 {{ results.length }} 条结果，耗时 {{ elapsed }}ms</div>
        <div class="space-y-3">
          <article v-for="(r, i) in results" :key="r.id"
            class="rounded-[var(--ui-radius-card)] border border-[var(--ui-border)] bg-[var(--ui-surface-muted)] p-4">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="text-xs font-bold text-gray-400">#{{ i + 1 }}</span>
                <FileTypeIcon :type="r.file_type" />
                <span class="text-sm font-medium text-gray-700 dark:text-gray-300">{{ r.filename }}</span>
              </div>
              <ScoreTag :score="r.score" />
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{{ r.content }}</p>
            <div v-if="r.tags && r.tags.length" class="flex flex-wrap gap-1.5 mt-2">
              <n-tag
                v-for="t in r.tags" :key="t" size="small" round
                :type="config.tags.includes(t) ? 'success' : 'default'"
                :bordered="false"
              >{{ t }}</n-tag>
            </div>
          </article>
        </div>
      </SurfaceCard>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { NInput, NInputNumber, NSelect, NSwitch, NButton, NTooltip, NIcon, NTag } from 'naive-ui'
import { HelpCircleOutline } from '@vicons/ionicons5'
import { searchTest } from '@/api/search'
import { getDocumentTags } from '@/api/knowledge'
import { useKnowledgeStore } from '@/stores/knowledge'
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import ScoreTag from '@/components/common/ScoreTag.vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'

const kbStore = useKnowledgeStore()
const query = ref('')
const loading = ref(false)
const results = ref([])
const elapsed = ref(0)
const config = ref({ method: 'hybrid', top_k: 5, rerank: true, knowledge_base_ids: [], tags: [] })

const kbOptions = computed(() => kbStore.list.map(kb => ({ label: kb.name, value: kb.id })))
const methodOptions = [
  { label: '混合检索', value: 'hybrid' },
  { label: '向量检索', value: 'vector' },
  { label: '关键词检索', value: 'keyword' },
]

const availableTags = ref([])
const tagOptions = computed(() => availableTags.value.map(t => ({ label: t, value: t })))

// 所选知识库变化时刷新可用标签，并剔除已选但不再可用的标签
async function refreshTags() {
  if (!config.value.knowledge_base_ids.length) {
    availableTags.value = []
    config.value.tags = []
    return
  }
  try { availableTags.value = await getDocumentTags(config.value.knowledge_base_ids) }
  catch { availableTags.value = [] }
  const allowed = new Set(availableTags.value)
  config.value.tags = config.value.tags.filter(t => allowed.has(t))
}

onMounted(() => kbStore.fetchList())
watch(() => config.value.knowledge_base_ids, refreshTags, { deep: true })

async function runSearch() {
  if (!query.value.trim()) return
  loading.value = true
  const t0 = Date.now()
  try {
    const res = await searchTest({ query: query.value, ...config.value })
    results.value = res.results
    elapsed.value = Date.now() - t0
  } finally {
    loading.value = false
  }
}
</script>
