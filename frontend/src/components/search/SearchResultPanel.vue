<template>
  <aside class="w-full h-full flex flex-col overflow-hidden border-l border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
    <!-- Results header -->
    <div class="shrink-0 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <span class="font-medium text-sm text-gray-800 dark:text-gray-200">检索结果</span>
        <div class="flex items-center gap-1.5">
          <n-tag v-if="searchStore.hasResultEvent" size="small" :type="retrievalState.type" :bordered="false" round>
            {{ retrievalState.label }}
          </n-tag>
          <n-button
            v-if="inDrawer"
            quaternary
            circle
            size="small"
            aria-label="关闭检索结果"
            title="关闭检索结果"
            @click="$emit('close')"
          >
            <template #icon><n-icon :size="18"><CloseOutline /></n-icon></template>
          </n-button>
        </div>
      </div>
      <div v-if="searchStore.intentDecision" class="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
        <n-tag type="info" size="small" :bordered="false" round>
          {{ searchStore.intentDecision.intent_name || searchStore.intentDecision.intent_code || '智能路由' }}
        </n-tag>
        <span>{{ responseModeLabel }}</span>
        <span aria-hidden="true">·</span>
        <span>{{ retrievalPolicyLabel }}</span>
      </div>
      <p
        v-if="decisionReason"
        class="mt-1.5 truncate text-xs text-gray-400"
        :title="decisionReason"
      >
        策略原因：{{ decisionReasonLabel }}
      </p>
    </div>

    <!-- Result list -->
    <div class="min-h-0 flex-1 overflow-y-auto px-2 py-2">
      <template v-if="searchStore.results.length">
        <DocumentResultItem
          v-for="(item, i) in searchStore.results"
          :key="item.id"
          :item="item"
          :rank="i + 1"
          @preview="$emit('preview', $event)"
        />
      </template>
      <div v-else class="flex h-40 flex-col items-center justify-center px-5 text-center text-gray-400">
        <n-icon :size="32" class="mb-2"><SearchOutline /></n-icon>
        <span class="text-sm text-gray-500 dark:text-gray-300">{{ emptyResultText }}</span>
        <span v-if="emptyResultHint" class="mt-1 text-xs leading-relaxed text-gray-400">{{ emptyResultHint }}</span>
      </div>
    </div>

    <!-- Search process -->
    <div class="shrink-0 px-4 py-3 border-t border-gray-200 dark:border-gray-700">
      <div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">检索过程</div>
      <SearchProcess />

      <!-- Meta info -->
      <div v-if="showExecutionMeta" class="mt-3 space-y-1.5">
        <div class="flex justify-between gap-3 text-xs">
          <span class="shrink-0 text-gray-500">执行状态</span>
          <span class="text-right text-gray-700 dark:text-gray-300">{{ retrievalExecutionLabel }}</span>
        </div>
        <div class="flex justify-between text-xs">
          <span class="text-gray-500">{{ searchStore.searchMeta.retrieval_executed === true ? '检索方式' : '配置方式' }}</span>
          <span class="text-gray-700 dark:text-gray-300">{{ methodLabel }}</span>
        </div>
        <div v-if="searchStore.searchMeta.top_k !== undefined" class="flex justify-between text-xs">
          <span class="text-gray-500">Top K</span>
          <span class="text-gray-700 dark:text-gray-300">{{ searchStore.searchMeta.top_k }}</span>
        </div>
        <div v-if="['hit', 'unverified'].includes(searchStore.searchMeta.evidence_status) && hitCount" class="flex justify-between text-xs">
          <span class="text-gray-500">{{ searchStore.searchMeta.evidence_status === 'hit' ? '有效命中' : '召回候选' }}</span>
          <span class="text-gray-700 dark:text-gray-300">{{ hitCount }} 条</span>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { NButton, NIcon, NTag } from 'naive-ui'
import { CloseOutline, SearchOutline } from '@vicons/ionicons5'
import { useSearchStore } from '@/stores/search'
import DocumentResultItem from './DocumentResultItem.vue'
import SearchProcess from './SearchProcess.vue'

const searchStore = useSearchStore()
defineProps({
  inDrawer: { type: Boolean, default: false },
})
defineEmits(['close', 'preview'])
const methodLabel = computed(() => {
  const m = { hybrid: '混合检索（向量+关键词）', vector: '向量检索', keyword: '关键词检索' }
  return m[searchStore.searchMeta.method] || (searchStore.searchMeta.retrieval_executed === true ? '未记录' : '—')
})

const responseModeLabel = computed(() => {
  const mode = searchStore.intentDecision?.response_mode
  return ({
    grounded_qa: '知识库问答',
    general_chat: '通用回答',
    writing: '写作模式',
    platform_help: '平台帮助',
  })[mode] || '已完成路由'
})

const retrievalPolicyLabel = computed(() => ({
  required: '必须检索',
  optional: '按证据检索',
  skip: '跳过检索',
})[searchStore.intentDecision?.retrieval_policy] || (searchStore.intentDecision?.need_retrieval ? '需要检索' : '无需检索'))

const evidenceStatus = computed(() => searchStore.searchMeta.evidence_status || '')
const hitCount = computed(() => Number(searchStore.searchMeta.hit_count ?? searchStore.totalCount ?? 0) || 0)
const retrievalState = computed(() => ({
  skipped: { label: '已跳过', type: 'default' },
  hit: { label: `${hitCount.value} 条命中`, type: 'success' },
  no_hit: { label: '未命中', type: 'warning' },
  unverified: { label: hitCount.value ? `${hitCount.value} 条待验证` : '状态未验证', type: 'default' },
  error: { label: '检索失败', type: 'error' },
})[evidenceStatus.value] || { label: '等待执行', type: 'default' })

const decisionReason = computed(() => (
  searchStore.searchMeta.decision_reason || searchStore.intentDecision?.decision_reason || ''
))
const decisionReasonLabel = computed(() => ({
  safe_fallback: '分类异常或置信度不足，采用安全检索兜底',
  classification_pending_policy: '分类已完成，等待策略层决策',
  general_chat_disabled: '系统已关闭非检索回答',
  classified_retrieval: '意图分类明确要求知识库检索',
  exact_greeting: '明确的问候或礼貌用语',
  explicit_platform_help: '明确询问当前 RAG 平台功能',
  platform_help_scope_guard: '并非当前平台帮助，策略保护已强制检索',
  inline_writing_content: '用户已提供待处理文本，无需查询知识库',
  knowledge_dependent_writing: '写作任务依赖知识库资料，必须先检索',
  selected_knowledge_context: '已选择知识库，允许使用知识证据',
  no_selected_knowledge: '未选择知识库，按非检索模式回答',
  invalid_action_fallback: '分类动作无效，采用安全检索兜底',
  legacy_action_mapping: '历史日志按原分类动作补全执行策略',
  legacy_probe: '旧接口通过轻量判断生成检索计划',
  explicit_need_retrieval: '调用方明确指定是否检索',
  retrieval_required: '检索策略要求执行知识库检索',
  retrieval_skipped: '检索策略明确跳过知识库检索',
  optional_auto_detection: '可选检索由轻量判断决定',
})[decisionReason.value] || decisionReason.value)

const retrievalExecutionLabel = computed(() => {
  if (searchStore.searchMeta.retrieval_executed === true) return '已执行知识库检索'
  if (searchStore.searchMeta.retrieval_executed === false) return '已按策略跳过检索'
  if (evidenceStatus.value === 'error') return '检索执行失败'
  return '执行状态未记录'
})

const showExecutionMeta = computed(() => (
  searchStore.hasResultEvent || searchStore.searchMeta.method || searchStore.searchMeta.top_k !== undefined
))

const emptyResultText = computed(() => {
  return ({
    skipped: '本次已跳过知识库检索',
    no_hit: '已完成检索，但没有找到相关内容',
    unverified: '检索状态暂未确认',
    error: '知识库检索失败',
  })[evidenceStatus.value] || (searchStore.intentDecision?.need_retrieval
    ? '正在等待知识库检索结果'
    : '发送问题后显示检索结果')
})

const emptyResultHint = computed(() => ({
  skipped: '这是后端策略的最终执行结果，并非“检索后无命中”。',
  no_hit: '可以调整问法、检索标签，或确认知识库中已录入相关文档。',
  unverified: '服务端未返回完整证据状态，结果可能来自旧版本接口或请求已中断。',
  error: '请稍后重试；若持续失败，可联系管理员检查检索服务。',
})[evidenceStatus.value] || '')
</script>
