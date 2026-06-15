<template>
  <div class="border border-gray-200 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-800 shadow-sm p-3">
    <n-input
      v-model:value="text"
      type="textarea"
      :autosize="{ minRows: 2, maxRows: 6 }"
      placeholder="请输入您的问题..."
      :bordered="false"
      class="text-sm"
      @keydown.enter.exact.prevent="handleSend"
    />
    <div class="flex items-center justify-between mt-2 pt-2 border-t border-gray-100 dark:border-gray-700">
      <div class="flex items-center gap-3 flex-1 min-w-0">
        <n-select
          v-model:value="chatStore.selectedKbIds"
          :options="kbOptions"
          size="small"
          multiple
          clearable
          placeholder="选择知识库"
          class="w-44"
          :max-tag-count="1"
        />
        <n-select
          v-model:value="chatStore.searchConfig.method"
          :options="methodOptions"
          size="small"
          class="w-40"
        />
        <n-select
          v-if="tagOptions.length"
          v-model:value="chatStore.searchConfig.tags"
          :options="tagOptions"
          size="small"
          multiple
          clearable
          placeholder="标签（软加权）"
          class="w-40"
          :max-tag-count="1"
        />
        <div class="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 shrink-0">
          <span>开启重排</span>
          <n-tooltip trigger="hover" placement="top">
            <template #trigger>
              <n-icon :size="15" class="text-gray-400 cursor-help"><HelpCircleOutline /></n-icon>
            </template>
            <div class="max-w-xs text-xs leading-relaxed">
              重排（Rerank）：先快速召回一批候选片段，再用大模型逐条评估它们与你问题的相关度并重新排序，把最相关的排前、剔除不相关的。<br>
              · 开启：回答更精准、来源更干净，但每次问答多一次模型调用，略慢、成本略增。<br>
              · 关闭：直接用初步检索结果，更快更省，但可能掺入不相关内容。
            </div>
          </n-tooltip>
          <n-switch v-model:value="chatStore.searchConfig.rerank" size="small" />
        </div>
      </div>
      <div class="flex items-center gap-2">
        <n-button v-if="chatStore.isStreaming" size="small" @click="chatStore.stopStreaming()">
          <template #icon><n-icon><StopOutline /></n-icon></template>
          停止
        </n-button>
        <n-button
          type="primary" size="small" :disabled="!text.trim() || chatStore.isStreaming"
          @click="handleSend"
        >
          <template #icon><n-icon><SendOutline /></n-icon></template>
        </n-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { NInput, NSelect, NSwitch, NButton, NIcon, NTooltip, useMessage } from 'naive-ui'
import { SendOutline, StopOutline, HelpCircleOutline } from '@vicons/ionicons5'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeStore } from '@/stores/knowledge'
import { getDocumentTags } from '@/api/knowledge'

const chatStore = useChatStore()
const kbStore = useKnowledgeStore()
const msg = useMessage()
const text = ref('')

const kbOptions = computed(() =>
  kbStore.list.map(kb => ({ label: kb.name, value: kb.id }))
)

const availableTags = ref([])
const tagOptions = computed(() => availableTags.value.map(t => ({ label: t, value: t })))

// 所选知识库变化时刷新可用标签，并把已选但不再可用的标签剔除，避免发送无效过滤
async function refreshTags() {
  if (!chatStore.selectedKbIds.length) {
    availableTags.value = []
    chatStore.searchConfig.tags = []
    return
  }
  try {
    availableTags.value = await getDocumentTags(chatStore.selectedKbIds)
  } catch {
    availableTags.value = []
  }
  const allowed = new Set(availableTags.value)
  chatStore.searchConfig.tags = (chatStore.searchConfig.tags || []).filter(t => allowed.has(t))
}

onMounted(async () => {
  await kbStore.fetchList()
  // selectedKbIds 存在 localStorage 且跨用户共用：仅保留当前用户可访问的知识库，
  // 剔除他人残留/无权访问的选择 —— 新用户登录后默认即为空。
  const allowed = new Set(kbStore.list.map(kb => kb.id))
  chatStore.selectedKbIds = chatStore.selectedKbIds.filter(id => allowed.has(id))
  await refreshTags()
})

watch(() => chatStore.selectedKbIds, refreshTags, { deep: true })

const methodOptions = [
  { label: '混合检索（向量 + 关键词）', value: 'hybrid' },
  { label: '向量检索', value: 'vector' },
  { label: '关键词检索', value: 'keyword' },
]

function handleSend() {
  if (!text.value.trim() || chatStore.isStreaming) return
  // 发送前校验：未选择知识库则提示并拦截，避免退化成无依据的自由作答
  if (!chatStore.selectedKbIds.length) {
    msg.warning('请先选择知识库')
    return
  }
  chatStore.sendMessage(text.value.trim())
  text.value = ''
}
</script>
