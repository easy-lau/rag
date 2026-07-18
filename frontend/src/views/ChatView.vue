<template>
  <div class="flex h-full">
    <!-- Main chat area -->
    <div class="flex flex-col flex-1 min-w-0">
      <div class="px-4 sm:px-6 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between gap-2">
        <n-button v-if="ui.isMobile" text size="small" @click="showResults = true">
          <template #icon><n-icon><SearchOutline /></n-icon></template>
          检索结果
        </n-button>
        <span v-else></span>
        <n-button text size="small" @click="chatStore.newConversation()">
          <template #icon><n-icon><AddOutline /></n-icon></template>
          新对话
        </n-button>
      </div>

      <!-- Messages -->
      <div ref="msgList" class="flex-1 overflow-y-auto px-4 sm:px-6 py-4">
        <div v-if="!chatStore.messages.length" class="flex flex-col items-center justify-center h-full text-gray-400">
          <n-icon :size="48" class="mb-3 text-blue-300"><HardwareChipOutline /></n-icon>
          <p class="text-base font-medium mb-1 text-gray-500 dark:text-gray-400">RAG 知识库问答</p>
          <p class="text-sm">从左侧选择知识库，然后提问</p>
        </div>
        <ChatMessage
          v-for="msg in chatStore.messages"
          :key="msg.id"
          :message="msg"
          @retry="handleRetry"
          @preview="openSourcePreview"
        />
      </div>

      <!-- Input -->
      <div class="px-4 sm:px-6 py-4 border-t border-gray-200 dark:border-gray-700">
        <ChatInput />
      </div>
    </div>

    <!-- 右侧检索结果：桌面内联，移动端抽屉（顶部按钮触发） -->
    <div v-if="!ui.isMobile" class="w-80 shrink-0">
      <SearchResultPanel />
    </div>
    <n-drawer v-else v-model:show="showResults" :width="320" placement="right" to="#app">
      <SearchResultPanel />
    </n-drawer>

    <!-- 来源文档只读预览（当前页弹窗，不跳转） -->
    <n-modal v-model:show="showPreview" to="#app" preset="card" :title="previewTitle" style="width: 90vw; max-width: 780px">
      <div class="relative" style="height: 70vh">
        <div v-if="previewLoading" class="absolute inset-0 flex items-center justify-center">
          <n-spin size="large" />
        </div>
        <div
          v-else
          class="markdown-body h-full overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 p-4"
          v-html="previewRendered"
        />
      </div>
      <template #footer>
        <div class="flex items-center justify-between">
          <span class="text-xs text-gray-400">仅供预览</span>
          <n-button @click="showPreview = false">关闭</n-button>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted, computed } from 'vue'
import { NButton, NIcon, NModal, NSpin, NDrawer, useMessage } from 'naive-ui'
import { AddOutline, HardwareChipOutline, SearchOutline } from '@vicons/ionicons5'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { useUiStore } from '@/stores/ui'
import { getDocument } from '@/api/document'
import { renderDocMarkdown } from '@/utils/markdown'
import ChatMessage from '@/components/chat/ChatMessage.vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import SearchResultPanel from '@/components/search/SearchResultPanel.vue'

const chatStore = useChatStore()
const settingsStore = useSettingsStore()
const ui = useUiStore()
const msg = useMessage()
const msgList = ref(null)

// 移动端：检索结果抽屉开关
const showResults = ref(false)

// 来源文档只读预览
const showPreview = ref(false)
const previewLoading = ref(false)
const previewTitle = ref('')
const previewContent = ref('')
const previewRendered = computed(() => renderDocMarkdown(previewContent.value))

async function openSourcePreview(src) {
  if (!src?.kb_id || !src?.doc_id) {
    msg.warning('该来源缺少文档信息，无法预览')
    return
  }
  showPreview.value = true
  previewLoading.value = true
  previewTitle.value = src.filename || '文档预览'
  previewContent.value = ''
  try {
    const doc = await getDocument(src.kb_id, src.doc_id)
    previewContent.value = doc.raw_content || '（暂无可预览内容）'
  } catch {
    msg.error('加载文档内容失败')
    showPreview.value = false
  } finally {
    previewLoading.value = false
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (msgList.value) msgList.value.scrollTop = msgList.value.scrollHeight
  })
}

// 进入页面（重新挂载）时，已有历史消息也直接定位到最新一条，而不是停在最早；
// 同时拉取系统设置，确保「显示参考来源」开关在刷新后仍生效
onMounted(() => {
  scrollToBottom()
  settingsStore.fetch()
})

// 新增消息时滚到底
watch(() => chatStore.messages.length, scrollToBottom)

// 切换会话时滚到底（消息条数恰好相同时 length 不变，这里兜底）
watch(() => chatStore.currentConvId, scrollToBottom)

// 流式输出时持续滚到底
watch(() => chatStore.messages[chatStore.messages.length - 1]?.content, scrollToBottom)

function handleRetry() {
  const msgs = chatStore.messages
  const lastUser = [...msgs].reverse().find(m => m.role === 'user')
  if (lastUser) chatStore.sendMessage(lastUser.content)
}
</script>
