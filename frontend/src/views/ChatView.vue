<template>
  <div class="flex h-full">
    <!-- Main chat area -->
    <div class="flex flex-col flex-1 min-w-0">
      <div class="chat-toolbar">
        <div class="chat-toolbar__leading">
          <div class="chat-toolbar__context" aria-label="问答工作台">
            <span class="chat-toolbar__context-dot" aria-hidden="true"></span>
            <span>对话工作台</span>
          </div>
          <!-- 桌面端已在侧栏提供主入口；移动端侧栏隐藏时保留新对话。 -->
          <n-button
            v-if="ui.isMobile"
            class="chat-toolbar__new"
            type="primary"
            size="small"
            :disabled="chatStore.isStreaming"
            title="生成中请先停止回答"
            @click="startNewConversation"
          >
            <template #icon><n-icon><AddOutline /></n-icon></template>
            新对话
          </n-button>
        </div>
        <n-button
          class="chat-toolbar__results"
          :class="{ 'is-active': showResults }"
          size="small"
          @click="showResults = ui.isMobile ? true : !showResults"
        >
          <template #icon><n-icon><SearchOutline /></n-icon></template>
          {{ showResults ? '收起检索结果' : '展开检索结果' }}
        </n-button>
      </div>

      <!-- 空会话以欢迎态引导提问；首条消息发送后切换到正常消息流。 -->
      <div
        ref="msgList"
        class="flex-1 overflow-y-auto px-4 sm:px-6 py-4"
        :class="showWelcome ? 'flex items-center' : ''"
      >
        <ChatWelcome
          v-if="showWelcome"
          :site-name="siteName"
          :user-name="userName"
          @select-example="setWelcomeQuestion"
        >
          <template #input>
            <ChatInput ref="welcomeInputRef" />
          </template>
        </ChatWelcome>

        <div v-else-if="chatStore.isConversationLoading" class="flex h-full items-center justify-center">
          <n-spin size="large" />
        </div>

        <template v-else>
          <ChatMessage
            v-for="msg in chatStore.messages"
            :key="msg.id"
            :message="msg"
            @retry="handleRetry"
            @preview="openSourcePreview"
          />
        </template>
      </div>

      <!-- 已开始的会话固定使用底部输入框，避免与欢迎态的输入框重复。 -->
      <div v-if="!showWelcome && !chatStore.isConversationLoading" class="px-4 sm:px-6 py-4 border-t border-gray-200 dark:border-gray-700">
        <div class="max-w-4xl mx-auto">
          <ChatInput />
        </div>
      </div>
    </div>

    <!-- 检索结果默认收起：桌面端内联展开，移动端以抽屉显示 -->
    <div v-if="!ui.isMobile && showResults" class="w-80 shrink-0">
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
import { useRoute, useRouter } from 'vue-router'
import { NButton, NIcon, NModal, NSpin, NDrawer, useMessage } from 'naive-ui'
import { AddOutline, SearchOutline } from '@vicons/ionicons5'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { getDocument } from '@/api/document'
import { renderDocMarkdown } from '@/utils/markdown'
import ChatMessage from '@/components/chat/ChatMessage.vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import ChatWelcome from '@/components/chat/ChatWelcome.vue'
import SearchResultPanel from '@/components/search/SearchResultPanel.vue'

const chatStore = useChatStore()
const settingsStore = useSettingsStore()
const ui = useUiStore()
const authStore = useAuthStore()
const siteStore = useSiteStore()
const route = useRoute()
const router = useRouter()
const msg = useMessage()
const msgList = ref(null)
const welcomeInputRef = ref(null)

const showWelcome = computed(() =>
  !chatStore.messages.length && !chatStore.isStreaming && !chatStore.isConversationLoading
)
const siteName = computed(() => siteStore.site_title || '知识工作台')
const userName = computed(() => authStore.user?.display_name || authStore.user?.username || '')

// 检索结果默认收起：桌面端控制右栏，移动端控制抽屉
const showResults = ref(false)

// 从历史会话回到欢迎态时收起右侧检索信息；用户仍可在欢迎态手动展开查看。
watch(showWelcome, visible => {
  if (visible) showResults.value = false
})

function normalizeConversationId(value) {
  return typeof value === 'string' && value.trim() ? value : null
}

function replaceConversationInRoute(conversationId) {
  const normalizedId = normalizeConversationId(conversationId)
  const currentId = normalizeConversationId(route.query.conversation)
  const hasConversationQuery = Object.prototype.hasOwnProperty.call(route.query, 'conversation')
  if (normalizedId === currentId && (normalizedId || !hasConversationQuery)) return

  const query = { ...route.query }
  if (normalizedId) query.conversation = normalizedId
  else delete query.conversation
  router.replace({ name: 'chat', query }).catch(() => {})
}

// 地址栏是当前会话的可恢复来源：刷新、复制链接、浏览器前进/后退都能回到同一段对话。
async function restoreConversationFromRoute(value) {
  const conversationId = normalizeConversationId(value)
  if (chatStore.isStreaming) {
    replaceConversationInRoute(chatStore.currentConvId)
    return
  }

  if (!conversationId) {
    if (Object.prototype.hasOwnProperty.call(route.query, 'conversation')) {
      replaceConversationInRoute(null)
    }
    if (chatStore.currentConvId || chatStore.messages.length || chatStore.isConversationLoading) {
      chatStore.newConversation()
    }
    return
  }

  if (conversationId === chatStore.currentConvId && !chatStore.isConversationLoading) return
  try {
    await chatStore.loadConversation(conversationId)
  } catch {
    msg.error('该历史对话无法加载，已回到新对话')
    replaceConversationInRoute(null)
  }
}

watch(() => route.query.conversation, restoreConversationFromRoute, { immediate: true })

// 新会话首次拿到后端 ID、选择历史或删除当前会话时，反向同步 URL。
watch(() => chatStore.currentConvId, conversationId => {
  replaceConversationInRoute(conversationId)
})

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

function setWelcomeQuestion(question) {
  welcomeInputRef.value?.setText(question)
}

function startNewConversation() {
  showResults.value = false
  if (chatStore.isStreaming) return

  const query = { ...route.query }
  if (!Object.prototype.hasOwnProperty.call(query, 'conversation')) {
    chatStore.newConversation()
    return
  }
  delete query.conversation
  router.push({ name: 'chat', query }).catch(() => {})
}
</script>

<style scoped>
.chat-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 58px;
  padding: 0 20px;
  border-bottom: 1px solid #e7edf5;
  background: rgba(255, 255, 255, .82);
  backdrop-filter: blur(12px);
}

.chat-toolbar__leading {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 10px;
}

.chat-toolbar__context {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #50637e;
  font-size: 12px;
  font-weight: 680;
  letter-spacing: .015em;
}

.chat-toolbar__context-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #5a94eb;
  box-shadow: 0 0 0 4px rgba(90, 148, 235, .12);
}

:deep(.chat-toolbar__new.n-button),
:deep(.chat-toolbar__results.n-button) {
  height: 34px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 650;
  letter-spacing: .01em;
}

:deep(.chat-toolbar__new.n-button) {
  box-shadow: 0 7px 15px rgba(60, 118, 220, .24);
}

:deep(.chat-toolbar__results.n-button) {
  --n-color: #f7f9fc !important;
  --n-color-hover: #eef4fe !important;
  --n-color-pressed: #e6effd !important;
  --n-border: 1px solid #e0e8f2 !important;
  --n-border-hover: 1px solid #bcd0ef !important;
  --n-border-pressed: 1px solid #9cbce8 !important;
  --n-text-color: #536781 !important;
  --n-text-color-hover: #3f70b9 !important;
  --n-text-color-pressed: #315e9f !important;
}

:deep(.chat-toolbar__results.is-active.n-button) {
  --n-color: #eaf2ff !important;
  --n-color-hover: #e4efff !important;
  --n-border: 1px solid #c6d9f7 !important;
  --n-text-color: #376ebd !important;
}

.dark .chat-toolbar {
  border-color: #334155;
  background: rgba(31, 41, 55, .84);
}

.dark .chat-toolbar__context { color: #afbdd0; }
.dark .chat-toolbar__context-dot { background: #73a9fa; box-shadow: 0 0 0 4px rgba(115, 169, 250, .12); }

.dark :deep(.chat-toolbar__results.n-button) {
  --n-color: #263242 !important;
  --n-color-hover: #2e3e55 !important;
  --n-color-pressed: #334763 !important;
  --n-border: 1px solid #3a4a62 !important;
  --n-border-hover: 1px solid #54709a !important;
  --n-border-pressed: 1px solid #6386b8 !important;
  --n-text-color: #b3c1d4 !important;
  --n-text-color-hover: #b9d3ff !important;
  --n-text-color-pressed: #c7dcff !important;
}

.dark :deep(.chat-toolbar__results.is-active.n-button) {
  --n-color: #263d62 !important;
  --n-border: 1px solid #42679f !important;
  --n-text-color: #a9caff !important;
}

@media (max-width: 639px) {
  .chat-toolbar { min-height: 54px; padding: 0 12px; }
  .chat-toolbar__context { font-size: 11px; }
  :deep(.chat-toolbar__results.n-button) { padding-right: 9px; padding-left: 9px; }
}
</style>
