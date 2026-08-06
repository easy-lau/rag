<template>
  <div class="chat-page">
    <!-- Main chat area -->
    <div class="chat-page__main">
      <!-- 空会话以欢迎态引导提问；首条消息发送后切换到正常消息流。 -->
      <div
        ref="msgList"
        class="chat-page__messages"
        :class="{ 'chat-page__messages--welcome': showWelcome }"
        @scroll.passive="handleMessageScroll"
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

        <div v-else-if="chatStore.conversationLoadError" class="flex h-full items-center justify-center py-8">
          <section class="chat-conversation-load-error" aria-live="polite">
            <p class="chat-conversation-load-error__title">暂时无法加载这段历史对话</p>
            <p class="chat-conversation-load-error__description">{{ conversationLoadErrorMessage }}</p>
            <p v-if="conversationLoadErrorMeta" class="chat-conversation-load-error__meta">{{ conversationLoadErrorMeta }}</p>
            <div class="chat-conversation-load-error__actions">
              <n-button type="primary" @click="retryConversationLoad">重新加载</n-button>
              <n-button @click="openNewConversation">新建对话</n-button>
            </div>
          </section>
        </div>

        <!-- 与底部输入区共用内容宽度，避免消息流在宽屏贴近页面两侧。 -->
        <div v-else class="chat-page__message-list">
          <ChatMessage
            v-for="msg in chatStore.messages"
            :key="msg.id"
            :message="msg"
            @retry="handleRetry"
            @clarify="handleClarification"
            @inspect="inspectMessageSearch"
            @preview="openSourcePreview"
          />
        </div>
      </div>

      <!-- 已开始的会话固定使用底部输入框，避免与欢迎态的输入框重复。 -->
      <div v-if="!showWelcome && !chatStore.isConversationLoading && !chatStore.conversationLoadError" class="chat-page__composer-dock">
        <div class="chat-page__composer-inner">
          <ChatInput />
        </div>
      </div>
    </div>

    <!-- 仅在 ≥1280px 常驻右栏；其余宽度统一在抽屉中查看，给对话内容留出可读宽度。 -->
    <div v-if="isResultsInline && ui.chatSearchOpen" id="chat-search-results" class="w-80 shrink-0">
      <SearchResultPanel @preview="openSourcePreview" />
    </div>
    <n-drawer
      v-else
      :show="ui.chatSearchOpen"
      :width="ui.isMobile ? 320 : 360"
      placement="right"
      to="#app"
      @update:show="updateResultsDrawer"
    >
      <SearchResultPanel in-drawer @close="closeResults" @preview="openSourcePreview" />
    </n-drawer>

    <!-- 来源文档只读预览（当前页弹窗，不跳转） -->
    <AppModal
      v-model:show="showPreview"
      class="source-preview-modal"
      :title="previewTitle"
      width="min(90vw, 780px)"
      :loading="previewLoading"
      @close="closePreview"
    >
      <div class="source-preview__intro">
        <div>
          <p class="source-preview__description">
            {{ previewHasFragment ? '当前来源对应一个具体命中片段，可切换查看并定位文档全文。' : '查看命中文档全文，不会离开当前对话。' }}
          </p>
          <p v-if="previewHasFragment" class="source-preview__fragment-meta">
            {{ previewFragmentName }}<template v-if="previewSectionName"> · {{ previewSectionName }}</template>
          </p>
        </div>
        <n-tag v-if="previewHasFragment" size="small" round :bordered="false" type="info">
          命中片段
        </n-tag>
      </div>

      <div class="source-preview__body">
        <div v-if="previewLoading" class="absolute inset-0 flex items-center justify-center">
          <n-spin size="large" />
        </div>
        <n-tabs
          v-else-if="previewHasFragment"
          v-model:value="previewPane"
          type="segment"
          size="small"
          class="source-preview__tabs"
          :animated="false"
        >
          <n-tab-pane name="fragment" tab="命中片段">
            <div
              class="source-preview__viewport markdown-body"
              v-html="previewFragmentRendered"
            />
          </n-tab-pane>
          <n-tab-pane name="document" tab="文档全文">
            <div class="source-preview__document-pane">
              <p
                v-if="previewLocation.text"
                class="source-preview__location"
                :class="`source-preview__location--${previewLocation.type}`"
                aria-live="polite"
              >
                {{ previewLocation.text }}
              </p>
              <div
                ref="previewDocumentBody"
                class="source-preview__viewport markdown-body"
                v-html="previewRendered"
              />
            </div>
          </n-tab-pane>
        </n-tabs>
        <div
          v-else-if="!previewLoading"
          ref="previewDocumentBody"
          class="source-preview__viewport markdown-body"
          v-html="previewRendered"
        />
      </div>
      <template #footer>
        <div class="flex w-full items-center justify-between gap-3">
          <span class="text-xs text-gray-400">{{ previewHasFragment ? '片段评分属于当前检索，不代表答案正确率' : '仅供预览' }}</span>
          <n-button :disabled="previewLoading" @click="closePreview">关闭</n-button>
        </div>
      </template>
    </AppModal>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onBeforeUnmount, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NButton, NSpin, NDrawer, NTabPane, NTabs, NTag, useMessage } from 'naive-ui'
import { useMediaQuery } from '@vueuse/core'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { getDocument } from '@/api/document'
import { renderDocMarkdown } from '@/utils/markdown'
import { recoverableRetryRequestId } from '@/utils/chatRequest'
import {
  evidenceFragmentContent,
  evidenceFragmentLabel,
  evidenceSectionLabel,
  matchingEvidenceBlockIndexes,
} from '@/utils/evidenceDocuments'
import ChatMessage from '@/components/chat/ChatMessage.vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import ChatWelcome from '@/components/chat/ChatWelcome.vue'
import SearchResultPanel from '@/components/search/SearchResultPanel.vue'
import AppModal from '@/components/ui/AppModal.vue'

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
  !chatStore.currentConvId &&
  !chatStore.messages.length &&
  !chatStore.isStreaming &&
  !chatStore.isConversationLoading &&
  !chatStore.conversationLoadError
)
const siteName = computed(() => siteStore.site_title || '知识工作台')
const userName = computed(() => authStore.user?.display_name || authStore.user?.username || '')
const conversationLoadErrorMessage = computed(() => {
  const error = chatStore.conversationLoadError
  if (error?.status === 403) return '当前账号没有查看这段会话的权限。'
  if (error?.status && error.status >= 500) return '服务暂时无法读取对话内容，请稍后重新加载。'
  if (!error?.status) return '无法连接后端服务或网络异常，请确认后端已启动后重新加载。'
  return error?.detail || '网络连接或服务响应异常。已保留当前链接，你可以重新加载或新建对话。'
})
const conversationLoadErrorMeta = computed(() => {
  const error = chatStore.conversationLoadError
  if (error?.status) return `请求状态：HTTP ${error.status}`
  if (error?.code === 'ECONNABORTED') return '请求状态：连接超时'
  if (error?.code) return `请求状态：${error.code}`
  return ''
})

// 检索结果默认收起：顶栏控制开关；只有超宽屏才常驻右栏，其他尺寸用抽屉保留主内容宽度。
const isResultsInline = useMediaQuery('(min-width: 1280px)')

// 从历史会话回到欢迎态时收起右侧检索信息，避免上一次问答的命中内容遗留在新会话里。
watch(showWelcome, visible => {
  if (visible) closeResults()
})

// 断点切换时不把已打开的固定栏突然变成抽屉（或反向），由用户按需重新打开。
watch(isResultsInline, () => {
  closeResults()
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

// 来源文档预览状态必须先于带 immediate 的路由监听器初始化。
// 否则刷新深链时 restoreConversationFromRoute() 会先调用 resetPreview()，
// 而 previewRequestId 仍处于 TDZ（暂时性死区）。
const showPreview = ref(false)
const previewLoading = ref(false)
const previewTitle = ref('')
const previewContent = ref('')
const previewSource = ref(null)
const previewPane = ref('document')
const previewDocumentBody = ref(null)
const previewLocation = ref({ type: 'neutral', text: '' })
const previewRendered = computed(() => renderDocMarkdown(previewContent.value))
const previewFragmentContent = computed(() => evidenceFragmentContent(previewSource.value))
const previewFragmentRendered = computed(() => renderDocMarkdown(previewFragmentContent.value))
const previewHasFragment = computed(() => Boolean(
  previewSource.value?._previewMode === 'fragment' && previewFragmentContent.value,
))
const previewFragmentName = computed(() => evidenceFragmentLabel(previewSource.value))
const previewSectionName = computed(() => evidenceSectionLabel(previewSource.value))
let previewRequestId = 0

// 地址栏是当前会话的可恢复来源：刷新、复制链接、浏览器前进/后退都能回到同一段对话。
let routeRestoreRequestId = 0

async function restoreConversationFromRoute(value) {
  const requestId = ++routeRestoreRequestId
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
      openNewConversation()
    }
    return
  }

  // 正常的同一路由重复触发无需重复请求；但错误态必须允许用户点击「重新加载」再次请求。
  if (
    conversationId === chatStore.currentConvId &&
    !chatStore.isConversationLoading &&
    !chatStore.conversationLoadError
  ) return

  // 这两项仅是当前页面 UI 清理，不属于历史接口请求；不要把任何 UI 运行时错误
  // 伪装成“历史对话无法加载”。
  closeResults()
  resetPreview()
  try {
    await chatStore.loadConversation(conversationId)
  } catch (error) {
    // 路由已切换时，旧请求的失败不能覆盖用户刚刚打开的新会话。
    if (requestId !== routeRestoreRequestId || normalizeConversationId(route.query.conversation) !== conversationId) return

    const status = Number(error?.response?.status)
    // 只有明确不存在或链接格式无效时才清空 URL；网络、403、5xx 都应保留深链供重试。
    if (status === 404 || status === 422) {
      msg.warning('该历史对话不存在或链接已失效，已回到新对话')
      openNewConversation()
      return
    }

    console.error('[chat] 加载历史对话失败', { conversationId, status, error })
    msg.error('历史对话暂时无法加载，已保留当前链接，可重新加载重试')
  }
}

watch(() => route.query.conversation, restoreConversationFromRoute, { immediate: true })

// 新会话首次拿到后端 ID、选择历史或删除当前会话时，反向同步 URL。
watch(() => chatStore.currentConvId, conversationId => {
  replaceConversationInRoute(conversationId)
})

async function openSourcePreview(src) {
  if (!src?.kb_id || !src?.doc_id) {
    msg.warning('该来源缺少文档信息，无法预览')
    return
  }
  const requestId = ++previewRequestId
  showPreview.value = true
  previewLoading.value = true
  previewTitle.value = src.filename || '文档预览'
  previewContent.value = ''
  previewSource.value = { ...src }
  previewPane.value = src._previewMode === 'fragment' && evidenceFragmentContent(src)
    ? 'fragment'
    : 'document'
  previewLocation.value = { type: 'neutral', text: '' }
  try {
    const doc = await getDocument(src.kb_id, src.doc_id)
    if (requestId !== previewRequestId) return
    previewContent.value = doc.raw_content || '（暂无可预览内容）'
  } catch {
    if (requestId !== previewRequestId) return
    msg.error('加载文档内容失败')
    resetPreview()
  } finally {
    if (requestId === previewRequestId) {
      previewLoading.value = false
      if (previewPane.value === 'document') await locatePreviewFragment()
    }
  }
}

async function locatePreviewFragment() {
  if (!previewHasFragment.value || previewPane.value !== 'document') return
  const requestId = previewRequestId
  previewLocation.value = { type: 'neutral', text: '正在定位命中片段…' }
  await nextTick()
  if (requestId !== previewRequestId || previewPane.value !== 'document') return

  let root = previewDocumentBody.value
  if (!root) {
    await nextTick()
    if (requestId !== previewRequestId || previewPane.value !== 'document') return
    root = previewDocumentBody.value
  }
  if (!root) {
    previewLocation.value = {
      type: 'warning',
      text: '已展示命中片段，但全文视图暂时无法完成自动定位。',
    }
    return
  }
  const blocks = [...root.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, tr, pre')]
  const indexes = matchingEvidenceBlockIndexes(
    blocks.map(block => block.textContent || ''),
    previewSource.value,
  )

  blocks.forEach(block => block.classList.remove('source-preview__matched-block'))
  if (!indexes.length) {
    previewLocation.value = {
      type: 'warning',
      text: '已展示命中片段，但该文档格式暂时无法在全文中精确定位。',
    }
    return
  }

  indexes.forEach(index => blocks[index]?.classList.add('source-preview__matched-block'))
  const firstMatch = blocks[indexes[0]]
  firstMatch?.scrollIntoView?.({ block: 'center', behavior: 'auto' })
  previewLocation.value = {
    type: 'success',
    text: `已定位并突出显示${previewFragmentName.value}对应的原文位置。`,
  }
}

watch(previewPane, pane => {
  if (pane === 'document' && !previewLoading.value) locatePreviewFragment()
})

function closePreview() {
  if (previewLoading.value) return
  resetPreview()
}

function resetPreview() {
  previewRequestId += 1
  showPreview.value = false
  previewLoading.value = false
  previewContent.value = ''
  previewSource.value = null
  previewPane.value = 'document'
  previewLocation.value = { type: 'neutral', text: '' }
}

function closeResults() {
  ui.closeChatSearch()
}

function updateResultsDrawer(show) {
  ui.chatSearchOpen = show
}

function openNewConversation() {
  closeResults()
  resetPreview()
  chatStore.newConversation()
  replaceConversationInRoute(null)
}

function retryConversationLoad() {
  const conversationId = normalizeConversationId(route.query.conversation) || normalizeConversationId(chatStore.currentConvId)
  if (!conversationId) {
    openNewConversation()
    return
  }
  restoreConversationFromRoute(conversationId)
}

const autoFollowLatest = ref(true)
const AUTO_SCROLL_BOTTOM_THRESHOLD = 96
let scrollFrameId = null
let scrollQueued = false
let forceScrollQueued = false

function isNearMessageListBottom() {
  const list = msgList.value
  if (!list) return true
  return list.scrollHeight - list.scrollTop - list.clientHeight <= AUTO_SCROLL_BOTTOM_THRESHOLD
}

function handleMessageScroll() {
  autoFollowLatest.value = isNearMessageListBottom()
}

// 流式增量会非常频繁：一帧最多滚动一次；用户上翻后不再强行把视角拉回底部。
function scheduleScrollToBottom({ force = false } = {}) {
  if (!force && !autoFollowLatest.value) return
  forceScrollQueued ||= force
  if (scrollQueued) return
  scrollQueued = true

  nextTick(() => {
    scrollFrameId = requestAnimationFrame(() => {
      scrollFrameId = null
      scrollQueued = false
      const shouldScroll = forceScrollQueued || autoFollowLatest.value
      forceScrollQueued = false
      if (!shouldScroll || !msgList.value) return
      msgList.value.scrollTop = msgList.value.scrollHeight
    })
  })
}

// 进入页面（重新挂载）时，已有历史消息也直接定位到最新一条，而不是停在最早；
// 同时拉取系统设置，确保「显示参考来源」开关在刷新后仍生效
onMounted(() => {
  scheduleScrollToBottom({ force: true })
  settingsStore.fetch()
})

onBeforeUnmount(() => {
  if (scrollFrameId !== null) cancelAnimationFrame(scrollFrameId)
  closeResults()
})

// 新增消息、切换会话和主动发送都会回到底部；正常流式输出只在用户仍停留在底部时跟随。
watch(() => chatStore.messages.length, () => scheduleScrollToBottom())
watch(() => chatStore.currentConvId, () => {
  autoFollowLatest.value = true
  scheduleScrollToBottom({ force: true })
})
watch(() => chatStore.isStreaming, (isStreaming, wasStreaming) => {
  if (isStreaming && !wasStreaming) {
    autoFollowLatest.value = true
    scheduleScrollToBottom({ force: true })
  }
})

watch(() => chatStore.messages[chatStore.messages.length - 1]?.content, () => scheduleScrollToBottom())

function handleRetry(answerMessage) {
  if (chatStore.isStreaming) return
  const msgs = chatStore.messages
  const answerIndex = msgs.findIndex(message => message.id === answerMessage?.id)
  const targetUser = answerIndex > -1
    ? [...msgs.slice(0, answerIndex)].reverse().find(message => message.role === 'user')
    : null
  if (!targetUser?.content) {
    msg.warning('未找到这条回答对应的问题')
    return
  }
  autoFollowLatest.value = true
  chatStore.sendMessage(targetUser.content, {
    // Transport/save recovery reuses the logical request.  A completed answer's
    // “重新生成” action intentionally gets a fresh id and a fresh model run.
    requestId: recoverableRetryRequestId(answerMessage, targetUser) || null,
  })
}

function handleClarification({ message, reply } = {}) {
  if (chatStore.isStreaming) return
  autoFollowLatest.value = true
  if (!chatStore.submitClarification(message, reply)) {
    msg.warning('该范围选择已失效，请在输入框重新说明需要查询的范围')
  }
}

function inspectMessageSearch(message) {
  if (!chatStore.restoreMessageSearch(message)) {
    msg.info('这条回答没有可恢复的检索快照')
    return
  }
  // The existing header toggle remains the single entry point; selecting a
  // historical answer simply loads its snapshot and opens that same panel.
  ui.chatSearchOpen = true
}

function setWelcomeQuestion(question) {
  welcomeInputRef.value?.setText(question)
}

</script>

<style scoped>
.chat-page {
  display: flex;
  height: 100%;
  min-height: 0;
  background: transparent;
}

.chat-page__main {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
}

.chat-page__messages {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 28px clamp(16px, 3vw, 40px);
  overscroll-behavior: contain;
}

.chat-page__messages--welcome {
  display: flex;
  align-items: center;
}

.chat-page__message-list,
.chat-page__composer-inner {
  width: min(100%, 920px);
  margin: 0 auto;
}

.chat-page__composer-dock {
  flex: 0 0 auto;
  border-top: 1px solid var(--ui-divider);
  background: linear-gradient(180deg, color-mix(in srgb, var(--ui-surface) 74%, transparent), var(--ui-surface));
  padding: 14px clamp(16px, 3vw, 40px) 16px;
}

.chat-conversation-load-error {
  width: min(100%, 480px);
  padding: 24px;
  color: var(--ui-text);
  text-align: center;
  background: var(--ui-surface);
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
}

.chat-conversation-load-error__title {
  margin: 0;
  font-size: 16px;
  font-weight: 650;
  line-height: 1.5;
}

.chat-conversation-load-error__description {
  margin: 8px 0 0;
  color: var(--ui-text-secondary);
  font-size: 13px;
  line-height: 1.65;
}

.chat-conversation-load-error__meta {
  margin: 8px 0 0;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  line-height: 1.5;
}

.chat-conversation-load-error__actions {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-top: 20px;
}

.source-preview__intro {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

:global(.source-preview-modal.ui-app-modal.n-card) {
  display: flex;
  max-height: calc(100dvh - 16px);
  flex-direction: column;
}

:global(.source-preview-modal.ui-app-modal .n-card__content) {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
}

:global(.source-preview-modal.ui-app-modal .n-card-header),
:global(.source-preview-modal.ui-app-modal .n-card__footer) {
  flex: 0 0 auto;
}

.source-preview__description {
  margin: 0;
  color: var(--ui-text-secondary);
  font-size: 13px;
  line-height: 1.6;
}

.source-preview__fragment-meta {
  margin: 3px 0 0;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  line-height: 1.5;
}

.source-preview__body {
  position: relative;
  min-height: min(64dvh, 560px);
}

.source-preview__tabs :deep(.n-tabs-nav) {
  margin-bottom: 10px;
}

.source-preview__document-pane {
  display: flex;
  height: min(62dvh, 600px);
  min-height: min(320px, 50dvh);
  flex-direction: column;
  gap: 8px;
}

.source-preview__viewport {
  height: min(64dvh, 620px);
  min-height: min(320px, 50dvh);
  overflow: auto;
  overscroll-behavior: contain;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface);
  padding: 16px;
  color: var(--ui-text);
}

.source-preview__document-pane .source-preview__viewport {
  min-height: 0;
  flex: 1 1 auto;
  height: auto;
}

.source-preview__location {
  flex: 0 0 auto;
  margin: 0;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface-muted);
  padding: 7px 9px;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.source-preview__location--success {
  border-color: color-mix(in srgb, var(--ui-success) 35%, var(--ui-border));
  color: var(--ui-success);
}

.source-preview__location--warning {
  border-color: color-mix(in srgb, var(--ui-warning) 38%, var(--ui-border));
  color: var(--ui-warning);
}

.source-preview__viewport :deep(.source-preview__matched-block) {
  scroll-margin-block: 28px;
  border-radius: 6px;
  outline: 2px solid var(--ui-warning);
  outline-offset: 3px;
  background: color-mix(in srgb, var(--ui-warning) 14%, transparent);
}

.source-preview__viewport :deep(tr.source-preview__matched-block > th),
.source-preview__viewport :deep(tr.source-preview__matched-block > td) {
  background: color-mix(in srgb, var(--ui-warning) 14%, var(--ui-surface));
}

@media (max-width: 639px) {
  .chat-page__messages { padding: 18px 12px; }
  .chat-page__messages--welcome { align-items: flex-start; }
  .chat-page__composer-dock { padding: 10px 10px 12px; }

  .source-preview__body {
    min-height: min(52dvh, 460px);
  }

  .source-preview__document-pane,
  .source-preview__viewport {
    height: min(52dvh, 460px);
    min-height: min(220px, 44dvh);
  }

  .source-preview__intro {
    align-items: center;
  }
}

@media (max-height: 520px) {
  .source-preview__intro {
    margin-bottom: 8px;
  }

  .source-preview__body {
    min-height: 0;
  }

  .source-preview__document-pane,
  .source-preview__viewport {
    height: 40dvh;
    min-height: 120px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .source-preview__viewport {
    scroll-behavior: auto;
  }
}
</style>
