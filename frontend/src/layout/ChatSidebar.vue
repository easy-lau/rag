<template>
  <aside class="w-full h-full flex flex-col bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
    <!-- 品牌 -->
    <div class="px-4 py-4 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center gap-2">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="w-8 h-8 rounded-lg object-cover shrink-0"
          alt="logo"
        />
        <div v-else class="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold text-sm shrink-0">
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="min-w-0">
          <div class="font-bold text-gray-800 dark:text-white text-sm truncate">{{ siteStore.site_title }}</div>
          <div class="text-xs text-gray-500 truncate">问答工作台</div>
        </div>
      </div>
    </div>

    <!-- 会话功能：问答工作台不渲染其他业务菜单。 -->
    <section class="px-3 py-3 flex-1 min-h-0 flex flex-col">
      <n-button class="chat-sidebar__new" type="primary" block :disabled="chatStore.isStreaming" @click="startNewConversation">
        <template #icon><n-icon><AddOutline /></n-icon></template>
        新对话
      </n-button>

      <div class="flex items-center justify-between mt-5 mb-2 shrink-0">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400">对话历史</span>
        <span v-if="chatStore.conversations.length" class="text-xs text-gray-400">{{ chatStore.conversations.length }}</span>
      </div>

      <div class="overflow-y-auto space-y-0.5 flex-1 pr-0.5">
        <div v-if="!chatStore.conversations.length" class="text-xs text-gray-400 text-center py-8">暂无历史对话</div>
        <div
          v-for="conv in chatStore.conversations"
          :key="conv.id"
          class="group flex items-center gap-1 rounded-lg text-sm"
          :class="[
            conv.id === chatStore.currentConvId
              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
            : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700',
            { 'cursor-not-allowed opacity-50': chatStore.isStreaming },
          ]"
        >
          <button
            type="button"
            class="min-w-0 flex-1 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/60"
            :disabled="chatStore.isStreaming"
            :aria-current="conv.id === chatStore.currentConvId ? 'page' : undefined"
            @click="selectConversation(conv.id)"
          >
            <span class="block truncate">{{ conv.title || '未命名对话' }}</span>
          </button>
          <n-dropdown
            class="chat-sidebar__history-menu"
            trigger="click"
            placement="bottom-end"
            :options="conversationActionOptions"
            :theme-overrides="historyMenuThemeOverrides"
            @select="action => handleConversationAction(action, conv)"
          >
            <n-button
              quaternary circle size="small" class="chat-sidebar__history-action shrink-0" :disabled="chatStore.isStreaming"
              title="会话操作" :aria-label="`操作会话：${conv.title || '未命名对话'}`"
            >
              <template #icon><n-icon :size="16"><EllipsisHorizontalOutline /></n-icon></template>
            </n-button>
          </n-dropdown>
        </div>
      </div>
    </section>

    <!-- 唯一的后台入口：桌面端保留当前问答上下文，移动端直接切页。 -->
    <div v-if="canEnterAdmin" class="px-3 py-3 border-t border-gray-200 dark:border-gray-700 shrink-0">
      <button
        type="button"
        class="group flex w-full items-center gap-2.5 rounded-xl px-2 py-2 text-left transition-colors hover:bg-blue-50 dark:hover:bg-blue-900/20"
        :title="ui.isMobile ? '进入管理后台' : '在新标签打开管理后台'"
        @click="openAdmin"
      >
        <span class="w-7 h-7 rounded-lg flex items-center justify-center bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-300 group-hover:bg-white dark:group-hover:bg-gray-700 group-hover:text-blue-500 transition-colors">
          <n-icon :size="16"><SettingsOutline /></n-icon>
        </span>
        <span class="min-w-0 flex-1">
          <span class="block text-sm font-medium text-gray-600 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-300 transition-colors">管理后台</span>
          <span class="block mt-0.5 text-[11px] text-gray-400 dark:text-gray-500 truncate">知识运营与系统管理</span>
        </span>
        <n-icon :size="15" class="text-gray-300 dark:text-gray-600 group-hover:text-blue-400 transition-colors"><ChevronForwardOutline /></n-icon>
      </button>
    </div>
  </aside>

  <n-modal
    v-model:show="showRenameModal"
    preset="card"
    title="重命名会话"
    style="width: 90vw; max-width: 420px"
    :mask-closable="!isRenaming"
    :closable="!isRenaming"
    to="#app"
  >
    <p class="mb-3 text-sm text-gray-500 dark:text-gray-400">为这段对话设置一个便于查找的名称。</p>
    <n-input
      ref="renameInputRef"
      v-model:value="renameTitle"
      :maxlength="200"
      show-count
      placeholder="请输入会话名称"
      :disabled="isRenaming"
      @keydown.enter="handleRenameEnter"
    />
    <template #footer>
      <div class="flex justify-end gap-2">
        <n-button :disabled="isRenaming" @click="showRenameModal = false">取消</n-button>
        <n-button type="primary" :loading="isRenaming" @click="submitRename">保存</n-button>
      </div>
    </template>
  </n-modal>

  <n-modal
    v-model:show="showDeleteModal"
    :mask-closable="!isDeleting"
    :close-on-esc="!isDeleting"
    to="#app"
  >
    <section class="chat-delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-conversation-title">
      <div class="chat-delete-dialog__header">
        <span class="chat-delete-dialog__icon" aria-hidden="true">
          <n-icon :size="21"><TrashOutline /></n-icon>
        </span>
        <div class="min-w-0">
          <p class="chat-delete-dialog__eyebrow">危险操作</p>
          <h2 id="delete-conversation-title">删除这段对话？</h2>
        </div>
      </div>

      <p class="chat-delete-dialog__description">
        将永久删除 <strong>「{{ pendingDeleteTitle }}」</strong> 及其中的全部问答内容。
      </p>
      <div class="chat-delete-dialog__notice">
        <n-icon :size="16" aria-hidden="true"><AlertCircleOutline /></n-icon>
        <span>此操作无法撤销。</span>
      </div>

      <div class="chat-delete-dialog__actions">
        <n-button class="chat-delete-dialog__cancel" :disabled="isDeleting" @click="closeDeleteModal">取消</n-button>
        <n-button class="chat-delete-dialog__confirm" type="error" :loading="isDeleting" @click="submitDelete">永久删除</n-button>
      </div>
    </section>
  </n-modal>
</template>

<script setup>
import { computed, h, nextTick, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NButton, NDropdown, NIcon, NInput, NModal, useMessage } from 'naive-ui'
import { AddOutline, AlertCircleOutline, ChevronForwardOutline, EllipsisHorizontalOutline, PencilOutline, SettingsOutline, TrashOutline } from '@vicons/ionicons5'
import { hasAdminAccess } from '@/router/menus'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { useSiteStore } from '@/stores/site'
import { useUiStore } from '@/stores/ui'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const chatStore = useChatStore()
const siteStore = useSiteStore()
const ui = useUiStore()
const message = useMessage()

const canEnterAdmin = computed(() => hasAdminAccess(authStore))
const showRenameModal = ref(false)
const renameConversationId = ref(null)
const renameTitle = ref('')
const isRenaming = ref(false)
const renameInputRef = ref(null)
const showDeleteModal = ref(false)
const pendingDeleteConversation = ref(null)
const isDeleting = ref(false)
const pendingDeleteTitle = computed(() => pendingDeleteConversation.value?.title || '未命名对话')

const renderMenuIcon = icon => () => h(NIcon, { size: 15 }, { default: () => h(icon) })
const conversationActionOptions = [
  { label: '重命名', key: 'rename', icon: renderMenuIcon(PencilOutline) },
  { type: 'divider', key: 'conversation-actions-divider' },
  {
    label: '删除对话',
    key: 'delete',
    icon: renderMenuIcon(TrashOutline),
    props: { class: 'chat-sidebar__history-delete-option' },
  },
]
const historyMenuThemeOverrides = {
  borderRadius: '14px',
  padding: '5px',
  peers: {
    Popover: { boxShadow: '0 14px 32px rgba(35, 61, 98, .14)' },
  },
}

onMounted(() => {
  if (authStore.hasPerm('menu:chat')) {
    chatStore.loadHistory().catch(() => message.error('加载对话历史失败，请刷新重试'))
  }
})

function startNewConversation() {
  if (chatStore.isStreaming) return
  ui.mobileNavOpen = false

  const query = { ...route.query }
  if (!Object.prototype.hasOwnProperty.call(query, 'conversation')) {
    chatStore.newConversation()
    return
  }
  delete query.conversation
  router.push({ name: 'chat', query }).catch(() => {})
}

function selectConversation(conversationId) {
  if (chatStore.isStreaming) return
  ui.mobileNavOpen = false
  if (String(route.query.conversation || '') === String(conversationId)) return
  router.push({
    name: 'chat',
    query: { ...route.query, conversation: conversationId },
  })
}

function openRenameModal(conversation) {
  if (chatStore.isStreaming) return
  renameConversationId.value = conversation.id
  renameTitle.value = conversation.title || ''
  showRenameModal.value = true
  nextTick(() => renameInputRef.value?.focus?.())
}

function handleRenameEnter(event) {
  if (event.isComposing) return
  event.preventDefault()
  submitRename()
}

async function submitRename() {
  const title = renameTitle.value.trim()
  if (chatStore.isStreaming) {
    message.warning('生成回答时暂不能修改会话')
    return
  }
  if (!renameConversationId.value || !title) {
    message.warning('请输入会话名称')
    return
  }

  isRenaming.value = true
  try {
    await chatStore.renameConversation(renameConversationId.value, title)
    showRenameModal.value = false
    message.success('会话已重命名')
  } catch (error) {
    message.error(error?.response?.data?.detail || '重命名失败，请稍后重试')
  } finally {
    isRenaming.value = false
  }
}

function confirmDeleteConversation(conversation) {
  pendingDeleteConversation.value = conversation
  showDeleteModal.value = true
}

function closeDeleteModal() {
  if (isDeleting.value) return
  showDeleteModal.value = false
}

async function submitDelete() {
  const conversation = pendingDeleteConversation.value
  if (!conversation) return
  if (chatStore.isStreaming) {
    message.warning('生成回答时暂不能删除会话')
    return
  }

  isDeleting.value = true
  try {
    await chatStore.removeConversation(conversation.id)
    showDeleteModal.value = false
    message.success('对话已删除')
  } catch (error) {
    message.error(error?.response?.data?.detail || '删除对话失败，请稍后重试')
  } finally {
    isDeleting.value = false
  }
}

function handleConversationAction(action, conversation) {
  if (chatStore.isStreaming) return
  if (action === 'rename') {
    openRenameModal(conversation)
    return
  }
  if (action !== 'delete') return
  confirmDeleteConversation(conversation)
}

function openAdmin() {
  if (!canEnterAdmin.value) return
  if (ui.isMobile) {
    ui.mobileNavOpen = false
    router.push('/admin')
    return
  }
  window.open('/admin', '_blank', 'noopener')
}
</script>

<style scoped>
:deep(.chat-sidebar__new.n-button) {
  --n-height: 40px !important;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 650;
  letter-spacing: .015em;
  box-shadow: 0 10px 18px rgba(53, 111, 213, .22);
  transition: transform .18s ease, box-shadow .18s ease;
}

:deep(.chat-sidebar__new.n-button:not(.n-button--disabled):hover) {
  transform: translateY(-1px);
  box-shadow: 0 13px 22px rgba(53, 111, 213, .28);
}

:deep(.chat-sidebar__history-action.n-button) {
  --n-height: 30px !important;
  --n-width: 30px !important;
  --n-icon-size: 16px !important;
  --n-border-radius: 10px !important;
  --n-color-hover: #eef4ff !important;
  --n-color-pressed: #e4efff !important;
  margin-right: 2px;
  opacity: .5;
  transition: opacity .18s ease, color .18s ease, background-color .18s ease;
}

.group:hover :deep(.chat-sidebar__history-action.n-button),
.group:focus-within :deep(.chat-sidebar__history-action.n-button) {
  opacity: 1;
}

:global(.chat-sidebar__history-menu.n-dropdown-menu) {
  --n-border-radius: 14px !important;
  border-radius: 14px !important;
  overflow: hidden;
}

:global(.chat-sidebar__history-menu .chat-sidebar__history-delete-option) {
  color: #dc2626 !important;
}

:global(.chat-sidebar__history-menu .chat-sidebar__history-delete-option .n-dropdown-option-body__prefix) {
  color: #dc2626 !important;
}

:global(.chat-sidebar__history-menu .chat-sidebar__history-delete-option.n-dropdown-option-body--pending::before) {
  background-color: #fef2f2 !important;
}

:global(.dark .chat-sidebar__history-menu .chat-sidebar__history-delete-option),
:global(.dark .chat-sidebar__history-menu .chat-sidebar__history-delete-option .n-dropdown-option-body__prefix) {
  color: #f87171 !important;
}

:global(.dark .chat-sidebar__history-menu .chat-sidebar__history-delete-option.n-dropdown-option-body--pending::before) {
  background-color: rgba(127, 29, 29, .32) !important;
}

.chat-delete-dialog {
  width: min(420px, calc(100vw - 32px));
  box-sizing: border-box;
  border: 1px solid #e3eaf3;
  border-radius: 20px;
  background: linear-gradient(145deg, #ffffff 0%, #fbfcff 100%);
  box-shadow: 0 24px 60px rgba(30, 54, 92, .20), 0 4px 14px rgba(30, 54, 92, .08);
  padding: 22px;
}

.chat-delete-dialog__header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.chat-delete-dialog__icon {
  display: inline-flex;
  width: 42px;
  height: 42px;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  border: 1px solid #ffd9d9;
  border-radius: 14px;
  color: #d84a4a;
  background: #fff2f2;
}

.chat-delete-dialog__eyebrow {
  margin: 0 0 3px;
  color: #c85a5a;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .06em;
}

.chat-delete-dialog h2 {
  margin: 0;
  color: #20304a;
  font-size: 17px;
  font-weight: 720;
  line-height: 1.35;
}

.chat-delete-dialog__description {
  margin: 18px 0 12px;
  color: #61718a;
  font-size: 13px;
  line-height: 1.75;
}

.chat-delete-dialog__description strong {
  color: #354a68;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.chat-delete-dialog__notice {
  display: flex;
  align-items: center;
  gap: 7px;
  border: 1px solid #f6dddd;
  border-radius: 11px;
  color: #b85a5a;
  background: #fff8f8;
  padding: 9px 10px;
  font-size: 12px;
  line-height: 1.45;
}

.chat-delete-dialog__actions {
  display: flex;
  justify-content: flex-end;
  gap: 9px;
  margin-top: 21px;
}

:deep(.chat-delete-dialog__cancel.n-button),
:deep(.chat-delete-dialog__confirm.n-button) {
  --n-height: 38px !important;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 650;
}

:deep(.chat-delete-dialog__cancel.n-button) {
  --n-color: #f6f8fc !important;
  --n-color-hover: #eef3f9 !important;
  --n-color-pressed: #e8eef6 !important;
  --n-border: 1px solid #dfe7f1 !important;
  --n-border-hover: 1px solid #c7d5e6 !important;
  --n-text-color: #53657e !important;
}

:deep(.chat-delete-dialog__confirm.n-button) {
  --n-color: #db5757 !important;
  --n-color-hover: #cc4747 !important;
  --n-color-pressed: #bb3f3f !important;
  --n-border: 1px solid #db5757 !important;
  --n-border-hover: 1px solid #cc4747 !important;
  box-shadow: 0 8px 16px rgba(207, 71, 71, .21);
}

.dark .chat-delete-dialog {
  border-color: #3d4c62;
  background: linear-gradient(145deg, #202c3d 0%, #1c2736 100%);
  box-shadow: 0 24px 60px rgba(0, 0, 0, .38), 0 4px 14px rgba(0, 0, 0, .18);
}

.dark .chat-delete-dialog h2 { color: #e7eef8; }
.dark .chat-delete-dialog__description { color: #aebdd0; }
.dark .chat-delete-dialog__description strong { color: #e1eaf7; }
.dark .chat-delete-dialog__icon { border-color: #71454c; color: #ff9696; background: #41292f; }
.dark .chat-delete-dialog__notice { border-color: #603f47; color: #f3a0a0; background: #35282f; }
</style>
