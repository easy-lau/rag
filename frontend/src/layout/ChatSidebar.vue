<template>
  <aside class="chat-sidebar">
    <!-- 品牌 -->
    <div class="chat-sidebar__brand-bar">
      <div class="chat-sidebar__brand-row">
        <div class="chat-sidebar__brand">
          <img
            v-if="siteStore.site_logo"
            :src="siteStore.site_logo"
            class="chat-sidebar__logo"
            alt="logo"
          />
          <div v-else class="chat-sidebar__logo chat-sidebar__logo--fallback">
            {{ (siteStore.site_title || 'R')[0] }}
          </div>
          <div class="chat-sidebar__brand-copy">
            <div class="chat-sidebar__brand-title">{{ siteStore.site_title }}</div>
            <div class="chat-sidebar__brand-caption">KNOWLEDGE CENTER</div>
          </div>
        </div>
        <n-button
          v-if="inDrawer"
          quaternary
          circle
          size="small"
          class="chat-sidebar__drawer-close shrink-0"
          aria-label="关闭会话菜单"
          title="关闭菜单"
          @click="closeDrawer"
        >
          <template #icon><n-icon :size="18"><CloseOutline /></n-icon></template>
        </n-button>
      </div>
    </div>

    <!-- 会话功能：问答工作台不渲染其他业务菜单。 -->
    <section class="chat-sidebar__body">
      <n-button class="chat-sidebar__new" type="primary" block :disabled="chatStore.isStreaming" @click="startNewConversation">
        <template #icon><n-icon><AddOutline /></n-icon></template>
        新对话
      </n-button>

      <div class="chat-sidebar__section-heading">
        <span>最近对话</span>
        <div v-if="chatStore.conversations.length" class="chat-sidebar__section-actions">
          <span class="chat-sidebar__history-count">{{ chatStore.conversations.length }}</span>
          <n-button
            v-if="!isSelectionMode"
            quaternary
            size="tiny"
            class="chat-sidebar__manage"
            :disabled="chatStore.isStreaming"
            aria-label="批量管理对话历史"
            @click="enterSelectionMode"
          >
            管理
          </n-button>
        </div>
      </div>

      <div v-if="isSelectionMode" class="chat-sidebar__batch-toolbar" aria-label="批量管理对话">
        <n-checkbox
          :checked="allConversationsSelected"
          :indeterminate="someConversationsSelected"
          :disabled="chatStore.isStreaming || isBatchDeleting"
          aria-label="全选对话"
          @update:checked="toggleAllConversations"
        >
          全选
        </n-checkbox>
        <span class="chat-sidebar__batch-count">已选 {{ selectedConversationCount }} 项</span>
        <n-button
          quaternary
          size="tiny"
          :disabled="isBatchDeleting"
          @click="exitSelectionMode"
        >
          取消
        </n-button>
        <n-button
          type="error"
          secondary
          size="tiny"
          :disabled="!selectedConversationCount || chatStore.isStreaming"
          @click="confirmBatchDelete"
        >
          删除
        </n-button>
      </div>

      <div class="chat-sidebar__history">
        <div v-if="!chatStore.conversations.length" class="chat-sidebar__empty">暂无历史对话</div>
        <div
          v-for="conv in chatStore.conversations"
          :key="conv.id"
          class="group chat-sidebar__history-row"
          :class="[
            conv.id === chatStore.currentConvId
              ? 'chat-sidebar__history-row--active'
            : '',
            { 'chat-sidebar__history-row--selected': isConversationSelected(conv.id) },
            { 'cursor-not-allowed opacity-50': chatStore.isStreaming },
          ]"
        >
          <n-checkbox
            v-if="isSelectionMode"
            class="chat-sidebar__history-checkbox"
            :checked="isConversationSelected(conv.id)"
            :disabled="chatStore.isStreaming || isBatchDeleting"
            :aria-label="`选择对话：${conv.title || '未命名对话'}`"
            @update:checked="checked => toggleConversationSelection(conv.id, checked)"
          />
          <button
            type="button"
            class="chat-sidebar__history-button"
            :disabled="chatStore.isStreaming"
            :aria-current="conv.id === chatStore.currentConvId ? 'page' : undefined"
            @click="handleConversationClick(conv)"
          >
            <span>{{ conv.title || '未命名对话' }}</span>
          </button>
          <n-dropdown
            v-if="!isSelectionMode"
            trigger="click"
            placement="bottom-end"
            :options="conversationActionOptions"
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

    <!-- 侧栏底部：唯一的后台入口与构建时注入的版本信息。 -->
    <div class="chat-sidebar__footer">
      <button
        v-if="canEnterAdmin"
        type="button"
        class="group chat-sidebar__admin"
        :title="ui.isCompact ? '进入管理后台' : '在新标签打开管理后台'"
        @click="openAdmin"
      >
        <span class="chat-sidebar__admin-icon">
          <n-icon :size="16"><SettingsOutline /></n-icon>
        </span>
        <span class="chat-sidebar__admin-copy">
          <span>管理后台</span>
          <small>知识运营与系统管理</small>
        </span>
        <n-icon :size="15" class="chat-sidebar__admin-arrow"><ChevronForwardOutline /></n-icon>
      </button>
      <AppVersion :class="{ 'mt-2': canEnterAdmin }" />
    </div>
  </aside>

  <AppModal
    v-model:show="showRenameModal"
    title="重命名会话"
    width="min(90vw, 420px)"
    :loading="isRenaming"
    @close="closeRenameModal"
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
      <n-button :disabled="isRenaming" @click="closeRenameModal">取消</n-button>
      <n-button type="primary" :loading="isRenaming" @click="submitRename">保存</n-button>
    </template>
  </AppModal>

  <DangerConfirm
    v-model:show="showDeleteModal"
    :loading="isDeleting"
    title="删除这段对话？"
    :subject="pendingDeleteTitle"
    description="其中的全部问答内容也会被永久删除，且无法恢复。"
    confirm-text="永久删除"
    @confirm="submitDelete"
    @cancel="clearPendingDelete"
  />

  <DangerConfirm
    v-model:show="showBatchDeleteModal"
    :loading="isBatchDeleting"
    title="批量删除对话？"
    :subject="`已选择 ${selectedConversationCount} 段对话`"
    description="这些对话中的全部问答内容都会被永久删除，且无法恢复。"
    :confirm-text="`永久删除 ${selectedConversationCount} 段`"
    @confirm="submitBatchDelete"
  />
</template>

<script setup>
import { computed, h, nextTick, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NButton, NCheckbox, NDropdown, NIcon, NInput, useMessage } from 'naive-ui'
import { AddOutline, ChevronForwardOutline, CloseOutline, EllipsisHorizontalOutline, PencilOutline, SettingsOutline, TrashOutline } from '@vicons/ionicons5'
import { hasAdminAccess } from '@/router/menus'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { useSiteStore } from '@/stores/site'
import { useUiStore } from '@/stores/ui'
import AppVersion from '@/components/ui/AppVersion.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'

const props = defineProps({
  inDrawer: { type: Boolean, default: false },
})
const emit = defineEmits(['close-drawer'])

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
const isSelectionMode = ref(false)
const selectedConversationIds = ref([])
const showBatchDeleteModal = ref(false)
const isBatchDeleting = ref(false)
const selectedConversationIdSet = computed(() => new Set(selectedConversationIds.value))
const selectedConversationCount = computed(() => selectedConversationIds.value.length)
const allConversationsSelected = computed(() => (
  chatStore.conversations.length > 0
  && selectedConversationCount.value === chatStore.conversations.length
))
const someConversationsSelected = computed(() => (
  selectedConversationCount.value > 0 && !allConversationsSelected.value
))

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
onMounted(() => {
  if (authStore.hasPerm('menu:chat')) {
    chatStore.loadHistory().catch(() => message.error('加载对话历史失败，请刷新重试'))
  }
})

function startNewConversation() {
  if (chatStore.isStreaming) return
  exitSelectionMode()
  ui.closeChatSearch()
  closeDrawer()

  const query = { ...route.query }
  if (!Object.prototype.hasOwnProperty.call(query, 'conversation')) {
    chatStore.newConversation()
    return
  }
  delete query.conversation
  router.push({ name: 'chat', query }).catch(() => {})
}

function enterSelectionMode() {
  if (chatStore.isStreaming || !chatStore.conversations.length) return
  selectedConversationIds.value = []
  isSelectionMode.value = true
}

function exitSelectionMode() {
  if (isBatchDeleting.value) return
  showBatchDeleteModal.value = false
  selectedConversationIds.value = []
  isSelectionMode.value = false
}

function isConversationSelected(conversationId) {
  return selectedConversationIdSet.value.has(String(conversationId))
}

function toggleConversationSelection(conversationId, checked) {
  if (!isSelectionMode.value || chatStore.isStreaming || isBatchDeleting.value) return
  const id = String(conversationId)
  const next = new Set(selectedConversationIds.value)
  if (checked) next.add(id)
  else next.delete(id)
  selectedConversationIds.value = [...next]
}

function toggleAllConversations(checked) {
  if (chatStore.isStreaming || isBatchDeleting.value) return
  selectedConversationIds.value = checked
    ? chatStore.conversations.map(conversation => String(conversation.id))
    : []
}

function handleConversationClick(conversation) {
  if (isSelectionMode.value) {
    toggleConversationSelection(
      conversation.id,
      !isConversationSelected(conversation.id),
    )
    return
  }
  selectConversation(conversation.id)
}

function confirmBatchDelete() {
  if (!selectedConversationCount.value || chatStore.isStreaming) return
  showBatchDeleteModal.value = true
}

async function submitBatchDelete() {
  if (chatStore.isStreaming) {
    message.warning('生成回答时暂不能删除会话')
    return
  }
  const existingIds = new Set(chatStore.conversations.map(item => String(item.id)))
  const ids = selectedConversationIds.value.filter(id => existingIds.has(id))
  if (!ids.length) {
    message.warning('请选择需要删除的对话')
    showBatchDeleteModal.value = false
    return
  }

  isBatchDeleting.value = true
  try {
    const result = await chatStore.removeConversations(ids)
    showBatchDeleteModal.value = false
    selectedConversationIds.value = []
    isSelectionMode.value = false
    message.success(`已删除 ${result?.deleted_count || ids.length} 段对话`)
  } catch (error) {
    message.error(error?.response?.data?.detail || '批量删除失败，请稍后重试')
  } finally {
    isBatchDeleting.value = false
  }
}

function selectConversation(conversationId) {
  if (chatStore.isStreaming) return
  ui.closeChatSearch()
  closeDrawer()
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

function closeRenameModal() {
  if (isRenaming.value) return
  showRenameModal.value = false
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

function clearPendingDelete() {
  if (!isDeleting.value) pendingDeleteConversation.value = null
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
    pendingDeleteConversation.value = null
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
  if (ui.isCompact) {
    closeDrawer()
    router.push('/admin')
    return
  }
  window.open('/admin', '_blank', 'noopener')
}

function closeDrawer() {
  if (!props.inDrawer) return
  emit('close-drawer')
}
</script>

<style scoped>
.chat-sidebar {
  display: flex;
  width: 100%;
  height: 100%;
  flex-direction: column;
  border-right: 1px solid var(--ui-divider);
  background: color-mix(in srgb, var(--ui-surface) 96%, var(--ui-primary-subtle));
}

.chat-sidebar__brand-bar {
  flex: 0 0 auto;
  border-bottom: 1px solid var(--ui-divider);
  padding: 18px 16px;
}

.chat-sidebar__brand-row,
.chat-sidebar__brand {
  display: flex;
  min-width: 0;
  align-items: center;
}

.chat-sidebar__brand-row { justify-content: space-between; gap: 8px; }
.chat-sidebar__brand { gap: 11px; }

.chat-sidebar__logo {
  width: 38px;
  height: 38px;
  flex: 0 0 38px;
  overflow: hidden;
  border-radius: 11px;
  object-fit: cover;
  box-shadow: 0 8px 20px color-mix(in srgb, var(--ui-primary) 18%, transparent);
}

.chat-sidebar__logo--fallback {
  display: grid;
  place-items: center;
  background: linear-gradient(180deg, var(--ui-primary) 0%, var(--ui-primary-hover) 100%);
  color: var(--ui-text-on-primary);
  font-size: 15px;
  font-weight: 750;
}

.chat-sidebar__brand-copy { min-width: 0; }

.chat-sidebar__brand-title {
  overflow: hidden;
  color: var(--ui-text);
  font-size: 14px;
  font-weight: 720;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-sidebar__brand-caption {
  overflow: hidden;
  margin-top: 4px;
  color: var(--ui-text-tertiary);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: .12em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-sidebar__body {
  display: flex;
  min-height: 0;
  flex: 1 1 auto;
  flex-direction: column;
  padding: 16px 12px 12px;
}

.chat-sidebar__section-heading {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: space-between;
  margin: 22px 4px 9px;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  font-weight: 680;
  letter-spacing: .06em;
}

.chat-sidebar__section-actions {
  display: flex;
  align-items: center;
  gap: 5px;
}

.chat-sidebar__history-count {
  display: grid;
  min-width: 22px;
  height: 22px;
  place-items: center;
  border-radius: var(--ui-radius-pill);
  background: var(--ui-surface-muted);
  color: var(--ui-text-tertiary);
  font-size: 10px;
}

.chat-sidebar__history {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
  padding-right: 2px;
  scrollbar-width: thin;
}

.chat-sidebar__empty {
  padding: 34px 10px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  text-align: center;
}

.chat-sidebar__history-row {
  display: flex;
  min-height: 42px;
  align-items: center;
  gap: 2px;
  margin-bottom: 3px;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-popover);
  color: var(--ui-text-secondary);
  font-size: 13px;
  transition: border-color .18s ease, background .18s ease, color .18s ease;
}

.chat-sidebar__history-row:not(.chat-sidebar__history-row--active):hover {
  background: var(--ui-surface-hover);
  color: var(--ui-text);
}

.chat-sidebar__history-row--active {
  border-color: color-mix(in srgb, var(--ui-primary) 20%, var(--ui-border));
  background: var(--ui-primary-subtle);
  color: var(--ui-primary);
}

.chat-sidebar__history-button {
  min-width: 0;
  flex: 1 1 auto;
  border: 0;
  border-radius: var(--ui-radius-control);
  background: transparent;
  padding: 10px 11px;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.chat-sidebar__history-button span {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-sidebar__history-button:disabled { cursor: not-allowed; }

.chat-sidebar__footer {
  flex: 0 0 auto;
  border-top: 1px solid var(--ui-divider);
  padding: 12px;
}

.chat-sidebar__admin {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 10px;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-popover);
  background: transparent;
  padding: 9px;
  color: var(--ui-text-secondary);
  cursor: pointer;
  font: inherit;
  text-align: left;
  transition: border-color .18s ease, background .18s ease, color .18s ease;
}

.chat-sidebar__admin:hover {
  border-color: var(--ui-border);
  background: var(--ui-surface-hover);
  color: var(--ui-primary);
}

.chat-sidebar__admin-icon {
  display: grid;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  place-items: center;
  border-radius: 10px;
  background: var(--ui-surface-muted);
  color: var(--ui-icon);
}

.chat-sidebar__admin:hover .chat-sidebar__admin-icon {
  background: var(--ui-primary-subtle);
  color: var(--ui-primary);
}

.chat-sidebar__admin-copy { min-width: 0; flex: 1 1 auto; }
.chat-sidebar__admin-copy > span { display: block; font-size: 13px; font-weight: 650; }
.chat-sidebar__admin-copy small { display: block; overflow: hidden; margin-top: 2px; color: var(--ui-text-tertiary); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.chat-sidebar__admin-arrow { color: var(--ui-icon); }

:deep(.chat-sidebar__new.n-button) {
  --n-height: 42px !important;
  border-radius: var(--ui-radius-control);
  font-size: 13px;
  font-weight: 650;
  letter-spacing: .015em;
  box-shadow: 0 6px 16px color-mix(in srgb, var(--ui-primary) 18%, transparent);
  transition: transform .18s ease, box-shadow .18s ease;
}

:deep(.chat-sidebar__new.n-button:not(.n-button--disabled):hover) {
  transform: translateY(-1px);
  box-shadow: var(--ui-shadow-float);
}

:deep(.chat-sidebar__manage.n-button) {
  --n-height: 32px !important;
  --n-border-radius: var(--ui-radius-control) !important;
}

.chat-sidebar__batch-toolbar {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 5px;
  min-height: 40px;
  margin-bottom: 8px;
  padding: 5px 6px 5px 9px;
  background: var(--ui-surface-muted);
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-popover);
}

.chat-sidebar__batch-count {
  min-width: 0;
  margin-right: auto;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.chat-sidebar__batch-toolbar :deep(.n-button) {
  --n-height: 32px !important;
  --n-border-radius: var(--ui-radius-control) !important;
  padding-inline: 7px;
}

.chat-sidebar__batch-toolbar :deep(.n-checkbox__label) {
  padding-left: 5px;
  font-size: 12px;
}

.chat-sidebar__history-row--selected {
  box-shadow: inset 0 0 0 1px var(--ui-border-focus);
}

.chat-sidebar__history-checkbox {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  min-width: 32px;
  min-height: 32px;
  padding-left: 5px;
}

:deep(.chat-sidebar__history-action.n-button) {
  --n-height: 30px !important;
  --n-width: 30px !important;
  --n-icon-size: 16px !important;
  --n-border-radius: 10px !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-color-pressed: var(--ui-surface-pressed) !important;
  margin-right: 2px;
  opacity: .5;
  transition: opacity .18s ease, color .18s ease, background-color .18s ease;
}

:deep(.chat-sidebar__drawer-close.n-button) {
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-color-pressed: var(--ui-surface-pressed) !important;
  --n-icon-color: var(--ui-icon) !important;
  --n-icon-color-hover: var(--ui-primary) !important;
  border-radius: 10px;
}

.group:hover :deep(.chat-sidebar__history-action.n-button),
.group:focus-within :deep(.chat-sidebar__history-action.n-button) {
  opacity: 1;
}

:global(.chat-sidebar__history-delete-option) {
  color: var(--ui-danger) !important;
}

:global(.chat-sidebar__history-delete-option .n-dropdown-option-body__prefix) {
  color: var(--ui-danger) !important;
}

:global(.chat-sidebar__history-delete-option.n-dropdown-option-body--pending::before) {
  background-color: var(--ui-danger-subtle) !important;
}

@media (max-width: 639px) {
  :deep(.chat-sidebar__manage.n-button),
  .chat-sidebar__batch-toolbar :deep(.n-button) {
    --n-height: 40px !important;
  }

  .chat-sidebar__history-checkbox {
    min-width: 40px;
    min-height: 40px;
  }
}

</style>
