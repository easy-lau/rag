<template>
  <header class="app-header">
    <div class="app-header__leading">
      <n-button
        v-if="ui.isCompact"
        text
        size="small"
        class="app-header__icon-button"
        title="打开会话菜单"
        aria-label="打开会话菜单"
        :aria-expanded="ui.mobileNavOpen"
        @click="ui.mobileNavOpen = true"
      >
        <template #icon><n-icon :size="20"><MenuOutline /></n-icon></template>
      </n-button>
      <div
        v-if="isChatRoute"
        class="app-header__chat-context"
        :title="conversationTitle"
        :aria-label="`当前会话：${conversationTitle}`"
      >
        <span class="app-header__chat-context-dot" aria-hidden="true"></span>
        <span class="truncate">{{ conversationTitle }}</span>
      </div>
      <div v-else class="text-sm font-medium text-gray-700 dark:text-gray-200 truncate">
        {{ currentPageTitle }}
      </div>
    </div>
    <div class="app-header__actions">
      <n-button
        v-if="isChatRoute"
        class="app-header__chat-search"
        :class="{ 'is-active': ui.chatSearchOpen }"
        size="small"
        :aria-label="ui.chatSearchOpen ? '收起检索结果' : '展开检索结果'"
        :aria-expanded="ui.chatSearchOpen"
        :title="ui.chatSearchOpen ? '收起检索结果' : '展开检索结果'"
        @click="ui.toggleChatSearch()"
      >
        <template #icon><n-icon><SearchOutline /></n-icon></template>
        <span class="hidden sm:inline">{{ ui.chatSearchOpen ? '收起检索' : '检索结果' }}</span>
      </n-button>

      <n-button class="app-header__icon-button" text size="small" title="切换深浅色主题" aria-label="切换深浅色主题" @click="ui.toggleTheme()">
        <template #icon>
          <n-icon><MoonOutline v-if="ui.mode !== 'dark'" /><SunnyOutline v-else /></n-icon>
        </template>
      </n-button>

      <!-- 用户菜单：点击展开「修改密码 / 退出登录」 -->
      <n-dropdown
        trigger="click"
        :options="userOptions"
        @select="handleSelect"
      >
        <button
          type="button"
          class="app-header__user"
          aria-label="打开用户菜单"
        >
          <div class="app-header__avatar">
            {{ (authStore.user?.display_name || authStore.user?.username || '用')[0] }}
          </div>
          <span class="app-header__user-name">{{ authStore.user?.display_name || authStore.user?.username || '用户' }}</span>
          <n-icon :size="14" class="app-header__user-chevron"><ChevronDownOutline /></n-icon>
        </button>
      </n-dropdown>
    </div>

    <!-- 修改密码弹窗 -->
    <AppModal
      v-model:show="showPwd"
      title="修改密码"
      width="min(90vw, 400px)"
      :loading="saving"
      @close="closePasswordModal"
    >
      <n-form ref="pwdFormRef" :model="pwdForm" :rules="pwdRules" label-placement="top">
        <n-form-item label="当前密码" path="old_password">
          <n-input v-model:value="pwdForm.old_password" type="password" show-password-on="click" placeholder="请输入当前密码" />
        </n-form-item>
        <n-form-item label="新密码" path="new_password">
          <n-input v-model:value="pwdForm.new_password" type="password" show-password-on="click" placeholder="至少 6 位" />
        </n-form-item>
        <n-form-item label="确认新密码" path="confirm">
          <n-input v-model:value="pwdForm.confirm" type="password" show-password-on="click" placeholder="再次输入新密码" />
        </n-form-item>
      </n-form>
      <template #footer>
        <n-button :disabled="saving" @click="closePasswordModal">取消</n-button>
        <n-button type="primary" :loading="saving" @click="handleChangePassword">确定</n-button>
      </template>
    </AppModal>
  </header>
</template>

<script setup>
import { computed, ref, h } from 'vue'
import { useRoute } from 'vue-router'
import { NButton, NIcon, NDropdown, NForm, NFormItem, NInput, useMessage } from 'naive-ui'
import { MoonOutline, SunnyOutline, ChevronDownOutline, KeyOutline, LogOutOutline, MenuOutline, SearchOutline } from '@vicons/ionicons5'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { changePassword } from '@/api/auth'
import AppModal from '@/components/ui/AppModal.vue'

const ui = useUiStore()
const authStore = useAuthStore()
const chatStore = useChatStore()
const route = useRoute()
const msg = useMessage()

const titles = {
  chat: '问答对话', knowledge: '知识库管理', documents: '文档管理',
  'search-test': '检索测试', settings: '系统设置', 'model-management': '模型管理',
  users: '用户管理', roles: '角色管理', 'audit-logs': '审计日志'
}
const currentPageTitle = computed(() => titles[route.name] || 'RAG 检索系统')
const isChatRoute = computed(() => route.name === 'chat')
const conversationTitle = computed(() => {
  const current = chatStore.conversations.find(item => item.id === chatStore.currentConvId)
  return current?.title || (chatStore.currentConvId ? '当前对话' : '新对话')
})

const renderIcon = (icon) => () => h(NIcon, null, { default: () => h(icon) })
const userOptions = [
  { label: '修改密码', key: 'change-password', icon: renderIcon(KeyOutline) },
  { type: 'divider', key: 'd1' },
  { label: '退出登录', key: 'logout', icon: renderIcon(LogOutOutline) },
]
const showPwd = ref(false)
const saving = ref(false)
const pwdFormRef = ref(null)
const pwdForm = ref({ old_password: '', new_password: '', confirm: '' })
const pwdRules = {
  old_password: { required: true, message: '请输入当前密码', trigger: 'blur' },
  new_password: { required: true, min: 6, message: '新密码至少 6 位', trigger: 'blur' },
  confirm: {
    required: true, trigger: 'blur',
    validator: (_r, v) => v === pwdForm.value.new_password || new Error('两次输入不一致'),
  },
}

function handleSelect(key) {
  if (key === 'logout') {
    authStore.logout()
  } else if (key === 'change-password') {
    pwdForm.value = { old_password: '', new_password: '', confirm: '' }
    showPwd.value = true
  }
}

function closePasswordModal() {
  if (saving.value) return
  showPwd.value = false
}

async function handleChangePassword() {
  try {
    await pwdFormRef.value?.validate()
  } catch {
    return
  }
  saving.value = true
  try {
    await changePassword({ old_password: pwdForm.value.old_password, new_password: pwdForm.value.new_password })
    msg.success('密码已修改')
    showPwd.value = false
  } catch (e) {
    msg.error(e?.response?.data?.detail || '修改失败，请检查当前密码')
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.app-header {
  display: flex;
  min-height: 64px;
  flex: 0 0 64px;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  border-bottom: 1px solid var(--ui-divider);
  background: color-mix(in srgb, var(--ui-surface) 92%, transparent);
  padding: 0 22px;
}

.app-header__leading,
.app-header__actions {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 9px;
}

.app-header__actions { flex: 0 0 auto; }

.app-header__chat-context {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 10px;
  color: var(--ui-text);
  font-size: 14px;
  font-weight: 680;
  line-height: 1;
}

.app-header__chat-context-dot {
  width: 8px;
  height: 8px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--ui-primary);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--ui-primary) 14%, transparent);
}

:deep(.app-header__chat-search.n-button) {
  height: var(--ui-control-height);
  border-radius: var(--ui-radius-control);
  font-size: 12px;
  font-weight: 650;
  --n-color: var(--ui-surface-muted) !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-color-pressed: var(--ui-surface-pressed) !important;
  --n-border: 1px solid var(--ui-border) !important;
  --n-border-hover: 1px solid var(--ui-border-strong) !important;
  --n-border-pressed: 1px solid var(--ui-border-focus) !important;
  --n-text-color: var(--ui-text-secondary) !important;
  --n-text-color-hover: var(--ui-primary) !important;
  --n-text-color-pressed: var(--ui-primary-pressed) !important;
}

:deep(.app-header__icon-button.n-button) {
  --n-width: var(--ui-control-height) !important;
  --n-height: var(--ui-control-height) !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-color-pressed: var(--ui-surface-pressed) !important;
  --n-icon-color: var(--ui-icon) !important;
  --n-icon-color-hover: var(--ui-primary) !important;
  border-radius: var(--ui-radius-control);
}

.app-header__user {
  display: flex;
  height: 40px;
  align-items: center;
  gap: 8px;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-popover);
  background: transparent;
  padding: 3px 7px 3px 4px;
  color: var(--ui-text-secondary);
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  transition: border-color .18s ease, background .18s ease, color .18s ease;
}

.app-header__user:hover {
  border-color: var(--ui-border);
  background: var(--ui-surface-hover);
  color: var(--ui-text);
}

.app-header__avatar {
  display: grid;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  place-items: center;
  border-radius: 10px;
  background: linear-gradient(180deg, var(--ui-primary) 0%, var(--ui-primary-hover) 100%);
  color: var(--ui-text-on-primary);
  font-size: 12px;
  font-weight: 700;
}

.app-header__user-name {
  max-width: 128px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.app-header__user-chevron { color: var(--ui-icon); }

:deep(.app-header__chat-search.is-active.n-button) {
  --n-color: var(--ui-primary-subtle) !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-border: 1px solid var(--ui-border-focus) !important;
  --n-text-color: var(--ui-primary) !important;
}

@media (max-width: 639px) {
  .app-header { min-height: 56px; flex-basis: 56px; padding: 0 12px; }
  .app-header__chat-context { max-width: 38vw; font-size: 12px; }
  :deep(.app-header__chat-search.n-button) { min-width: var(--ui-control-height-compact); padding: 0 8px; }
  .app-header__user { width: 40px; padding: 3px; }
  .app-header__user-name,
  .app-header__user-chevron { display: none; }
}
</style>
