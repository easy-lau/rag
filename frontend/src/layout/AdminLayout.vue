<template>
  <n-config-provider :theme-overrides="adminThemeOverrides">
  <div class="admin-workspace flex h-screen overflow-hidden">
    <!-- 桌面端后台导航 -->
    <div v-if="!ui.isCompact" class="admin-workspace__sidebar w-64 shrink-0 z-10">
      <AdminSidebar />
    </div>

    <!-- 紧凑屏后台导航：平板和手机统一改为抽屉，避免内容区被侧栏挤压。 -->
    <n-drawer v-else v-model:show="ui.mobileNavOpen" :width="288" placement="left" to="#app">
      <n-drawer-content
        title="管理导航"
        closable
        :native-scrollbar="false"
        :header-style="drawerHeaderStyle"
        :body-content-style="drawerBodyStyle"
      >
        <AdminSidebar />
      </n-drawer-content>
    </n-drawer>

    <div class="admin-workspace__main min-w-0 flex-1 flex flex-col">
      <header class="admin-topbar h-16 shrink-0 flex items-center justify-between px-4 sm:px-6">
        <div class="flex items-center min-w-0 gap-3">
          <n-button v-if="ui.isCompact" quaternary circle size="small" aria-label="打开后台菜单" @click="ui.mobileNavOpen = true">
            <template #icon><n-icon :size="20"><MenuOutline /></n-icon></template>
          </n-button>
          <div class="min-w-0 flex items-center gap-2">
            <span class="admin-topbar__status w-2 h-2 rounded-full" aria-hidden="true"></span>
            <span class="admin-topbar__title text-sm leading-5 font-semibold">{{ pageTitle }}</span>
          </div>
        </div>

        <div class="flex items-center gap-1.5 sm:gap-2">
          <n-button v-if="canReturnToChat" secondary size="small" class="!hidden sm:!inline-flex" @click="backToChat">
            <template #icon><n-icon><ChatbubbleEllipsesOutline /></n-icon></template>
            返回问答
          </n-button>
          <n-button v-if="canReturnToChat" quaternary circle size="small" aria-label="返回问答" class="sm:!hidden" @click="backToChat">
            <template #icon><n-icon :size="18"><ChatbubbleEllipsesOutline /></n-icon></template>
          </n-button>
          <n-button quaternary circle size="small" aria-label="切换主题" @click="ui.toggleTheme()">
            <template #icon>
              <n-icon :size="18"><MoonOutline v-if="ui.mode !== 'dark'" /><SunnyOutline v-else /></n-icon>
            </template>
          </n-button>

          <n-dropdown trigger="click" :options="userOptions" @select="handleUserMenu">
            <button
              type="button"
              class="admin-topbar__user ml-1 flex items-center gap-2 px-1.5 sm:px-2 py-1.5 text-left transition-colors"
            >
              <span class="admin-topbar__avatar w-7 h-7 flex items-center justify-center text-xs font-semibold">
                {{ userInitial }}
              </span>
              <span class="admin-topbar__user-name hidden sm:block max-w-32 text-sm truncate">
                {{ userName }}
              </span>
              <n-icon :size="14" class="admin-topbar__chevron hidden sm:block"><ChevronDownOutline /></n-icon>
            </button>
          </n-dropdown>
        </div>
      </header>

      <main class="min-h-0 flex-1 overflow-hidden">
        <router-view />
      </main>

      <footer
        v-if="siteStore.site_copyright"
        class="admin-workspace__footer shrink-0 px-4 py-2 text-center text-xs"
      >
        {{ siteStore.site_copyright }}
      </footer>
    </div>

    <AppModal
      v-model:show="showPasswordModal"
      title="修改密码"
      width="min(90vw, 400px)"
      :loading="savingPassword"
      @close="closePasswordModal"
    >
      <n-form ref="passwordFormRef" :model="passwordForm" :rules="passwordRules" label-placement="top">
        <n-form-item label="当前密码" path="old_password">
          <n-input v-model:value="passwordForm.old_password" type="password" show-password-on="click" placeholder="请输入当前密码" />
        </n-form-item>
        <n-form-item label="新密码" path="new_password">
          <n-input v-model:value="passwordForm.new_password" type="password" show-password-on="click" placeholder="至少 6 位" />
        </n-form-item>
        <n-form-item label="确认新密码" path="confirm">
          <n-input v-model:value="passwordForm.confirm" type="password" show-password-on="click" placeholder="再次输入新密码" />
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="savingPassword" @click="closePasswordModal">取消</n-button>
          <n-button type="primary" :loading="savingPassword" @click="changeCurrentPassword">确定</n-button>
        </div>
      </template>
    </AppModal>
  </div>
  </n-config-provider>
</template>

<script setup>
import { computed, h, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NButton,
  NConfigProvider,
  NDrawer,
  NDrawerContent,
  NDropdown,
  NForm,
  NFormItem,
  NIcon,
  NInput,
  useMessage,
} from 'naive-ui'
import {
  ChatbubbleEllipsesOutline,
  ChevronDownOutline,
  KeyOutline,
  LogOutOutline,
  MenuOutline,
  MoonOutline,
  SunnyOutline,
} from '@vicons/ionicons5'
import AdminSidebar from './AdminSidebar.vue'
import AppModal from '@/components/ui/AppModal.vue'
import { changePassword } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { useUiStore } from '@/stores/ui'
import { createAdminNaiveThemeOverrides } from '@/theme/naive'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const siteStore = useSiteStore()
const ui = useUiStore()
const message = useMessage()
const adminThemeOverrides = computed(() => createAdminNaiveThemeOverrides(ui.mode))
const pageTitle = computed(() => route.meta?.title || '管理后台')

const userName = computed(() => authStore.user?.display_name || authStore.user?.username || '用户')
const userInitial = computed(() => userName.value.slice(0, 1))
const canReturnToChat = computed(() => authStore.hasPerm('menu:chat'))
const drawerHeaderStyle = { padding: '16px 18px', borderBottom: '1px solid var(--admin-ui-border)' }
const drawerBodyStyle = { padding: 0 }

// 两套布局共用移动端抽屉状态；切换路由时主动收起，避免抽屉跨布局残留。
watch(() => route.fullPath, () => {
  ui.mobileNavOpen = false
})

const renderIcon = icon => () => h(NIcon, null, { default: () => h(icon) })
const userOptions = [
  { label: '修改密码', key: 'change-password', icon: renderIcon(KeyOutline) },
  { type: 'divider', key: 'divider' },
  { label: '退出登录', key: 'logout', icon: renderIcon(LogOutOutline) },
]

const showPasswordModal = ref(false)
const savingPassword = ref(false)
const passwordFormRef = ref(null)
const passwordForm = ref({ old_password: '', new_password: '', confirm: '' })
const passwordRules = {
  old_password: { required: true, message: '请输入当前密码', trigger: 'blur' },
  new_password: { required: true, min: 6, message: '新密码至少 6 位', trigger: 'blur' },
  confirm: {
    required: true,
    trigger: 'blur',
    validator: (_rule, value) => value === passwordForm.value.new_password || new Error('两次输入不一致'),
  },
}

function backToChat() {
  ui.mobileNavOpen = false
  router.push('/chat')
}

function handleUserMenu(key) {
  if (key === 'logout') {
    authStore.logout()
    return
  }
  if (key === 'change-password') {
    passwordForm.value = { old_password: '', new_password: '', confirm: '' }
    showPasswordModal.value = true
  }
}

function closePasswordModal() {
  if (savingPassword.value) return
  showPasswordModal.value = false
}

async function changeCurrentPassword() {
  try {
    await passwordFormRef.value?.validate()
  } catch {
    return
  }
  savingPassword.value = true
  try {
    await changePassword({
      old_password: passwordForm.value.old_password,
      new_password: passwordForm.value.new_password,
    })
    message.success('密码已修改')
    showPasswordModal.value = false
  } catch (error) {
    message.error(error?.response?.data?.detail || '修改失败，请检查当前密码')
  } finally {
    savingPassword.value = false
  }
}
</script>

<style scoped>
.admin-workspace {
  --ui-bg: var(--admin-ui-bg);
  --ui-bg-subtle: var(--admin-ui-bg-subtle);
  --ui-surface: var(--admin-ui-surface);
  --ui-surface-raised: var(--admin-ui-surface-raised);
  --ui-surface-muted: var(--admin-ui-surface-muted);
  --ui-surface-hover: var(--admin-ui-surface-hover);
  --ui-surface-pressed: var(--admin-ui-surface-pressed);
  --ui-surface-disabled: var(--admin-ui-surface-disabled);
  --ui-text: var(--admin-ui-text);
  --ui-text-secondary: var(--admin-ui-text-secondary);
  --ui-text-tertiary: var(--admin-ui-text-tertiary);
  --ui-text-disabled: var(--admin-ui-text-disabled);
  --ui-placeholder: var(--admin-ui-placeholder);
  --ui-icon: var(--admin-ui-icon);
  --ui-border: var(--admin-ui-border);
  --ui-border-strong: var(--admin-ui-border-strong);
  --ui-border-focus: var(--admin-ui-border-focus);
  --ui-divider: var(--admin-ui-divider);
  --ui-primary: var(--admin-ui-primary);
  --ui-primary-hover: var(--admin-ui-primary-hover);
  --ui-primary-pressed: var(--admin-ui-primary-pressed);
  --ui-primary-subtle: var(--admin-ui-primary-subtle);
  --ui-text-on-primary: var(--admin-ui-text-on-primary);
  --ui-info: var(--admin-ui-info);
  --ui-accent-violet: var(--admin-ui-accent-violet);
  --ui-success: var(--admin-ui-success);
  --ui-warning: var(--admin-ui-warning);
  --ui-danger: var(--admin-ui-danger);
  --ui-danger-subtle: var(--admin-ui-danger-subtle);
  --ui-focus-outline: var(--admin-ui-focus-outline);
  --ui-focus-ring: var(--admin-ui-focus-ring);
  --ui-overlay-mask: var(--admin-ui-overlay-mask);
  --ui-shadow-card: var(--admin-ui-shadow-card);
  --ui-shadow-float: var(--admin-ui-shadow-float);
  --ui-shadow-dialog: var(--admin-ui-shadow-dialog);
  --ui-radius-control: var(--admin-ui-radius-control);
  --ui-radius-popover: var(--admin-ui-radius-popover);
  --ui-radius-card: var(--admin-ui-radius-card);
  --ui-radius-dialog: var(--admin-ui-radius-dialog);
  --ui-radius-pill: var(--admin-ui-radius-pill);
  background: var(--ui-bg);
  color: var(--ui-text);
}

.admin-workspace__sidebar {
  border-right: 1px solid var(--admin-sidebar-divider);
  box-shadow: 4px 0 18px rgba(18, 25, 51, 0.035);
}

.admin-topbar {
  border-bottom: 1px solid var(--ui-border);
  background: color-mix(in srgb, var(--ui-surface) 96%, transparent);
}

.admin-topbar__status {
  background: var(--ui-primary);
  box-shadow: 0 0 0 4px var(--ui-primary-subtle);
}

.admin-topbar__title { color: var(--ui-text); }

.admin-topbar__user {
  border: 1px solid transparent;
  border-radius: var(--ui-radius-control);
  background: transparent;
  color: var(--ui-text-secondary);
}

.admin-topbar__user:hover {
  border-color: var(--ui-border);
  background: var(--ui-surface-hover);
  color: var(--ui-text);
}

.admin-topbar__avatar {
  border-radius: 8px;
  background: var(--ui-primary);
  color: var(--ui-text-on-primary);
  box-shadow: 0 3px 8px rgba(0, 75, 255, 0.18);
}

.admin-topbar__user-name { color: inherit; }
.admin-topbar__chevron { color: var(--ui-icon); }

.admin-workspace__footer {
  border-top: 1px solid var(--ui-border);
  background: var(--ui-surface);
  color: var(--ui-text-tertiary);
}
</style>
