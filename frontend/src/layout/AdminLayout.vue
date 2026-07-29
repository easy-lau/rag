<template>
  <div class="flex h-screen overflow-hidden bg-slate-100 dark:bg-slate-950">
    <!-- 桌面端后台导航 -->
    <div v-if="!ui.isCompact" class="w-64 shrink-0 shadow-xl shadow-slate-900/10 z-10">
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

    <div class="min-w-0 flex-1 flex flex-col">
      <header class="h-16 shrink-0 flex items-center justify-between px-4 sm:px-6 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800">
        <div class="flex items-center min-w-0 gap-3">
          <n-button v-if="ui.isCompact" quaternary circle size="small" aria-label="打开后台菜单" @click="ui.mobileNavOpen = true">
            <template #icon><n-icon :size="20"><MenuOutline /></n-icon></template>
          </n-button>
          <div class="min-w-0 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-blue-500 shadow-sm shadow-blue-500/40" aria-hidden="true"></span>
            <span class="text-sm leading-5 font-semibold text-slate-700 dark:text-slate-200">管理后台</span>
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
              class="ml-1 flex items-center gap-2 rounded-xl px-1.5 sm:px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
              <span class="w-7 h-7 rounded-lg flex items-center justify-center bg-gradient-to-br from-blue-500 to-indigo-600 text-xs font-semibold text-white shadow-sm">
                {{ userInitial }}
              </span>
              <span class="hidden sm:block max-w-32 text-sm text-slate-700 dark:text-slate-200 truncate">
                {{ userName }}
              </span>
              <n-icon :size="14" class="hidden sm:block text-slate-400"><ChevronDownOutline /></n-icon>
            </button>
          </n-dropdown>
        </div>
      </header>

      <main class="min-h-0 flex-1 overflow-hidden">
        <router-view />
      </main>

      <footer
        v-if="siteStore.site_copyright"
        class="shrink-0 px-4 py-2 text-center text-xs text-slate-400 dark:text-slate-500 border-t border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70"
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
</template>

<script setup>
import { computed, h, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NButton,
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

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const siteStore = useSiteStore()
const ui = useUiStore()
const message = useMessage()

const userName = computed(() => authStore.user?.display_name || authStore.user?.username || '用户')
const userInitial = computed(() => userName.value.slice(0, 1))
const canReturnToChat = computed(() => authStore.hasPerm('menu:chat'))
const drawerHeaderStyle = { padding: '16px 18px', borderBottom: '1px solid var(--ui-border)' }
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
