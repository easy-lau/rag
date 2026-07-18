<template>
  <header class="h-12 flex items-center justify-between px-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shrink-0">
    <div class="flex items-center gap-2 min-w-0">
      <n-button v-if="ui.isMobile" text size="small" title="菜单" @click="ui.mobileNavOpen = true">
        <template #icon><n-icon :size="20"><MenuOutline /></n-icon></template>
      </n-button>
      <div class="text-sm font-medium text-gray-700 dark:text-gray-200 truncate">
        {{ currentPageTitle }}
      </div>
    </div>
    <div class="flex items-center gap-3">
      <n-button text size="small" @click="ui.toggleTheme()">
        <template #icon>
          <n-icon><MoonOutline v-if="ui.mode !== 'dark'" /><SunnyOutline v-else /></n-icon>
        </template>
      </n-button>

      <!-- 用户菜单：点击展开「修改密码 / 退出登录」 -->
      <n-dropdown trigger="click" :options="userOptions" @select="handleSelect">
        <div class="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg px-2 py-1 transition-colors">
          <div class="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
            {{ (authStore.user?.display_name || authStore.user?.username || '用')[0] }}
          </div>
          <span>{{ authStore.user?.display_name || authStore.user?.username || '用户' }}</span>
          <n-icon :size="14" class="text-gray-400"><ChevronDownOutline /></n-icon>
        </div>
      </n-dropdown>
    </div>

    <!-- 修改密码弹窗 -->
    <n-modal v-model:show="showPwd" preset="card" title="修改密码" style="width: 90vw; max-width: 400px">
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
        <div class="flex justify-end gap-2">
          <n-button @click="showPwd = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleChangePassword">确定</n-button>
        </div>
      </template>
    </n-modal>
  </header>
</template>

<script setup>
import { computed, ref, h } from 'vue'
import { useRoute } from 'vue-router'
import { NButton, NIcon, NDropdown, NModal, NForm, NFormItem, NInput, useMessage } from 'naive-ui'
import { MoonOutline, SunnyOutline, ChevronDownOutline, KeyOutline, LogOutOutline, MenuOutline } from '@vicons/ionicons5'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'
import { changePassword } from '@/api/auth'

const ui = useUiStore()
const authStore = useAuthStore()
const route = useRoute()
const msg = useMessage()

const titles = {
  chat: '问答对话', knowledge: '知识库管理', documents: '文档管理',
  'search-test': '检索测试', settings: '系统设置',
  users: '用户管理', roles: '角色管理', 'audit-logs': '审计日志'
}
const currentPageTitle = computed(() => titles[route.name] || 'RAG 检索系统')

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
