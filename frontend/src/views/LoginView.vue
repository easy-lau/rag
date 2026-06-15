<template>
  <div class="relative flex h-screen w-full items-center justify-center bg-gray-50 dark:bg-gray-900">
    <div class="w-full max-w-sm mx-4">
      <!-- Logo -->
      <div class="flex flex-col items-center mb-8">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="w-12 h-12 rounded-xl object-cover mb-3"
          alt="logo"
        />
        <div v-else class="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center text-white font-bold text-xl mb-3">
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="text-lg font-bold text-gray-800 dark:text-white">{{ siteStore.site_title }}</div>
        <div class="text-xs text-gray-500 mt-1">{{ siteStore.site_description }}</div>
      </div>

      <!-- Card -->
      <div class="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm p-6">
        <h2 class="text-base font-semibold text-gray-800 dark:text-gray-200 mb-5 text-center">账号登录</h2>
        <n-form ref="formRef" :model="form" :rules="rules" label-placement="top" @keyup.enter="handleLogin">
          <n-form-item label="用户名" path="username">
            <n-input v-model:value="form.username" placeholder="请输入用户名" :input-props="{ autocomplete: 'username' }">
              <template #prefix><n-icon><PersonOutline /></n-icon></template>
            </n-input>
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input
              v-model:value="form.password" type="password" show-password-on="click"
              placeholder="请输入密码" :input-props="{ autocomplete: 'current-password' }"
            >
              <template #prefix><n-icon><LockClosedOutline /></n-icon></template>
            </n-input>
          </n-form-item>
          <n-button type="primary" block size="large" class="mt-2" :loading="loading" @click="handleLogin">
            登录
          </n-button>
        </n-form>
      </div>
    </div>
    <footer
      v-if="siteStore.site_copyright"
      class="absolute bottom-0 inset-x-0 py-4 text-center text-xs text-gray-400 dark:text-gray-500"
    >
      {{ siteStore.site_copyright }}
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { NForm, NFormItem, NInput, NButton, NIcon, useMessage } from 'naive-ui'
import { PersonOutline, LockClosedOutline } from '@vicons/ionicons5'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { firstAccessibleRoute } from '@/router/menus'

const router = useRouter()
const message = useMessage()
const authStore = useAuthStore()
const siteStore = useSiteStore()

const formRef = ref(null)
const loading = ref(false)
const form = ref({ username: '', password: '' })
const rules = {
  username: { required: true, message: '请输入用户名', trigger: 'blur' },
  password: { required: true, message: '请输入密码', trigger: 'blur' },
}

async function handleLogin() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    await authStore.login(form.value.username, form.value.password)
    // 跳转到第一个有权限的菜单（无则回退 /chat）
    router.push(firstAccessibleRoute(authStore))
  } catch (e) {
    const detail = e?.response?.data?.detail
    message.error(detail || '登录失败，请检查用户名或密码')
  } finally {
    loading.value = false
  }
}
</script>
