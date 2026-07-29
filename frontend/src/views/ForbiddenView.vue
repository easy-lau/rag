<template>
  <main class="h-full flex items-center justify-center px-6 bg-gray-50 dark:bg-gray-900">
    <section class="max-w-md text-center">
      <div class="mx-auto w-12 h-12 rounded-2xl bg-gray-200 dark:bg-gray-800 flex items-center justify-center text-gray-500 dark:text-gray-400">
        <n-icon :size="25"><LockClosedOutline /></n-icon>
      </div>
      <h1 class="mt-5 text-lg font-semibold text-gray-800 dark:text-gray-100">暂无可访问的工作区</h1>
      <p class="mt-2 text-sm leading-6 text-gray-500 dark:text-gray-400">请联系管理员为当前账号分配问答或后台管理权限。</p>
      <n-button class="mt-6" @click="authStore.logout()">返回登录页</n-button>
    </section>
  </main>
</template>

<script setup>
import { NButton, NIcon } from 'naive-ui'
import { LockClosedOutline } from '@vicons/ionicons5'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()
</script>
