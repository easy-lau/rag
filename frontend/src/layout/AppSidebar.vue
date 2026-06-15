<template>
  <aside class="w-52 flex flex-col bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 shrink-0">
    <!-- Logo -->
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
          <div class="text-xs text-gray-500 truncate">{{ siteStore.site_description }}</div>
        </div>
      </div>
    </div>

    <!-- Nav -->
    <nav class="shrink-0 px-2 py-3 space-y-1">
      <router-link
        v-for="item in navItems" :key="item.to"
        :to="item.to"
        class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer"
        :class="(item.match || [item.to]).some(p => $route.path.startsWith(p))
          ? 'bg-blue-500 text-white'
          : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'"
      >
        <n-icon :size="16"><component :is="item.icon" /></n-icon>
        {{ item.label }}
      </router-link>
    </nav>

    <!-- Conversation history -->
    <div v-if="$route.path.startsWith('/chat')" class="px-3 py-3 border-t border-gray-200 dark:border-gray-700 flex-1 overflow-hidden flex flex-col">
      <div class="flex items-center justify-between mb-2 shrink-0">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400">对话历史</span>
        <n-button text size="tiny" @click="chatStore.newConversation()">新建</n-button>
      </div>
      <div class="overflow-y-auto space-y-0.5 flex-1">
        <div v-if="!chatStore.conversations.length" class="text-xs text-gray-400 text-center py-2">暂无历史对话</div>
        <div
          v-for="conv in chatStore.conversations"
          :key="conv.id"
          class="group flex items-center gap-1 px-2 py-1.5 rounded cursor-pointer text-xs"
          :class="conv.id === chatStore.currentConvId
            ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
            : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'"
          @click="chatStore.loadConversation(conv.id)"
        >
          <span class="truncate flex-1">{{ conv.title || '未命名对话' }}</span>
          <n-button
            text size="tiny"
            class="opacity-0 group-hover:opacity-100 shrink-0"
            @click.stop="chatStore.removeConversation(conv.id)"
          >
            <n-icon :size="12"><TrashOutline /></n-icon>
          </n-button>
        </div>
      </div>
    </div>
    <div v-else class="flex-1"></div>

    <!-- Current user + logout -->
    <div class="mt-auto px-3 py-3 border-t border-gray-200 dark:border-gray-700 shrink-0">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs shrink-0">
          {{ (authStore.user?.display_name || authStore.user?.username || '用')[0] }}
        </div>
        <div class="min-w-0 flex-1">
          <div class="text-sm text-gray-700 dark:text-gray-200 truncate">
            {{ authStore.user?.display_name || authStore.user?.username || '未登录' }}
          </div>
          <div v-if="authStore.user?.role_name" class="text-xs text-gray-400 truncate">{{ authStore.user.role_name }}</div>
        </div>
        <n-button text size="tiny" title="退出登录" @click="authStore.logout()">
          <n-icon :size="16"><LogOutOutline /></n-icon>
        </n-button>
      </div>
    </div>

  </aside>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { NIcon, NButton } from 'naive-ui'
import { TrashOutline, LogOutOutline } from '@vicons/ionicons5'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { accessibleMenus } from '@/router/menus'

const chatStore = useChatStore()
const authStore = useAuthStore()
const siteStore = useSiteStore()

// 数据驱动菜单：按当前用户权限过滤（guide 公开常显）。
// /documents 没有独立菜单项，归属于「知识库管理」（其 match 覆盖 /documents）。
const navItems = computed(() => accessibleMenus(authStore))

onMounted(() => {
  if (authStore.hasPerm('menu:chat')) chatStore.loadHistory()
})
</script>
