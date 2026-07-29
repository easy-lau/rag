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
          class="group flex items-center gap-1 px-2.5 py-2 rounded-lg cursor-pointer text-sm"
          :class="[
            conv.id === chatStore.currentConvId
              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700',
            { 'cursor-not-allowed opacity-50': chatStore.isStreaming },
          ]"
          @click="selectConversation(conv.id)"
        >
          <span class="truncate flex-1">{{ conv.title || '未命名对话' }}</span>
          <n-button
            text size="tiny" class="opacity-0 group-hover:opacity-100 shrink-0" :disabled="chatStore.isStreaming"
            title="删除对话" @click.stop="chatStore.removeConversation(conv.id)"
          >
            <n-icon :size="13"><TrashOutline /></n-icon>
          </n-button>
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
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NButton, NIcon, useMessage } from 'naive-ui'
import { AddOutline, ChevronForwardOutline, SettingsOutline, TrashOutline } from '@vicons/ionicons5'
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
</style>
