<template>
  <div class="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
    <!-- 桌面端：问答专用侧栏；移动端由顶栏的菜单按钮打开抽屉。 -->
    <div v-if="!ui.isMobile" class="w-64 shrink-0">
      <ChatSidebar />
    </div>
    <n-drawer v-else v-model:show="ui.mobileNavOpen" :width="280" placement="left" to="#app">
      <ChatSidebar />
    </n-drawer>

    <div class="flex flex-col flex-1 min-w-0">
      <AppHeader />
      <main class="flex-1 overflow-hidden">
        <router-view />
      </main>
      <footer
        v-if="siteStore.site_copyright"
        class="shrink-0 py-2 px-4 text-center text-xs text-gray-400 dark:text-gray-500 border-t border-gray-100 dark:border-gray-800"
      >
        {{ siteStore.site_copyright }}
      </footer>
    </div>
  </div>
</template>

<script setup>
import { watch } from 'vue'
import { NDrawer } from 'naive-ui'
import { useRoute } from 'vue-router'
import AppHeader from './AppHeader.vue'
import ChatSidebar from './ChatSidebar.vue'
import { useSiteStore } from '@/stores/site'
import { useUiStore } from '@/stores/ui'

const siteStore = useSiteStore()
const ui = useUiStore()
const route = useRoute()

// 两套布局共用移动端抽屉状态；切换路由时主动收起，避免从后台返回后仍遮住问答页。
watch(() => route.fullPath, () => {
  ui.mobileNavOpen = false
})
</script>
