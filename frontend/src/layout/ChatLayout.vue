<template>
  <div class="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
    <!-- 宽屏保留常驻会话栏；1024px 以下改为抽屉，避免主对话区被两侧栏挤压。 -->
    <div v-if="!ui.isCompact" class="w-64 shrink-0">
      <ChatSidebar />
    </div>
    <n-drawer
      v-else
      v-model:show="ui.mobileNavOpen"
      :width="ui.isMobile ? 280 : 304"
      placement="left"
      to="#app"
      @update:show="updateNavDrawer"
    >
      <ChatSidebar in-drawer @close-drawer="closeNavDrawer" />
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
import { onBeforeUnmount, watch } from 'vue'
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
  closeNavDrawer()
})

// 从紧凑屏切回宽屏时，抽屉已经不再渲染，不能留下一个跨布局的打开状态。
watch(() => ui.isCompact, isCompact => {
  if (!isCompact) closeNavDrawer()
})

// 检索结果只属于当前问答工作台；离开后不应带入下一次进入或后台页面。
onBeforeUnmount(() => {
  ui.closeChatSearch()
})

function updateNavDrawer(show) {
  ui.mobileNavOpen = show
}

function closeNavDrawer() {
  ui.mobileNavOpen = false
}
</script>
