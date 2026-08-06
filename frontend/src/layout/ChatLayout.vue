<template>
  <div class="chat-workspace">
    <!-- 宽屏保留常驻会话栏；1024px 以下改为抽屉，避免主对话区被两侧栏挤压。 -->
    <div v-if="!ui.isCompact" class="chat-workspace__sidebar">
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

    <div class="chat-workspace__main">
      <AppHeader />
      <main class="chat-workspace__content">
        <router-view />
      </main>
      <footer
        v-if="siteStore.site_copyright"
        class="chat-workspace__footer"
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

<style scoped>
.chat-workspace {
  display: flex;
  height: 100dvh;
  min-height: 0;
  overflow: hidden;
  background:
    radial-gradient(circle at 72% -8%, color-mix(in srgb, var(--ui-primary) 7%, transparent), transparent 34%),
    var(--ui-bg-subtle);
}

.chat-workspace__sidebar {
  width: 276px;
  flex: 0 0 276px;
}

.chat-workspace__main {
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
}

.chat-workspace__content {
  min-height: 0;
  flex: 1 1 auto;
  overflow: hidden;
}

.chat-workspace__footer {
  flex: 0 0 auto;
  border-top: 1px solid var(--ui-divider);
  background: color-mix(in srgb, var(--ui-surface) 88%, transparent);
  padding: 7px 16px;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  text-align: center;
}
</style>
