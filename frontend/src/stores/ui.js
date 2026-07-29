import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useColorMode, useMediaQuery } from '@vueuse/core'

export const useUiStore = defineStore('ui', () => {
  const mode = useColorMode({ storageKey: 'rag-theme' })
  const sidebarCollapsed = ref(false)
  const rightPanelVisible = ref(true)
  // 仅用于问答工作台的检索结果面板；默认收起，由顶栏入口统一控制。
  const chatSearchOpen = ref(false)

  // 响应式分层：手机只负责极窄触控布局；紧凑屏（手机 + 平板）负责抽屉导航、
  // 表格降级等“内容宽度不足”的场景。不要再用一个 768px 断点同时表达两种语义。
  const isMobile = useMediaQuery('(max-width: 639px)')
  const isCompact = useMediaQuery('(max-width: 1023px)')
  // 移动端侧边导航抽屉开关
  const mobileNavOpen = ref(false)

  function toggleTheme() {
    mode.value = mode.value === 'dark' ? 'light' : 'dark'
  }

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function toggleChatSearch() {
    chatSearchOpen.value = !chatSearchOpen.value
  }

  function closeChatSearch() {
    chatSearchOpen.value = false
  }

  return {
    mode, sidebarCollapsed, rightPanelVisible, chatSearchOpen,
    isMobile, isCompact, mobileNavOpen,
    toggleTheme, toggleSidebar, toggleChatSearch, closeChatSearch,
  }
})
