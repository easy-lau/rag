import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useColorMode, useMediaQuery } from '@vueuse/core'

export const useUiStore = defineStore('ui', () => {
  const mode = useColorMode({ storageKey: 'rag-theme' })
  const sidebarCollapsed = ref(false)
  const rightPanelVisible = ref(true)

  // 移动端断点（≤768px）。各处响应式适配（侧边栏抽屉、表格横向滚动等）统一以此为准。
  const isMobile = useMediaQuery('(max-width: 768px)')
  // 移动端侧边导航抽屉开关
  const mobileNavOpen = ref(false)

  function toggleTheme() {
    mode.value = mode.value === 'dark' ? 'light' : 'dark'
  }

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  return { mode, sidebarCollapsed, rightPanelVisible, isMobile, mobileNavOpen, toggleTheme, toggleSidebar }
})
