import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useColorMode, useMediaQuery } from '@vueuse/core'

export const useUiStore = defineStore('ui', () => {
  const mode = useColorMode({ storageKey: 'rag-theme' })
  let themeUnlockFrame = 0
  let themeUnlockSecondFrame = 0
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

  function lockThemeTransition() {
    if (typeof document === 'undefined' || typeof requestAnimationFrame === 'undefined') return
    const root = document.documentElement
    if (themeUnlockFrame) cancelAnimationFrame(themeUnlockFrame)
    if (themeUnlockSecondFrame) cancelAnimationFrame(themeUnlockSecondFrame)
    root.classList.add('theme-switching')
    // 保持两帧无过渡：第一帧让 Vue/Naive UI 完成主题变量更新，第二帧再恢复
    // hover 等正常动效，避免局部控件仍使用上一主题颜色产生闪帧。
    themeUnlockFrame = requestAnimationFrame(() => {
      themeUnlockSecondFrame = requestAnimationFrame(() => {
        root.classList.remove('theme-switching')
        themeUnlockFrame = 0
        themeUnlockSecondFrame = 0
      })
    })
  }

  function toggleTheme() {
    const nextMode = mode.value === 'dark' ? 'light' : 'dark'
    lockThemeTransition()
    // 先同步根节点 class，再更新响应式状态。App.vue 会在同一调用栈读取新 token，
    // 从而让 Tailwind dark 样式与 Naive UI Theme Overrides 原子切换。
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', nextMode === 'dark')
    }
    mode.value = nextMode
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
