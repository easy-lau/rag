import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useColorMode } from '@vueuse/core'

export const useUiStore = defineStore('ui', () => {
  const mode = useColorMode({ storageKey: 'rag-theme' })
  const sidebarCollapsed = ref(false)
  const rightPanelVisible = ref(true)

  function toggleTheme() {
    mode.value = mode.value === 'dark' ? 'light' : 'dark'
  }

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  return { mode, sidebarCollapsed, rightPanelVisible, toggleTheme, toggleSidebar }
})
