<template>
  <div class="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
    <AppSidebar />
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
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'
import { useSiteStore } from '@/stores/site'

const siteStore = useSiteStore()
</script>
