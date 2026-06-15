<template>
  <n-config-provider :theme="theme" :theme-overrides="themeOverrides" :locale="zhCN" :date-locale="dateZhCN">
    <n-message-provider>
      <n-dialog-provider>
        <div class="h-screen w-full overflow-hidden">
          <router-view />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { NConfigProvider, NMessageProvider, NDialogProvider, darkTheme, zhCN, dateZhCN } from 'naive-ui'
import { useUiStore } from '@/stores/ui'
import { useSiteStore } from '@/stores/site'

const ui = useUiStore()
const siteStore = useSiteStore()

// 任何页面/登录页加载时都拉取公开品牌信息，应用品牌与浏览器标题
onMounted(() => { siteStore.fetchSite() })
const isDark = computed(() => ui.mode === 'dark')
const theme = computed(() => isDark.value ? darkTheme : null)
const themeOverrides = {
  common: { primaryColor: '#3B82F6', primaryColorHover: '#2563EB', primaryColorPressed: '#1D4ED8' }
}
</script>
