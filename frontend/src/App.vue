<template>
  <n-config-provider :theme="theme" :theme-overrides="themeOverrides" :locale="zhCN" :date-locale="dateZhCN">
    <!-- Provider-generated overlays are mounted inside #app so they retain the
         same stacking context and Tailwind's #app-scoped utility styles. -->
    <n-loading-bar-provider to="#app">
      <n-message-provider to="#app">
        <n-notification-provider to="#app">
          <n-dialog-provider to="#app">
            <n-modal-provider to="#app">
              <div class="h-screen w-full overflow-hidden">
                <router-view />
              </div>
            </n-modal-provider>
          </n-dialog-provider>
        </n-notification-provider>
      </n-message-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import {
  NConfigProvider,
  NDialogProvider,
  NLoadingBarProvider,
  NMessageProvider,
  NModalProvider,
  NNotificationProvider,
  darkTheme,
  dateZhCN,
  zhCN
} from 'naive-ui'
import { useUiStore } from '@/stores/ui'
import { useSiteStore } from '@/stores/site'
import { createNaiveThemeOverrides } from '@/theme/naive'

const ui = useUiStore()
const siteStore = useSiteStore()

const isDark = computed(() => ui.mode === 'dark')
const theme = computed(() => isDark.value ? darkTheme : null)
const themeOverrides = ref(createNaiveThemeOverrides())

function syncThemeOverrides(dark = isDark.value) {
  // useColorMode、Tailwind dark class 与 Naive UI 必须在同一帧使用同一主题。
  // 主动同步根节点后再读取 token，避免 Theme Overrides 晚一帧造成控件闪白。
  if (typeof document !== 'undefined') {
    document.documentElement.classList.toggle('dark', dark)
  }
  themeOverrides.value = createNaiveThemeOverrides()
}

// 任何页面/登录页加载时都拉取公开品牌信息，应用品牌与浏览器标题。
onMounted(() => {
  siteStore.fetchSite()
  syncThemeOverrides()
})

watch(isDark, dark => syncThemeOverrides(dark), { flush: 'sync' })
</script>
