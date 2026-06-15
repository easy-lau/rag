import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getSiteSettings } from '@/api/settings'

// 公开品牌信息（无需权限），与需要 settings:read 权限的 settings store 区分开。
// 任何页面（含登录页、普通用户）都可读取，用于侧边栏品牌区与浏览器标题。
export const useSiteStore = defineStore('site', () => {
  const site_title = ref('RAG 检索系统')
  const site_description = ref('知识增强·精准问答')
  const site_logo = ref('')
  const browser_title = ref('')
  const site_copyright = ref('')

  async function fetchSite() {
    try {
      const data = await getSiteSettings()
      site_title.value = data.site_title || site_title.value
      site_description.value = data.site_description || site_description.value
      site_logo.value = data.site_logo || ''
      browser_title.value = data.browser_title || ''
      site_copyright.value = data.site_copyright || ''
    } catch {
      // 静默保留默认值，不阻断页面渲染
    } finally {
      document.title = browser_title.value || site_title.value || 'RAG 检索系统'
    }
  }

  return { site_title, site_description, site_logo, browser_title, site_copyright, fetchSite }
})
