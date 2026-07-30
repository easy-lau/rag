import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getSiteSettings } from '@/api/settings'

const SITE_FAVICON_SELECTOR = 'link[data-site-favicon="true"]'

function syncDocumentBranding({ title, logo }) {
  document.title = title || 'RAG 检索系统'

  const logoUrl = typeof logo === 'string' ? logo.trim() : ''
  let favicon = document.head.querySelector(SITE_FAVICON_SELECTOR)

  if (!logoUrl) {
    favicon?.remove()
    return
  }

  if (!favicon) {
    favicon = document.createElement('link')
    favicon.rel = 'icon'
    favicon.dataset.siteFavicon = 'true'
    document.head.appendChild(favicon)
  }

  // 上传接口使用唯一文件名；仅在地址变化时更新，避免每次拉取设置都重复请求图片。
  if (favicon.getAttribute('href') !== logoUrl) {
    favicon.setAttribute('href', logoUrl)
  }
}

// 公开品牌信息（无需权限），与需要 settings:read 权限的 settings store 区分开。
// 任何页面（含登录页、普通用户）都可读取，用于侧边栏品牌区、浏览器标题与标签图标。
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
      syncDocumentBranding({
        title: browser_title.value || site_title.value,
        logo: site_logo.value,
      })
    }
  }

  return { site_title, site_description, site_logo, browser_title, site_copyright, fetchSite }
})
