import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getSettings, updateSettings } from '@/api/settings'

export const useSettingsStore = defineStore('settings', () => {
  const data = ref({
    llm_api_key: '',
    llm_base_url: 'https://api.openai.com/v1',
    chat_model: 'gpt-4o',
    temperature: 0.7,
    max_tokens: 2048,
    embedding_api_key: '',
    embedding_base_url: 'https://api.openai.com/v1',
    embedding_model: 'text-embedding-3-small',
    vision_api_key: '',
    vision_base_url: 'https://api.openai.com/v1',
    vision_model: 'gpt-4o',
    top_k: 5,
    rerank_enabled: true,
    show_sources: true,
    site_title: 'RAG 检索系统',
    site_description: '知识增强·精准问答',
    site_logo: '',
    browser_title: '',
  })
  const loading = ref(false)

  async function fetch() {
    loading.value = true
    try { data.value = await getSettings() }
    finally { loading.value = false }
  }

  async function save(payload) {
    data.value = await updateSettings(payload)
  }

  return { data, loading, fetch, save }
})
