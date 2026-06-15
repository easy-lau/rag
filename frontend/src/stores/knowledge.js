import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase } from '@/api/knowledge'

export const useKnowledgeStore = defineStore('knowledge', () => {
  const list = ref([])
  const loading = ref(false)

  async function fetchList() {
    loading.value = true
    try {
      list.value = await getKnowledgeBases()
    } finally {
      loading.value = false
    }
  }

  async function create(data) {
    const kb = await createKnowledgeBase(data)
    list.value.unshift(kb)
    return kb
  }

  async function update(id, data) {
    const kb = await updateKnowledgeBase(id, data)
    const idx = list.value.findIndex(k => k.id === id)
    if (idx !== -1) list.value[idx] = kb
    return kb
  }

  async function remove(id) {
    await deleteKnowledgeBase(id)
    list.value = list.value.filter(kb => kb.id !== id)
  }

  return { list, loading, fetchList, create, update, remove }
})
