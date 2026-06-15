import { defineStore } from 'pinia'
import { ref } from 'vue'

const STEPS = [
  { key: 'analyze',  label: '问题分析' },
  { key: 'expand',   label: '查询扩展' },
  { key: 'retrieve', label: '检索' },
  { key: 'rerank',   label: '重排' },
  { key: 'generate', label: '生成' },
]

export const useSearchStore = defineStore('search', () => {
  const results = ref([])
  const totalCount = ref(0)
  const searchMeta = ref({})
  const steps = ref(STEPS.map(s => ({ ...s, status: 'pending' })))

  function resetSteps() {
    steps.value = STEPS.map(s => ({ ...s, status: 'pending' }))
    results.value = []
    totalCount.value = 0
    searchMeta.value = {}
  }

  function updateStep(key, status) {
    const step = steps.value.find(s => s.key === key)
    if (step) step.status = status
  }

  // 流程结束兜底：把仍在进行中的步骤收尾，避免步骤条一直停在蓝色转圈
  function finishSteps() {
    steps.value.forEach(s => { if (s.status === 'active') s.status = 'done' })
  }

  function setResults(data) {
    results.value = data.results || []
    totalCount.value = data.total || 0
  }

  return { results, totalCount, searchMeta, steps, resetSteps, updateStep, finishSteps, setResults }
})
