import { defineStore } from 'pinia'
import { ref } from 'vue'

const STEPS = [
  { key: 'analyze',  label: '问题分析' },
  { key: 'expand',   label: '查询扩展' },
  { key: 'retrieve', label: '检索' },
  { key: 'rerank',   label: '重排' },
  { key: 'generate', label: '生成' },
]

const EVIDENCE_STATUSES = new Set(['skipped', 'hit', 'no_hit', 'unverified', 'error'])

function firstDefined(...values) {
  return values.find(value => value !== undefined && value !== null)
}

function normalizeIntentDecision(decision) {
  if (!decision) return null

  // 新接口把“意图、回答模式、是否检索”拆开保存；旧接口只返回 action，
  // 这里仅作为版本兼容层补齐字段，业务组件不再自行根据 action 猜测检索状态。
  const legacyNeedRetrieval = decision.action === 'retrieve'
  const needRetrieval = Boolean(firstDefined(
    decision.need_retrieval,
    decision.needs_retrieval,
    legacyNeedRetrieval,
  ))
  const responseMode = decision.response_mode || ({
    retrieve: 'grounded_qa',
    chat: 'general_chat',
    writing: 'writing',
    system_help: 'platform_help',
  }[decision.action]) || ''

  return {
    ...decision,
    response_mode: responseMode,
    retrieval_policy: decision.retrieval_policy || (needRetrieval ? 'required' : 'skip'),
    need_retrieval: needRetrieval,
    decision_reason: decision.decision_reason || decision.reason || '',
  }
}

export const useSearchStore = defineStore('search', () => {
  const results = ref([])
  const totalCount = ref(0)
  const searchMeta = ref({})
  const intentDecision = ref(null)
  const hasResultEvent = ref(false)
  const steps = ref(STEPS.map(s => ({ ...s, status: 'pending' })))

  function resetSteps() {
    steps.value = STEPS.map(s => ({ ...s, status: 'pending' }))
    results.value = []
    totalCount.value = 0
    searchMeta.value = {}
    intentDecision.value = null
    hasResultEvent.value = false
  }

  function updateStep(key, status) {
    const step = steps.value.find(s => s.key === key)
    if (step) step.status = status
  }

  // 流程结束兜底：把仍在进行中的步骤收尾，避免步骤条一直停在蓝色转圈
  function finishSteps() {
    steps.value.forEach(s => { if (s.status === 'active') s.status = 'done' })
  }

  function setResults(data, fallbackMeta = {}) {
    results.value = data.results || []
    totalCount.value = data.total || 0
    hasResultEvent.value = true

    const eventMeta = data.search_meta || data.meta || {}
    const explicitStatus = firstDefined(data.evidence_status, eventMeta.evidence_status)
    let retrievalExecuted = firstDefined(data.retrieval_executed, eventMeta.retrieval_executed)
    if (typeof retrievalExecuted !== 'boolean') {
      if (explicitStatus === 'skipped') retrievalExecuted = false
      else if (explicitStatus === 'hit' || explicitStatus === 'no_hit') retrievalExecuted = true
      else retrievalExecuted = null
    }

    let evidenceStatus = EVIDENCE_STATUSES.has(explicitStatus) ? explicitStatus : ''
    if (!evidenceStatus) {
      if (results.value.length) evidenceStatus = 'hit'
      else if (retrievalExecuted === true) evidenceStatus = 'no_hit'
      else if (retrievalExecuted === false || intentDecision.value?.need_retrieval === false) evidenceStatus = 'skipped'
      else evidenceStatus = 'unverified'
    }

    searchMeta.value = {
      ...searchMeta.value,
      ...fallbackMeta,
      ...eventMeta,
      method: firstDefined(data.method, eventMeta.method, fallbackMeta.method, searchMeta.value.method),
      top_k: firstDefined(data.top_k, eventMeta.top_k, fallbackMeta.top_k, searchMeta.value.top_k),
      rerank: firstDefined(data.rerank, eventMeta.rerank, fallbackMeta.rerank, searchMeta.value.rerank),
      retrieval_executed: retrievalExecuted,
      evidence_status: evidenceStatus,
      hit_count: Number(firstDefined(data.hit_count, eventMeta.hit_count, data.total, results.value.length)) || 0,
      decision_reason: firstDefined(
        data.decision_reason,
        eventMeta.decision_reason,
        intentDecision.value?.decision_reason,
        '',
      ),
    }
  }

  function setIntentDecision(decision) {
    intentDecision.value = normalizeIntentDecision(decision)
    if (intentDecision.value?.decision_reason) {
      searchMeta.value = {
        ...searchMeta.value,
        decision_reason: intentDecision.value.decision_reason,
      }
    }
  }

  return {
    results, totalCount, searchMeta, intentDecision, hasResultEvent, steps,
    resetSteps, updateStep, finishSteps, setResults, setIntentDecision,
  }
})
