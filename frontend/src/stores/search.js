import { defineStore } from 'pinia'
import { ref } from 'vue'

const STEPS = [
  { key: 'analyze',  label: '问题分析' },
  { key: 'expand',   label: '查询扩展' },
  { key: 'retrieve', label: '检索' },
  { key: 'rerank',   label: '重排' },
  { key: 'generate', label: '生成' },
]

const EVIDENCE_STATUSES = new Set([
  'skipped',
  'hit',
  'partial',
  'version_mismatch',
  'no_hit',
  'unverified',
  'error',
])
const EVIDENCE_ROLES = new Set(['direct', 'related', 'irrelevant'])
const EXECUTED_EVIDENCE_STATUSES = new Set([
  'hit',
  'partial',
  'version_mismatch',
  'no_hit',
])

function firstDefined(...values) {
  return values.find(value => value !== undefined && value !== null)
}

function nonNegativeCount(value) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : null
}

function normalizedEvidenceRole(result) {
  const explicitRole = typeof result?.evidence_role === 'string'
    ? result.evidence_role.trim().toLowerCase()
    : ''
  if (EVIDENCE_ROLES.has(explicitRole)) return explicitRole

  // 新后端可能先返回约束判定、稍后再补 evidence_role。明确不匹配的资料只能
  // 作为相近参考，不能因为旧版 score 很高就被前端展示成回答依据。
  const constraintStatus = typeof result?.constraint_status === 'string'
    ? result.constraint_status.trim().toLowerCase()
    : ''
  if (constraintStatus.includes('mismatch') || constraintStatus.includes('conflict')) {
    return 'related'
  }
  return ''
}

function normalizeResult(result) {
  if (!result || typeof result !== 'object') return result
  return {
    ...result,
    evidence_role: normalizedEvidenceRole(result),
  }
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
    const rawResults = Array.isArray(data?.results) ? data.results : []
    let normalizedResults = rawResults.map(normalizeResult)
    hasResultEvent.value = true

    const eventMeta = data?.search_meta || data?.meta || {}
    const rawExplicitStatus = firstDefined(data.evidence_status, eventMeta.evidence_status)
    const explicitStatus = typeof rawExplicitStatus === 'string'
      ? rawExplicitStatus.trim().toLowerCase()
      : rawExplicitStatus
    let retrievalExecuted = firstDefined(data.retrieval_executed, eventMeta.retrieval_executed)
    if (typeof retrievalExecuted !== 'boolean') {
      if (explicitStatus === 'skipped') retrievalExecuted = false
      else if (EXECUTED_EVIDENCE_STATUSES.has(explicitStatus)) retrievalExecuted = true
      else retrievalExecuted = null
    }

    let evidenceStatus = EVIDENCE_STATUSES.has(explicitStatus) ? explicitStatus : ''
    if (retrievalExecuted === false && evidenceStatus !== 'error') {
      // 没有执行检索就不可能产生回答依据。异常/旧协议即使同时声称 hit，
      // 也以实际执行状态为准，避免把上一轮或伪造候选显示为本轮依据。
      evidenceStatus = 'skipped'
    }
    if (!evidenceStatus) {
      // 旧接口只有 results 时只能证明有召回候选，不能证明存在回答依据。
      // 但明确未执行或路由明确无需检索时，不能被残留 results 覆盖成待验证。
      if (retrievalExecuted === false || intentDecision.value?.need_retrieval === false) evidenceStatus = 'skipped'
      else if (normalizedResults.length) evidenceStatus = 'unverified'
      else if (retrievalExecuted === true) evidenceStatus = 'no_hit'
      else evidenceStatus = 'unverified'
    }

    const hasNoAnswerEvidence = ['no_hit', 'skipped', 'error'].includes(evidenceStatus)
    if (evidenceStatus === 'no_hit') {
      // no_hit 可以保留宽召回候选供右侧解释，但任何 direct 角色都与最终状态
      // 冲突，必须降级为相近资料，卡片不得继续显示绿色“回答依据”。
      normalizedResults = normalizedResults.map(item => (
        item?.evidence_role === 'direct'
          ? { ...item, evidence_role: 'related' }
          : item
      ))
    } else if (evidenceStatus === 'skipped' || evidenceStatus === 'error') {
      // 未执行或执行失败没有可归属于本轮的候选。清空异常旧载荷，避免展示
      // 上一轮残留结果，同时让展示数量与实际列表保持一致。
      normalizedResults = []
    }
    results.value = normalizedResults

    const derivedDirectCount = results.value.filter(item => item?.evidence_role === 'direct').length
    const derivedRelatedCount = results.value.filter(item => item?.evidence_role === 'related').length
    const derivedIrrelevantCount = results.value.filter(item => item?.evidence_role === 'irrelevant').length
    const explicitDirectCount = nonNegativeCount(firstDefined(
      data.direct_evidence_count,
      eventMeta.direct_evidence_count,
    ))
    const explicitRelatedCount = nonNegativeCount(firstDefined(
      data.related_reference_count,
      eventMeta.related_reference_count,
    ))
    // 最终证据状态拥有最高优先级。兼容旧后端或异常载荷时，即使同时携带
    // legacy hit_count/direct 数，也不能让界面出现“未命中 + 回答依据 5”的矛盾。
    const directEvidenceCount = hasNoAnswerEvidence
      ? 0
      : (explicitDirectCount ?? derivedDirectCount)
    const relatedReferenceCount = evidenceStatus === 'no_hit'
      ? Math.max(explicitRelatedCount ?? 0, derivedRelatedCount)
      : (evidenceStatus === 'skipped' || evidenceStatus === 'error'
          ? 0
          : (explicitRelatedCount ?? derivedRelatedCount))
    const displayedCandidateCount = evidenceStatus === 'skipped' || evidenceStatus === 'error'
      ? 0
      : (nonNegativeCount(firstDefined(
          data.displayed_candidate_count,
          eventMeta.displayed_candidate_count,
          data.displayed_result_count,
          eventMeta.displayed_result_count,
          data.total,
          eventMeta.total,
        )) ?? results.value.length)
    totalCount.value = displayedCandidateCount
    const hitCount = hasNoAnswerEvidence
      ? 0
      : (nonNegativeCount(firstDefined(
          data.hit_count,
          eventMeta.hit_count,
        )) ?? directEvidenceCount)
    const effectiveDirectCount = Math.max(directEvidenceCount, hitCount)
    const evidenceRoleKnown = explicitDirectCount !== null
      || explicitRelatedCount !== null
      || results.value.some(item => Boolean(item?.evidence_role))
    const unverifiedReferenceCount = Math.max(
      0,
      displayedCandidateCount
        - effectiveDirectCount
        - relatedReferenceCount
        - derivedIrrelevantCount,
    )

    searchMeta.value = {
      ...searchMeta.value,
      ...fallbackMeta,
      ...eventMeta,
      method: firstDefined(data.method, eventMeta.method, fallbackMeta.method, searchMeta.value.method),
      top_k: firstDefined(data.top_k, eventMeta.top_k, fallbackMeta.top_k, searchMeta.value.top_k),
      rerank: firstDefined(data.rerank, eventMeta.rerank, fallbackMeta.rerank, searchMeta.value.rerank),
      trace_id: firstDefined(data.trace_id, eventMeta.trace_id, searchMeta.value.trace_id),
      query_constraints: firstDefined(
        data.query_constraints,
        eventMeta.query_constraints,
        searchMeta.value.query_constraints,
        {},
      ),
      retrieval_executed: retrievalExecuted,
      evidence_status: evidenceStatus,
      displayed_candidate_count: displayedCandidateCount,
      hit_count: hitCount,
      direct_evidence_count: directEvidenceCount,
      related_reference_count: relatedReferenceCount,
      irrelevant_reference_count: derivedIrrelevantCount,
      unverified_reference_count: unverifiedReferenceCount,
      evidence_role_known: evidenceRoleKnown,
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
