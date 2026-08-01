import { defineStore } from 'pinia'
import { ref } from 'vue'
import { normalizeEvidenceClarification } from '../utils/chatClarification.js'

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
  'needs_clarification',
  'no_hit',
  'unverified',
  'error',
])
const EVIDENCE_ROLES = new Set(['direct', 'related', 'irrelevant'])
const EXECUTED_EVIDENCE_STATUSES = new Set([
  'hit',
  'partial',
  'version_mismatch',
  'needs_clarification',
  'no_hit',
])
const NON_ANSWER_EVIDENCE_STATUSES = new Set([
  'skipped',
  'needs_clarification',
  'no_hit',
  'error',
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

  const schemaVersion = String(decision.schema_version || '')
  const routeDecision = decision.route_decision || decision.semantic_decision
    || (schemaVersion.includes('route_decision') ? decision : null)
  const taskContract = decision.task_contract || decision.contract || decision.execution_contract || decision.route_summary
    || (schemaVersion.includes('task_contract') ? decision : null)
  const isContractPayload = Boolean(
    routeDecision
    || taskContract
    || schemaVersion.includes('route_decision')
    || schemaVersion.includes('task_contract')
  )

  // 新协议必须以服务端编译合同为准，不能再从 operation/action 反推执行策略。
  // action 映射只服务于没有协议版本和合同对象的旧 SSE 事件。
  let needRetrieval = firstDefined(
    taskContract?.need_retrieval,
    taskContract?.needs_retrieval,
    taskContract?.retrieval?.required,
    taskContract?.execution?.need_retrieval,
    !isContractPayload ? decision.need_retrieval : undefined,
    !isContractPayload ? decision.needs_retrieval : undefined,
  )
  if (typeof needRetrieval !== 'boolean') {
    needRetrieval = isContractPayload ? null : decision.action === 'retrieve'
  }
  const responseMode = firstDefined(
    taskContract?.response_mode,
    taskContract?.execution?.response_mode,
    !isContractPayload ? decision.response_mode : undefined,
    !isContractPayload ? ({
      retrieve: 'grounded_qa',
      chat: 'general_chat',
      writing: 'writing',
      system_help: 'platform_help',
    }[decision.action]) : '',
  ) || ''
  const retrievalPolicy = firstDefined(
    taskContract?.retrieval_policy,
    taskContract?.retrieval?.policy,
    taskContract?.execution?.retrieval_policy,
    !isContractPayload ? decision.retrieval_policy : undefined,
    !isContractPayload && typeof needRetrieval === 'boolean'
      ? (needRetrieval ? 'required' : 'skip')
      : '',
  ) || ''
  const operation = routeDecision?.operation || routeDecision?.intent_code || decision.operation || ''

  return {
    ...decision,
    route_decision: routeDecision,
    task_contract: taskContract,
    operation,
    relation: routeDecision?.relation || decision.relation || '',
    readiness: routeDecision?.readiness || decision.readiness || '',
    intent_code: decision.intent_code || operation,
    intent_name: decision.intent_name || operation,
    response_mode: responseMode,
    retrieval_policy: retrievalPolicy,
    need_retrieval: needRetrieval,
    decision_reason: taskContract?.decision_reason || taskContract?.reason || decision.decision_reason || decision.reason || '',
  }
}

export const useSearchStore = defineStore('search', () => {
  const results = ref([])
  const totalCount = ref(0)
  const searchMeta = ref({})
  const intentDecision = ref(null)
  const hasResultEvent = ref(false)
  // live: the current SSE turn; history: a persisted, static snapshot.  The
  // latter intentionally hides the animated process so a refresh never
  // fabricates steps that were not stored by the server.
  const contextMode = ref('live')
  const steps = ref(STEPS.map(s => ({ ...s, status: 'pending' })))

  function resetSteps() {
    steps.value = STEPS.map(s => ({ ...s, status: 'pending' }))
    results.value = []
    totalCount.value = 0
    searchMeta.value = {}
    intentDecision.value = null
    hasResultEvent.value = false
    contextMode.value = 'live'
  }

  function updateStep(key, status) {
    const step = steps.value.find(s => s.key === key)
    if (step) step.status = status
  }

  // 流程结束兜底：把仍在进行中的步骤收尾，避免步骤条一直停在蓝色转圈
  function finishSteps() {
    steps.value.forEach(s => { if (s.status === 'active') s.status = 'done' })
  }

  function setResults(data, fallbackMeta = {}, options = {}) {
    const rawResults = Array.isArray(data?.results) ? data.results : []
    if (!options.historical) contextMode.value = 'live'
    let normalizedResults = rawResults.map(normalizeResult)
    hasResultEvent.value = true

    const eventMeta = data?.search_meta || data?.meta || {}
    const previousClarification = normalizeEvidenceClarification(searchMeta.value.clarification)
      ? searchMeta.value.clarification
      : null
    const incomingClarification = firstDefined(data.clarification, eventMeta.clarification, null)
    const effectiveClarification = previousClarification
      || (normalizeEvidenceClarification(incomingClarification) ? incomingClarification : null)
    // Once a stream has entered clarification, later/out-of-order result
    // events cannot promote candidates back into answer evidence.
    const clarificationLocked = Boolean(effectiveClarification)
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
    if (clarificationLocked) evidenceStatus = 'needs_clarification'

    const hasNoAnswerEvidence = NON_ANSWER_EVIDENCE_STATUSES.has(evidenceStatus)
    if (evidenceStatus === 'no_hit' || evidenceStatus === 'needs_clarification') {
      // no_hit 和 needs_clarification 都可以保留宽召回候选供右侧解释，但任何
      // direct 角色都与最终状态冲突，必须降级为相近资料。用户确认适用范围前，
      // 卡片不得继续显示绿色“回答依据”。
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
    const relatedReferenceCount = evidenceStatus === 'no_hit' || evidenceStatus === 'needs_clarification'
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
    const contextEvidenceCount = hasNoAnswerEvidence
      ? 0
      : (nonNegativeCount(firstDefined(
          data.context_evidence_count,
          eventMeta.context_evidence_count,
          data.answer_source_count,
          eventMeta.answer_source_count,
        )) ?? 0)
    const rawCoverageStatus = firstDefined(
      data.coverage_status,
      eventMeta.coverage_status,
    )
    const coverageStatus = clarificationLocked
      ? 'insufficient'
      : (['complete', 'partial', 'insufficient'].includes(rawCoverageStatus)
          ? rawCoverageStatus
          : '')
    const missingRequirementCount = nonNegativeCount(firstDefined(
      data.missing_requirement_count,
      eventMeta.missing_requirement_count,
    )) ?? 0

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
      clarification: firstDefined(
        effectiveClarification,
        null,
      ),
      retrieval_executed: retrievalExecuted,
      evidence_status: evidenceStatus,
      displayed_candidate_count: displayedCandidateCount,
      hit_count: hitCount,
      direct_evidence_count: directEvidenceCount,
      related_reference_count: relatedReferenceCount,
      irrelevant_reference_count: derivedIrrelevantCount,
      unverified_reference_count: unverifiedReferenceCount,
      context_evidence_count: contextEvidenceCount,
      answer_source_count: contextEvidenceCount,
      coverage_status: coverageStatus,
      missing_requirement_count: missingRequirementCount,
      expansion_attempted: Boolean(firstDefined(
        data.expansion_attempted,
        eventMeta.expansion_attempted,
        false,
      )),
      joint_support_score: firstDefined(
        data.joint_support_score,
        eventMeta.joint_support_score,
        null,
      ),
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

  function setTraceId(traceId) {
    if (typeof traceId !== 'string' || !traceId.trim()) return
    searchMeta.value = { ...searchMeta.value, trace_id: traceId.trim() }
  }

  /**
   * Restore a persisted search result without pretending that an SSE process
   * is still running.  The snapshot shape is intentionally additive: callers
   * may provide `results`, `candidates`, `search_meta`, or any of the server's
   * evidence counters.  `setResults` remains the single fail-closed normalizer.
   */
  function restoreSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') {
      resetSteps()
      return false
    }
    resetSteps()
    contextMode.value = 'history'
    const decision = snapshot.intent_decision
      || snapshot.intentDecision
      || snapshot.route_decision
      || snapshot.task_contract
    if (decision) setIntentDecision(decision)
    const meta = snapshot.search_meta && typeof snapshot.search_meta === 'object'
      ? snapshot.search_meta
      : (snapshot.meta && typeof snapshot.meta === 'object' ? snapshot.meta : {})
    const results = Array.isArray(snapshot.results)
      ? snapshot.results
      : (Array.isArray(snapshot.candidates) ? snapshot.candidates : [])
    setResults(
      { ...meta, ...snapshot, results },
      { ...meta, historical: true },
      { historical: true },
    )
    contextMode.value = 'history'
    return true
  }

  function setClarification(clarification) {
    results.value = results.value.map(item => (
      item?.evidence_role === 'direct'
        ? { ...item, evidence_role: 'related' }
        : item
    ))
    const relatedReferenceCount = results.value.filter(item => item?.evidence_role === 'related').length
    searchMeta.value = {
      ...searchMeta.value,
      evidence_status: 'needs_clarification',
      hit_count: 0,
      direct_evidence_count: 0,
      related_reference_count: Math.max(
        nonNegativeCount(searchMeta.value.related_reference_count) ?? 0,
        relatedReferenceCount,
      ),
      context_evidence_count: 0,
      answer_source_count: 0,
      coverage_status: 'insufficient',
      clarification,
    }
  }

  return {
    results, totalCount, searchMeta, intentDecision, hasResultEvent, contextMode, steps,
    resetSteps, updateStep, finishSteps, setResults, setIntentDecision, setClarification,
    setTraceId, restoreSnapshot,
  }
})
