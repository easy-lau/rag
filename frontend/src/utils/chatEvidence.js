import {
  isNonAnswerEvidenceStatus,
  normalizeEvidenceStatus,
} from './evidenceStatus.js'

const EVIDENCE_ROLES = new Set(['direct', 'related', 'irrelevant'])

function normalizedStatus(value) {
  return normalizeEvidenceStatus(value)
}

function normalizedRole(source) {
  const role = typeof source?.evidence_role === 'string'
    ? source.evidence_role.trim().toLowerCase()
    : ''
  return EVIDENCE_ROLES.has(role) ? role : ''
}

function nonNegativeCount(value) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : null
}

function objectList(value, limit) {
  if (!Array.isArray(value)) return []
  return value
    .filter(item => item && typeof item === 'object')
    .slice(0, limit)
}

function eventEvidenceStatus(data) {
  return normalizedStatus(
    data?.evidence_status
      ?? data?.search_meta?.evidence_status
      ?? data?.meta?.evidence_status,
  )
}

/**
 * 从一次 search_results 事件中取得“实际交给回答模型的知识库依据”。
 *
 * results 属于右侧检索面板的展示候选，不能直接挂到回答消息。新版协议明确
 * 返回 answer_sources；旧版协议缺少该字段时，只在能够由 context 数量或 direct
 * 角色证明候选参与了生成时才保守兼容。
 */
export function answerSourcesFromSearchEvent(data, limit = 20) {
  const safeLimit = Math.max(0, Math.trunc(Number(limit) || 0))
  if (!safeLimit || !data || typeof data !== 'object') return []

  const evidenceStatus = eventEvidenceStatus(data)
  // 非回答状态即使错误地携带了候选，也没有任何知识库正文进入回答上下文。
  if (isNonAnswerEvidenceStatus(evidenceStatus)) return []

  if (Object.prototype.hasOwnProperty.call(data, 'answer_sources')) {
    return objectList(data.answer_sources, safeLimit)
  }

  const results = objectList(data.results, safeLimit)
  if (!results.length) return []

  const contextCount = nonNegativeCount(
    data.context_evidence_count
      ?? data.search_meta?.context_evidence_count
      ?? data.meta?.context_evidence_count,
  )
  if (contextCount === 0) return []

  const directSources = results.filter(source => normalizedRole(source) === 'direct')
  if (directSources.length) {
    const count = contextCount === null
      ? directSources.length
      : Math.min(contextCount, directSources.length)
    return directSources.slice(0, count)
  }

  // 当前旧协议会同时返回 context_evidence_count。没有 direct 的 partial、
  // version_mismatch 或 required/unverified 场景，可按该数量还原参与生成的前 N 条。
  if (contextCount !== null) return results.slice(0, Math.min(contextCount, safeLimit))

  // 更早的协议没有角色和上下文数量；只有明确 hit 时才保留兼容，否则宁可不在
  // 回答下展示来源，也不能把召回候选误称为回答依据。
  const hasKnownRole = results.some(source => Boolean(normalizedRole(source)))
  return evidenceStatus === 'hit' && !hasKnownRole ? results : []
}

/**
 * 过滤历史消息中旧版本曾误存入 sources 的非回答状态检索候选。
 */
export function persistedAnswerSources(message) {
  if (!message || typeof message !== 'object') return []
  const messageStatus = normalizedStatus(
    message.evidence_status ?? message.search_meta?.evidence_status,
  )
  if (isNonAnswerEvidenceStatus(messageStatus)) return []

  return objectList(message.sources, Number.MAX_SAFE_INTEGER).filter(source => {
    const sourceStatus = normalizedStatus(
      source.evidence_status ?? source.search_meta?.evidence_status,
    )
    return !isNonAnswerEvidenceStatus(sourceStatus)
  })
}
