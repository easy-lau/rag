/**
 * The frontend's single evidence-status contract.
 *
 * A status is not presentation-only: it decides whether a persisted source
 * may be shown as answer evidence and whether a malformed/rolling payload is
 * allowed to retain result candidates.  Keeping this decision in one module
 * prevents the live panel, history restoration and message citations from
 * slowly accepting different status vocabularies.
 *
 * `version_mismatch` is retained solely for old persisted/API payloads.  New
 * V2 execution emits the broader `scope_mismatch`, which must never expose a
 * rejected document as a related reference or answer source.
 */

const STATUS_DEFINITIONS = Object.freeze({
  skipped: {
    label: '已跳过检索',
    tagType: 'default',
    retrievalExecuted: false,
    answerEvidence: false,
    candidatePolicy: 'clear',
  },
  hit: {
    label: '直接命中',
    tagType: 'success',
    retrievalExecuted: true,
    answerEvidence: true,
    candidatePolicy: 'retain',
  },
  partial: {
    label: '部分支撑',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: true,
    candidatePolicy: 'retain',
  },
  // Read compatibility only.  Historical records may have persisted a
  // bounded source before the fail-closed scope contract was introduced.
  version_mismatch: {
    label: '版本不匹配（历史兼容）',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: true,
    candidatePolicy: 'retain',
    legacy: true,
  },
  scope_mismatch: {
    label: '适用范围不匹配',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: false,
    // A rejected candidate belongs to another explicit product/version/project
    // scope, so even a "related" UI card would leak it back into the answer.
    candidatePolicy: 'clear',
  },
  needs_clarification: {
    label: '等待选择范围',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: false,
    candidatePolicy: 'related_only',
  },
  no_hit: {
    label: '无有效证据',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: false,
    candidatePolicy: 'related_only',
  },
  insufficient_evidence: {
    label: '相关资料但证据不足',
    tagType: 'warning',
    retrievalExecuted: true,
    answerEvidence: false,
    candidatePolicy: 'related_only',
  },
  unverified: {
    label: '未验证',
    tagType: 'default',
    retrievalExecuted: null,
    answerEvidence: false,
    candidatePolicy: 'retain',
  },
  error: {
    label: '检索异常',
    tagType: 'error',
    retrievalExecuted: null,
    answerEvidence: false,
    candidatePolicy: 'clear',
  },
})

export const EVIDENCE_STATUSES = new Set(Object.keys(STATUS_DEFINITIONS))

export function normalizeEvidenceStatus(value) {
  if (typeof value !== 'string') return ''
  const status = value.trim().toLowerCase()
  return EVIDENCE_STATUSES.has(status) ? status : ''
}

export function evidenceStatusMeta(value) {
  const status = normalizeEvidenceStatus(value)
  return status ? STATUS_DEFINITIONS[status] : null
}

export function isNonAnswerEvidenceStatus(value) {
  const meta = evidenceStatusMeta(value)
  return Boolean(meta && !meta.answerEvidence)
}

export function isExecutedEvidenceStatus(value) {
  return evidenceStatusMeta(value)?.retrievalExecuted === true
}

export function evidenceCandidatePolicy(value) {
  return evidenceStatusMeta(value)?.candidatePolicy || 'retain'
}

export function evidenceStatusLabel(value, fallback = '—') {
  return evidenceStatusMeta(value)?.label || fallback
}

export function evidenceStatusTagType(value, fallback = 'default') {
  return evidenceStatusMeta(value)?.tagType || fallback
}
