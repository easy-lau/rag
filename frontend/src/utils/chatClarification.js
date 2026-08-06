const MAX_CLARIFICATION_CHOICES = 20
const MAX_LABEL_CHARS = 240
const MAX_REPLY_CHARS = 120
const MAX_IDENTIFIER_CHARS = 160
const MAX_CHOICE_METADATA_ITEMS = 12
const MAX_KB_SNAPSHOT_ITEMS = 100

const CLARIFICATION_STATE_SCHEMA = 'rag_clarification_state.v1'
const CLARIFICATION_DIMENSION_RE = /^[a-z][a-z0-9_]{0,63}$/

function recordValue(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null
}

function boundedText(value, maxChars) {
  if (typeof value !== 'string') return ''
  return value.replace(/\s+/g, ' ').trim().slice(0, maxChars)
}

function boundedIdentifier(value) {
  return boundedText(value, MAX_IDENTIFIER_CHARS)
}

function boundedMetadataList(value) {
  if (!Array.isArray(value)) return []
  return value
    .filter(item => typeof item === 'string')
    .slice(0, MAX_CHOICE_METADATA_ITEMS)
    .map(item => boundedText(item, MAX_LABEL_CHARS))
    .filter(Boolean)
}

function boundedIdentifierList(value, limit = MAX_KB_SNAPSHOT_ITEMS) {
  if (!Array.isArray(value)) return []
  return [...new Set(value
    .filter(item => typeof item === 'string')
    .map(item => boundedIdentifier(item))
    .filter(Boolean))]
    .slice(0, limit)
}

function routeStateRevision(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
    ? value
    : null
}

function firstChoiceLabel(choice) {
  const direct = boundedText(choice.label, MAX_LABEL_CHARS)
  if (direct) return direct

  for (const field of ['products', 'versions', 'projects', 'filenames']) {
    const values = Array.isArray(choice[field]) ? choice[field] : []
    const label = boundedText(values.find(value => typeof value === 'string'), MAX_LABEL_CHARS)
    if (label) return label
  }
  return ''
}

function normalizeChoices(rawChoices) {
  if (!Array.isArray(rawChoices) || rawChoices.length === 0) return []
  // Never turn a too-large or malformed server list into a partial list. A
  // partial picker would falsely imply that every applicable scope is shown.
  if (rawChoices.length > MAX_CLARIFICATION_CHOICES) return []

  const choices = rawChoices.map((rawChoice, rawIndex) => {
    const choice = recordValue(rawChoice)
    if (!choice) return null
    const index = rawIndex + 1
    const label = firstChoiceLabel(choice)
    if (!label) return null
    const rawKey = boundedText(choice.key, MAX_REPLY_CHARS)
    const reply = /^[A-Za-z][A-Za-z0-9_-]{0,31}$/.test(rawKey)
      ? rawKey
      : String(index)
    return {
      id: `${reply}-${index}`,
      index,
      key: rawKey || null,
      label,
      reply,
      products: boundedMetadataList(choice.products),
      versions: boundedMetadataList(choice.versions),
      projects: boundedMetadataList(choice.projects),
      filenames: boundedMetadataList(choice.filenames),
    }
  })

  if (choices.some(choice => choice === null)) return []

  const seenReplies = new Set()
  return choices.map(choice => {
    if (!seenReplies.has(choice.reply)) {
      seenReplies.add(choice.reply)
      return choice
    }
    // Duplicate keys are not a safe selection token. The bounded numeric
    // index remains accepted by the backend and preserves the advertised row.
    const fallbackReply = String(choice.index)
    seenReplies.add(fallbackReply)
    return { ...choice, reply: fallbackReply, id: `${fallbackReply}-${choice.index}` }
  })
}

export function normalizeClarification(value) {
  const payload = recordValue(value)
  if (
    !payload
    || payload.schema_version !== CLARIFICATION_STATE_SCHEMA
    || payload.needs_clarification === false
  ) return null

  const rawChoices = Array.isArray(payload.choices) ? payload.choices : []
  const choices = normalizeChoices(rawChoices)
  const dimension = boundedText(payload.dimension, 40)
  const submittedReply = boundedText(payload.submitted_reply, MAX_REPLY_CHARS)
  const pendingStateId = boundedIdentifier(payload.pending_state_id)
  const clarificationMessageId = boundedIdentifier(payload.clarification_message_id)
  const submissionRequestId = boundedIdentifier(payload.submission_request_id)
  const lastSubmissionRequestId = boundedIdentifier(payload.last_submission_request_id)
  const lastSubmittedReply = boundedText(payload.last_submitted_reply, MAX_REPLY_CHARS)
  const revision = routeStateRevision(payload.route_state_revision)
  const selectedKbIdsSnapshot = boundedIdentifierList(payload.selected_kb_ids_snapshot)
  const persisted = payload.persisted === true
  const status = ['proposed', 'active'].includes(payload.status)
    ? payload.status
    : 'proposed'

  return {
    schema_version: boundedText(payload.schema_version, 80) || null,
    needs_clarification: true,
    adapter: ['semantic', 'evidence'].includes(payload.adapter) ? payload.adapter : null,
    dimension: CLARIFICATION_DIMENSION_RE.test(dimension) ? dimension : null,
    selection_mode: ['choice', 'refine'].includes(payload.selection_mode)
      ? payload.selection_mode
      : null,
    reason: boundedText(payload.reason_code || payload.reason, 160),
    choices,
    requires_refinement: rawChoices.length > MAX_CLARIFICATION_CHOICES || choices.length === 0,
    submitted: payload.submitted === true,
    submitted_reply: submittedReply || null,
    status,
    persisted,
    invalidated: payload.invalidated === true,
    invalid_reason: boundedText(payload.invalid_reason, 80) || null,
    pending_state_id: pendingStateId || null,
    clarification_message_id: clarificationMessageId || null,
    route_state_revision: revision,
    conversation_id: boundedIdentifier(payload.conversation_id) || null,
    selected_kb_ids_snapshot: selectedKbIdsSnapshot,
    submission_pending: payload.submission_pending === true && payload.submitted === true,
    submission_request_id: submissionRequestId || null,
    retryable: payload.retryable === true && payload.submitted !== true,
    retry_reason: boundedText(payload.retry_reason, 80) || null,
    last_submitted_reply: lastSubmittedReply || null,
    last_submission_request_id: lastSubmissionRequestId || null,
  }
}

export function isClarificationActive(value) {
  const clarification = normalizeClarification(value)
  return Boolean(
    clarification
    && clarification.status === 'active'
    && clarification.persisted
    && !clarification.invalidated
    && !clarification.submitted,
  )
}

export function isClarificationSubmittable(value) {
  const clarification = normalizeClarification(value)
  return Boolean(isClarificationActive(clarification) && clarification.choices.length)
}

export function clarificationFromSearchEvent(data, fallback = null) {
  const event = recordValue(data)
  if (!event) {
    const normalizedFallback = normalizeClarification(fallback)
    return normalizedFallback?.schema_version === CLARIFICATION_STATE_SCHEMA
      ? normalizedFallback
      : null
  }
  const eventMeta = recordValue(event.search_meta) || recordValue(event.meta) || {}
  const candidate = event.clarification ?? eventMeta.clarification ?? fallback
  const clarification = normalizeClarification(candidate)
  return clarification?.schema_version === CLARIFICATION_STATE_SCHEMA
    ? clarification
    : null
}

export function attachClarification(message, payload) {
  const target = recordValue(message)
  const clarification = normalizeClarification(payload)
  if (!target || !clarification) return null

  const previous = normalizeClarification(target.clarification)
  const priorFailureReason = boundedText(target.clarification_failure_reason, 80)
  target.clarification = {
    ...clarification,
    submitted: previous?.submitted === true || clarification.submitted,
    submitted_reply: previous?.submitted_reply || clarification.submitted_reply,
    // Proposed/search events can attach facts but cannot unlock the picker.
    // Only activateClarification() may accept a persisted active event.
    status: previous?.status === 'active' ? 'active' : 'proposed',
    persisted: previous?.persisted === true,
    invalidated: previous?.invalidated === true || Boolean(priorFailureReason),
    invalid_reason: previous?.invalid_reason || priorFailureReason || null,
    pending_state_id: previous?.pending_state_id || clarification.pending_state_id,
    clarification_message_id: previous?.clarification_message_id || clarification.clarification_message_id,
    route_state_revision: previous?.route_state_revision ?? clarification.route_state_revision,
    conversation_id: previous?.conversation_id || clarification.conversation_id,
    submission_pending: previous?.submission_pending === true,
    submission_request_id: previous?.submission_request_id || null,
    retryable: previous?.retryable === true,
    retry_reason: previous?.retry_reason || null,
    last_submitted_reply: previous?.last_submitted_reply || null,
    last_submission_request_id: previous?.last_submission_request_id || null,
  }
  return target.clarification
}

export function activateClarification(message, payload) {
  const target = recordValue(message)
  const clarification = normalizeClarification(target?.clarification)
  const stateEvent = recordValue(payload)
  if (!target || !clarification || !stateEvent || clarification.invalidated || clarification.submitted) return null

  const pendingStateId = boundedIdentifier(stateEvent.pending_state_id)
  const clarificationMessageId = boundedIdentifier(stateEvent.clarification_message_id)
  const conversationId = boundedIdentifier(stateEvent.conversation_id)
  const revision = routeStateRevision(stateEvent.route_state_revision)
  const selectedKbIdsSnapshot = boundedIdentifierList(stateEvent.selected_kb_ids_snapshot)
  if (
    stateEvent.type !== 'clarification_state'
    || stateEvent.schema_version !== CLARIFICATION_STATE_SCHEMA
    || stateEvent.status !== 'active'
    || stateEvent.persisted !== true
    || !pendingStateId
    || !clarificationMessageId
    || !conversationId
    || revision === null
    || (clarification.adapter === 'evidence' && selectedKbIdsSnapshot.length === 0)
  ) return null

  // If the original event already identified a state, a different active event must
  // not unlock it. Do not compare clarification_message_id with the temporary
  // client message id: repeat events can legitimately point at an older,
  // persisted assistant message.
  if (clarification.pending_state_id && clarification.pending_state_id !== pendingStateId) return null
  if (
    clarification.clarification_message_id
    && clarification.clarification_message_id !== clarificationMessageId
  ) return null
  if (clarification.conversation_id && clarification.conversation_id !== conversationId) return null
  if (
    clarification.route_state_revision !== null
    && clarification.route_state_revision !== revision
  ) return null

  target.clarification = {
    ...clarification,
    status: 'active',
    persisted: true,
    invalidated: false,
    invalid_reason: null,
    pending_state_id: pendingStateId,
    clarification_message_id: clarificationMessageId,
    route_state_revision: revision,
    conversation_id: conversationId,
    selected_kb_ids_snapshot: selectedKbIdsSnapshot,
  }
  return target.clarification
}

export function invalidateClarification(message, reason = 'stream_failed') {
  const target = recordValue(message)
  if (!target) return null
  const normalizedReason = boundedText(reason, 80) || 'stream_failed'
  if (!boundedText(target.clarification_failure_reason, 80)) {
    target.clarification_failure_reason = normalizedReason
  }
  const clarification = normalizeClarification(target?.clarification)
  if (!clarification || clarification.submitted) return null
  if (clarification.invalidated) return clarification

  target.clarification = {
    ...clarification,
    invalidated: true,
    invalid_reason: normalizedReason,
  }
  return target.clarification
}

export function lockMessageClarification(message, value = message?.clarification) {
  const target = recordValue(message)
  const clarification = normalizeClarification(value)
  if (!target || !clarification) return null

  target.clarification = clarification
  target.sources = []
  target.evidence_status = 'needs_clarification'
  target.search_meta = {
    ...(recordValue(target.search_meta) || {}),
    evidence_status: 'needs_clarification',
    hit_count: 0,
    direct_evidence_count: 0,
    context_evidence_count: 0,
    answer_source_count: 0,
    coverage_status: 'insufficient',
    answer_sources: [],
    clarification,
  }
  return clarification
}

export function applyClarificationLifecycleEvent(message, event) {
  const payload = recordValue(event)
  if (!payload) return normalizeClarification(message?.clarification)

  if (payload.type === 'clarification_state') {
    if (payload.schema_version !== CLARIFICATION_STATE_SCHEMA) {
      invalidateClarification(message, 'clarification_schema_mismatch')
      return null
    }
    const attached = attachClarification(message, payload)
    if (!attached || payload.status !== 'active') return attached
    return activateClarification(message, payload)
  }
  if (payload.type === 'error') {
    return invalidateClarification(message, 'server_error')
  }
  if (payload.type === 'done') {
    const clarification = normalizeClarification(message?.clarification)
    if (clarification && clarification.status !== 'active') {
      return invalidateClarification(message, 'missing_active_state')
    }
    return clarification
  }
  return normalizeClarification(message?.clarification)
}

export function restoreHistoryMessageClarification(message) {
  const source = recordValue(message)
  if (!source || source.role !== 'assistant') return source

  const clarification = normalizeClarification(source.clarification)
  if (!clarification) return source

  const messageId = boundedIdentifier(source.id)
  const isVerifiedActiveState = clarification.schema_version === CLARIFICATION_STATE_SCHEMA
    && isClarificationActive(clarification)
    && Boolean(messageId)
    && clarification.clarification_message_id === messageId
    && Boolean(clarification.pending_state_id)
    && clarification.route_state_revision !== null

  // The backend only emits clarification for its validated active pending
  // state. Keep a second fail-closed boundary in the client so malformed or
  // stale history payloads cannot create a clickable picker.
  if (!isVerifiedActiveState) return { ...source, clarification: null }

  return {
    ...source,
    sources: [],
    evidence_status: 'needs_clarification',
    search_meta: {
      ...(recordValue(source.search_meta) || {}),
      evidence_status: 'needs_clarification',
      hit_count: 0,
      direct_evidence_count: 0,
      context_evidence_count: 0,
      answer_source_count: 0,
      coverage_status: 'insufficient',
      clarification,
    },
    clarification,
  }
}

export function markClarificationSubmitted(
  message,
  reply,
  { allowFreeText = false, requestId = null } = {},
) {
  const target = recordValue(message)
  const clarification = normalizeClarification(target?.clarification)
  const normalizedReply = boundedText(reply, MAX_REPLY_CHARS)
  const canSubmit = allowFreeText
    ? isClarificationActive(clarification)
    : isClarificationSubmittable(clarification)
  if (!target || !canSubmit || !normalizedReply) return false
  if (!allowFreeText) {
    const isChoiceReply = clarification.choices.some(choice => choice.reply === normalizedReply)
    const isCompareReply = clarification.choices.length > 1 && normalizedReply === '都对比'
    if (!isChoiceReply && !isCompareReply) return false
  }

  target.clarification = {
    ...clarification,
    submitted: true,
    submitted_reply: normalizedReply,
    submission_pending: true,
    submission_request_id: boundedIdentifier(requestId) || null,
    retryable: false,
    retry_reason: null,
    last_submission_request_id: clarification.last_submission_request_id || null,
  }
  return true
}

/**
 * Re-open a persisted picker when the request created by a user's choice did
 * not complete.  Matching the client request id prevents an old failed stream
 * from reactivating a newer attempt after an out-of-order callback.
 */
export function restoreClarificationSubmissionForRetry(
  message,
  requestId,
  reason = 'request_failed',
) {
  const target = recordValue(message)
  const clarification = normalizeClarification(target?.clarification)
  const expectedRequestId = boundedIdentifier(requestId)
  if (
    !target
    || !clarification
    || !clarification.submitted
    || !clarification.submission_pending
  ) return null
  if (
    expectedRequestId
    && clarification.submission_request_id
    && clarification.submission_request_id !== expectedRequestId
  ) return null

  target.clarification = {
    ...clarification,
    submitted: false,
    submitted_reply: null,
    submission_pending: false,
    submission_request_id: null,
    retryable: true,
    retry_reason: boundedText(reason, 80) || 'request_failed',
    last_submitted_reply: clarification.submitted_reply,
    last_submission_request_id: clarification.submission_request_id || expectedRequestId || null,
  }
  return target.clarification
}

export { MAX_CLARIFICATION_CHOICES }
