const MAX_CLARIFICATION_CHOICES = 6
const MAX_QUESTION_CHARS = 2000
const MAX_LABEL_CHARS = 240
const MAX_REPLY_CHARS = 32
const MAX_IDENTIFIER_CHARS = 160
const MAX_CHOICE_METADATA_ITEMS = 12

const CLARIFICATION_DIMENSIONS = new Set([
  'version',
  'product',
  'product_version',
  'project',
  'document',
])

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

export function normalizeEvidenceClarification(value) {
  const payload = recordValue(value)
  if (!payload || payload.needs_clarification === false) return null

  const rawChoices = Array.isArray(payload.choices) ? payload.choices : []
  const choices = normalizeChoices(rawChoices)
  const dimension = boundedText(payload.dimension, 40)
  const submittedReply = boundedText(payload.submitted_reply, MAX_REPLY_CHARS)
  const pendingStateId = boundedIdentifier(payload.pending_state_id)
  const clarificationMessageId = boundedIdentifier(payload.clarification_message_id)
  const revision = routeStateRevision(payload.route_state_revision)
  const persisted = payload.persisted === true
  // An acknowledgement is only actionable when it identifies a concrete,
  // persisted route-state revision. Merely receiving choices must never make
  // a live picker clickable.
  const acknowledged = payload.acknowledged === true
    && persisted
    && Boolean(pendingStateId)
    && Boolean(clarificationMessageId)
    && revision !== null

  return {
    schema_version: boundedText(payload.schema_version, 80) || null,
    needs_clarification: true,
    dimension: CLARIFICATION_DIMENSIONS.has(dimension) ? dimension : null,
    question: boundedText(payload.question, MAX_QUESTION_CHARS),
    reason: boundedText(payload.reason, 160),
    choices,
    requires_refinement: rawChoices.length > MAX_CLARIFICATION_CHOICES || choices.length === 0,
    submitted: payload.submitted === true,
    submitted_reply: submittedReply || null,
    acknowledged,
    persisted,
    invalidated: payload.invalidated === true,
    invalid_reason: boundedText(payload.invalid_reason, 80) || null,
    pending_state_id: pendingStateId || null,
    clarification_message_id: clarificationMessageId || null,
    route_state_revision: revision,
    conversation_id: boundedIdentifier(payload.conversation_id) || null,
    ack_schema_version: boundedText(payload.ack_schema_version, 80) || null,
  }
}

export function isClarificationActive(value) {
  const clarification = normalizeEvidenceClarification(value)
  return Boolean(
    clarification
    && clarification.acknowledged
    && clarification.persisted
    && !clarification.invalidated
    && !clarification.submitted,
  )
}

export function isClarificationSubmittable(value) {
  const clarification = normalizeEvidenceClarification(value)
  return Boolean(isClarificationActive(clarification) && clarification.choices.length)
}

export function clarificationFromSearchEvent(data, fallback = null) {
  const event = recordValue(data)
  if (!event) {
    const normalizedFallback = normalizeEvidenceClarification(fallback)
    return normalizedFallback?.schema_version === 'rag_evidence_clarification.v1'
      ? normalizedFallback
      : null
  }
  const eventMeta = recordValue(event.search_meta) || recordValue(event.meta) || {}
  const candidate = event.clarification ?? eventMeta.clarification ?? fallback
  const clarification = normalizeEvidenceClarification(candidate)
  return clarification?.schema_version === 'rag_evidence_clarification.v1'
    ? clarification
    : null
}

export function attachEvidenceClarification(message, payload) {
  const target = recordValue(message)
  const clarification = normalizeEvidenceClarification(payload)
  if (!target || !clarification) return null

  const previous = normalizeEvidenceClarification(target.clarification)
  const priorFailureReason = boundedText(target.clarification_failure_reason, 80)
  target.clarification = {
    ...clarification,
    submitted: previous?.submitted === true || clarification.submitted,
    submitted_reply: previous?.submitted_reply || clarification.submitted_reply,
    // This helper is used for live clarification/search events. Those events
    // precede persistence, so an acknowledgement embedded in them is ignored.
    // Only acknowledgeEvidenceClarification() may unlock the picker.
    acknowledged: previous?.acknowledged === true,
    persisted: previous?.persisted === true,
    invalidated: previous?.invalidated === true || Boolean(priorFailureReason),
    invalid_reason: previous?.invalid_reason || priorFailureReason || null,
    pending_state_id: previous?.pending_state_id || clarification.pending_state_id,
    clarification_message_id: previous?.clarification_message_id || clarification.clarification_message_id,
    route_state_revision: previous?.route_state_revision ?? clarification.route_state_revision,
    conversation_id: previous?.conversation_id || clarification.conversation_id,
    ack_schema_version: previous?.ack_schema_version || null,
  }
  return target.clarification
}

export function acknowledgeEvidenceClarification(message, payload) {
  const target = recordValue(message)
  const clarification = normalizeEvidenceClarification(target?.clarification)
  const ack = recordValue(payload)
  if (!target || !clarification || !ack || clarification.invalidated || clarification.submitted) return null

  const pendingStateId = boundedIdentifier(ack.pending_state_id)
  const clarificationMessageId = boundedIdentifier(ack.clarification_message_id)
  const conversationId = boundedIdentifier(ack.conversation_id)
  const revision = routeStateRevision(ack.route_state_revision)
  if (
    ack.type !== 'evidence_clarification_ack'
    || ack.schema_version !== 'rag_evidence_clarification_ack.v1'
    || ack.persisted !== true
    || !pendingStateId
    || !clarificationMessageId
    || !conversationId
    || revision === null
  ) return null

  // If the original event already identified a state, a different ack must
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
    acknowledged: true,
    persisted: true,
    invalidated: false,
    invalid_reason: null,
    pending_state_id: pendingStateId,
    clarification_message_id: clarificationMessageId,
    route_state_revision: revision,
    conversation_id: conversationId,
    ack_schema_version: boundedText(ack.schema_version, 80) || null,
  }
  return target.clarification
}

export function invalidateEvidenceClarification(message, reason = 'stream_failed') {
  const target = recordValue(message)
  if (!target) return null
  const normalizedReason = boundedText(reason, 80) || 'stream_failed'
  if (!boundedText(target.clarification_failure_reason, 80)) {
    target.clarification_failure_reason = normalizedReason
  }
  const clarification = normalizeEvidenceClarification(target?.clarification)
  if (!clarification || clarification.submitted) return null
  if (clarification.invalidated) return clarification

  target.clarification = {
    ...clarification,
    invalidated: true,
    invalid_reason: normalizedReason,
  }
  return target.clarification
}

export function lockMessageClarificationEvidence(message, value = message?.clarification) {
  const target = recordValue(message)
  const clarification = normalizeEvidenceClarification(value)
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
  if (!payload) return normalizeEvidenceClarification(message?.clarification)

  if (payload.type === 'evidence_clarification') {
    if (payload.schema_version !== 'rag_evidence_clarification.v1') {
      invalidateEvidenceClarification(message, 'clarification_schema_mismatch')
      return null
    }
    return attachEvidenceClarification(message, payload)
  }
  if (payload.type === 'evidence_clarification_ack') {
    return acknowledgeEvidenceClarification(message, payload)
  }
  if (payload.type === 'error') {
    return invalidateEvidenceClarification(message, 'server_error')
  }
  if (payload.type === 'done') {
    const clarification = normalizeEvidenceClarification(message?.clarification)
    if (clarification && !clarification.acknowledged) {
      return invalidateEvidenceClarification(message, 'missing_persistence_ack')
    }
    return clarification
  }
  return normalizeEvidenceClarification(message?.clarification)
}

export function restoreHistoryMessageClarification(message) {
  const source = recordValue(message)
  if (!source || source.role !== 'assistant') return source

  const clarification = normalizeEvidenceClarification(source.clarification)
  if (!clarification) return source

  const messageId = boundedIdentifier(source.id)
  const isVerifiedActiveState = clarification.schema_version === 'rag_evidence_clarification.v1'
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

export function markClarificationSubmitted(message, reply, { allowFreeText = false } = {}) {
  const target = recordValue(message)
  const clarification = normalizeEvidenceClarification(target?.clarification)
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
  }
  return true
}

export { MAX_CLARIFICATION_CHOICES }
