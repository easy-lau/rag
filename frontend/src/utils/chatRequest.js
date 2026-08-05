const MAX_ID_CHARS = 128

/**
 * Return a client generated id for one logical chat turn.
 *
 * The id is deliberately generated in the browser and sent both as a body
 * field and a header.  A retry must pass the original value back to
 * `createChatStream`; the server can then make persistence idempotent without
 * exposing any user text in a diagnostic identifier.
 */
export function createClientRequestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()

  // Older embedded browsers may not expose randomUUID.  This fallback still
  // has UUID shape and uses the platform CSPRNG when it is available.
  const bytes = new Uint8Array(16)
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = [...bytes].map(value => value.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function normalizeClientRequestId(value) {
  if (typeof value !== 'string') return ''
  const normalized = value.trim().slice(0, MAX_ID_CHARS)
  // Only allow opaque UUID-like tokens to cross the network.  This prevents a
  // caller accidentally putting a question, cookie, or stack trace in a
  // request header while retaining compatibility with server-issued ids.
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(normalized) ? normalized : ''
}

export function normalizeTraceId(value) {
  if (typeof value !== 'string') return ''
  const normalized = value.trim().slice(0, MAX_ID_CHARS)
  // Trace ids are opaque, but keep the display/copy surface conservative.
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{3,127}$/.test(normalized) ? normalized : ''
}

export function normalizeTurnId(value) {
  if (typeof value !== 'string') return ''
  const normalized = value.trim()
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(normalized)
    ? normalized
    : ''
}

function messageRequestId(message) {
  return normalizeClientRequestId(message?.request_id)
    || normalizeClientRequestId(message?.server_request_id)
}

const RETRY_PRESENTATION_FIELDS = [
  'content',
  'sources',
  'tokens',
  'clarification',
  'clarification_failure_reason',
  'stream_errors',
  'evidence_status',
  'answer_provenance',
  'retrieval_executed',
  'search_snapshot',
  'search_meta',
  'intent_decision',
]

/**
 * Collapse accidental local duplicates for one durable request while keeping
 * the first user/assistant pair in its original conversation position.
 */
export function coalesceLogicalTurnMessages(messages, requestId) {
  const source = Array.isArray(messages) ? messages : []
  const normalizedRequestId = normalizeClientRequestId(requestId)
  if (!normalizedRequestId) {
    return { messages: source, userMessage: null, assistantMessage: null, removedCount: 0 }
  }

  const matching = source.filter(message => messageRequestId(message) === normalizedRequestId)
  const userMessage = matching.find(message => message?.role === 'user') || null
  const assistantMessage = matching.find(message => message?.role === 'assistant') || null
  if (!userMessage && !assistantMessage) {
    return { messages: source, userMessage: null, assistantMessage: null, removedCount: 0 }
  }

  const collapsed = source.filter(message => {
    if (messageRequestId(message) !== normalizedRequestId) return true
    if (message?.role === 'user') return message === userMessage
    if (message?.role === 'assistant') return message === assistantMessage
    return true
  })
  return {
    messages: collapsed,
    userMessage,
    assistantMessage,
    removedCount: source.length - collapsed.length,
  }
}

/** Reset only transient answer state; durable identity and list position stay. */
export function resetAssistantForLogicalRetry(message, requestId) {
  if (!message || typeof message !== 'object') return null
  const normalizedRequestId = normalizeClientRequestId(requestId)
  if (!normalizedRequestId) return null
  const preserved = {
    id: message.id,
    created_at: message.created_at,
    turn_id: message.turn_id || null,
    clarification_parent_message_id: message.clarification_parent_message_id || null,
  }
  Object.assign(message, {
    role: 'assistant',
    content: '',
    sources: [],
    stopped: false,
    tokens: null,
    clarification: null,
    stream_errors: [],
    request_id: normalizedRequestId,
    server_request_id: normalizedRequestId,
    trace_id: null,
    evidence_status: null,
    answer_provenance: null,
    retrieval_executed: null,
    delivery_status: 'streaming',
    persistence_status: 'pending',
    turn_status: preserved.turn_id ? 'accepted' : null,
    retryable: false,
    failure_reason: null,
    same_request_recoverable: null,
    retry_with_new_request_id: false,
    replayed: false,
    error_code: null,
    http_status: null,
    error_detail: null,
    clarification_failure_reason: null,
    search_snapshot: null,
    search_meta: null,
    ...preserved,
  })
  return message
}

/**
 * Keep the last visible answer while an idempotent replay is being accepted.
 * The referenced arrays/objects are safe here because reset replaces them; it
 * does not mutate the previous evidence or clarification payload in place.
 */
export function captureAssistantPresentationForRetry(message) {
  if (!message || typeof message !== 'object') return null
  const snapshot = {}
  for (const field of RETRY_PRESENTATION_FIELDS) {
    snapshot[field] = {
      present: Object.prototype.hasOwnProperty.call(message, field),
      value: message[field],
    }
  }
  return snapshot
}

/** Restore only user-visible answer data; keep the newest transport metadata. */
export function restoreAssistantPresentationAfterUnconfirmedRetry(message, snapshot) {
  if (!message || typeof message !== 'object' || !snapshot || typeof snapshot !== 'object') {
    return false
  }
  for (const field of RETRY_PRESENTATION_FIELDS) {
    const entry = snapshot[field]
    if (entry?.present) message[field] = entry.value
    else delete message[field]
  }
  return true
}

export function retryWithNewRequestIdForHttpStatus(status) {
  return Number(status) === 409
}

const AUTHORITATIVE_TURN_STATUSES = new Set([
  'completed',
  'failed',
  'error',
  'persist_failed',
  'save_failed',
  'cancelled',
])

/**
 * A transport handshake or progress event must not erase the previous visible
 * answer. Only an event that supplies replacement presentation state, closes
 * the turn, or persists a new picker makes the retry snapshot obsolete.
 */
export function streamEventConfirmsAssistantPresentation(event) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) return false
  const type = typeof event.type === 'string' ? event.type.trim() : ''
  if (type === 'text_delta') return typeof event.content === 'string' && event.content.length > 0
  if (type === 'search_results' || type === 'error' || type === 'evidence_clarification_ack') {
    return true
  }
  if (type === 'done') return true
  if (type !== 'turn_state') return false
  const status = String(event.turn_status ?? event.status ?? '').trim().toLowerCase()
  return AUTHORITATIVE_TURN_STATUSES.has(status)
}

function boundedErrorText(value, maxChars = 500) {
  if (typeof value !== 'string') return ''
  return value.replace(/\s+/g, ' ').trim().slice(0, maxChars)
}

/**
 * Preserve FastAPI's string detail and future structured error payloads while
 * deriving a fallback retry contract from the HTTP status.  New servers return
 * the authoritative retry flags because a 503 can mean either an uncertain
 * first persistence attempt (same id) or a terminal contract rejection (new
 * id).  Status-only handling remains solely for rolling-upgrade compatibility.
 */
export function chatRequestHttpFailure(status, body = null) {
  const numericStatus = Number.isFinite(Number(status)) ? Number(status) : null
  const payload = body && typeof body === 'object' && !Array.isArray(body) ? body : {}
  const structuredDetail = payload.detail && typeof payload.detail === 'object' && !Array.isArray(payload.detail)
    ? payload.detail
    : null
  const publicMessage = boundedErrorText(
    typeof payload.detail === 'string'
      ? payload.detail
      : (structuredDetail?.message || payload.message),
    180,
  ) || '请求失败，请稍后重试'
  const errorCode = boundedErrorText(
    payload.error_code
      || payload.code
      || structuredDetail?.error_code
      || structuredDetail?.code,
    80,
  ) || (numericStatus === 409 ? 'request_conflict' : '')
  const declaredRetryWithNewRequestId = typeof structuredDetail?.retry_with_new_request_id === 'boolean'
    ? structuredDetail.retry_with_new_request_id
    : (typeof payload.retry_with_new_request_id === 'boolean'
        ? payload.retry_with_new_request_id
        : null)
  const declaredSameRequestRecoverable = typeof structuredDetail?.same_request_recoverable === 'boolean'
    ? structuredDetail.same_request_recoverable
    : (typeof payload.same_request_recoverable === 'boolean'
        ? payload.same_request_recoverable
        : null)
  const legacyRetryWithNewRequestId = retryWithNewRequestIdForHttpStatus(numericStatus)
  const retryWithNewRequestId = declaredRetryWithNewRequestId
    ?? (declaredSameRequestRecoverable === null
        ? legacyRetryWithNewRequestId
        : !declaredSameRequestRecoverable)
  const sameRequestRecoverable = declaredSameRequestRecoverable
    ?? (declaredRetryWithNewRequestId === null
        ? (legacyRetryWithNewRequestId ? false : (numericStatus === 503 ? true : null))
        : !declaredRetryWithNewRequestId)

  return {
    status: numericStatus,
    detail: structuredDetail || boundedErrorText(payload.detail, 500) || null,
    errorCode: errorCode || null,
    publicMessage,
    failureReason: retryWithNewRequestId ? 'request_conflict' : 'request_failed',
    retryWithNewRequestId,
    sameRequestRecoverable,
  }
}

export function responseHeader(response, name) {
  if (!response?.headers) return ''
  try {
    return response.headers.get?.(name) || response.headers.get?.(name.toLowerCase()) || ''
  } catch {
    return ''
  }
}

export function traceIdFromResponse(response) {
  return normalizeTraceId(responseHeader(response, 'X-RAG-Trace-ID'))
}

/**
 * Reuse an idempotency key only to recover an uncertain/durability-failed
 * delivery.  A user clicking “重新生成” on a completed, cancelled, or explicit
 * model-failed turn is asking for a new logical turn and must receive a new
 * request id.
 */
export function recoverableRetryRequestId(answerMessage, userMessage) {
  const answer = answerMessage && typeof answerMessage === 'object' ? answerMessage : {}
  const serverTurnStatus = typeof answer.turn_status === 'string'
    ? answer.turn_status.trim().toLowerCase()
    : ''
  const hasServerErrorCode = typeof answer.error_code === 'string' && Boolean(answer.error_code.trim())
  // The server has reloaded the durable turn and knows that no generated
  // payload can be recovered. Its explicit fresh-request instruction wins over
  // generic failed persistence fields from the same event.
  if (
    answer.retry_with_new_request_id === true
    || answer.failure_reason === 'persistence_unrecoverable'
    || answer.failure_reason === 'request_conflict'
    || (
      answer.same_request_recoverable !== true
      && hasServerErrorCode
      && ['failed', 'cancelled'].includes(serverTurnStatus)
    )
  ) {
    return ''
  }
  const recoveryReasons = new Set([
    'request_failed',
    'stream_incomplete',
    'protocol_parse_error',
    'protocol_handler_error',
    'persistence_failed',
    'turn_in_progress',
  ])
  const shouldRecover = answer.same_request_recoverable === true
    || answer.persistence_status === 'failed'
    || answer.turn_status === 'persist_failed'
    || recoveryReasons.has(answer.failure_reason)
  if (!shouldRecover) return ''
  return normalizeClientRequestId(
    answer.request_id,
  ) || normalizeClientRequestId(
    answer.server_request_id,
  ) || normalizeClientRequestId(userMessage?.request_id)
}
