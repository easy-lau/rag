const FAILED_STATUSES = new Set(['failed', 'error', 'persist_failed', 'save_failed', 'cancelled'])
const PERSISTENCE_FAILURE_STATUSES = new Set(['failed', 'error', 'persist_failed', 'save_failed'])
const IN_PROGRESS_STATUSES = new Set(['accepted', 'generating', 'generated'])
const COMPLETED_STATUSES = new Set(['completed', 'delivered'])
const LEGACY_PERSISTENCE_FAILURE_MESSAGE = '回答已生成，但保存失败，请重试'

function normalizedStatus(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : ''
}

function eventValue(event, key) {
  if (key === 'turn_status') {
    return event?.turn_status ?? event?.status ?? event?.turn?.status ?? event?.meta?.turn_status ?? null
  }
  return event?.[key] ?? event?.turn?.[key] ?? event?.meta?.[key] ?? null
}

/**
 * Apply monotonic delivery/persistence state.  Once an error or failed save is
 * observed, a trailing SSE `done` may close the transport but must not relabel
 * the turn as delivered/persisted.
 */
export function applyTurnLifecycleState(message, event) {
  if (!message || typeof message !== 'object' || !event || typeof event !== 'object') return message
  const delivery = normalizedStatus(eventValue(event, 'delivery_status'))
  const persistence = normalizedStatus(eventValue(event, 'persistence_status'))
  const turnStatus = normalizedStatus(eventValue(event, 'turn_status'))
  if (typeof event.replayed === 'boolean') message.replayed = event.replayed
  if (typeof event.same_request_recoverable === 'boolean') {
    message.same_request_recoverable = event.same_request_recoverable
  }
  if (typeof event.retry_with_new_request_id === 'boolean') {
    message.retry_with_new_request_id = event.retry_with_new_request_id
  }
  const previousFailed = message.delivery_status === 'failed'
    || message.persistence_status === 'failed'
    || message.turn_status === 'failed'
    || message.retryable === true
  const eventFailed = event.type === 'error'
    || FAILED_STATUSES.has(delivery)
    || FAILED_STATUSES.has(persistence)
    || FAILED_STATUSES.has(turnStatus)

  if (previousFailed || eventFailed) {
    message.delivery_status = 'failed'
    if (
      FAILED_STATUSES.has(persistence)
      || turnStatus === 'persist_failed'
      || message.persistence_status === 'failed'
    ) {
      message.persistence_status = 'failed'
    } else if (!message.persistence_status || message.persistence_status === 'pending') {
      // A generation/retrieval failure does not prove that an assistant row was
      // attempted, so keep persistence unknown rather than claiming save fail.
      message.persistence_status = 'unknown'
    }
    message.turn_status = 'failed'
    message.retryable = true
    return message
  }

  if (event.type === 'done') {
    const effectiveTurnStatus = turnStatus || normalizedStatus(message.turn_status)
    if (IN_PROGRESS_STATUSES.has(effectiveTurnStatus)) {
      // A replay response may end its SSE transport while the original logical
      // request is still running. Closing that transport is not answer delivery.
      message.delivery_status = delivery || 'pending'
      message.persistence_status = persistence || 'pending'
      message.turn_status = effectiveTurnStatus
      return message
    }

    message.delivery_status = delivery || 'delivered'
    if (persistence) message.persistence_status = persistence
    else if (effectiveTurnStatus === 'completed' || message.persistence_status === 'completed') {
      // A final durable turn_state normally precedes a protocol-level done
      // without persistence fields. Preserve that stronger acknowledgement.
      message.persistence_status = 'completed'
    } else {
      message.persistence_status = 'unknown'
    }
    message.turn_status = effectiveTurnStatus || 'completed'
    return message
  }

  if (event.type === 'turn_state' && turnStatus === 'completed') {
    message.delivery_status = delivery || 'delivered'
    message.persistence_status = persistence || 'completed'
  }

  if (delivery) message.delivery_status = delivery
  if (persistence) message.persistence_status = persistence
  if (turnStatus) message.turn_status = turnStatus
  return message
}

/**
 * A duplicate request can replay an accepted/generating turn without waiting
 * for the original worker. The UI may poll with the same request id, but must
 * not claim that an answer was delivered or launch a fresh logical request.
 */
export function isPendingTurnReplay(event) {
  return Boolean(
    event?.replayed === true
    && IN_PROGRESS_STATUSES.has(normalizedStatus(eventValue(event, 'turn_status'))),
  )
}

function hasTextContent(message) {
  return typeof message?.content === 'string' && Boolean(message.content.trim())
}

function hasClarificationPresentation(message) {
  return Boolean(
    message?.clarification
    && typeof message.clarification === 'object'
    && !Array.isArray(message.clarification),
  )
}

function messageRequestIdentity(message) {
  const value = message?.request_id || message?.server_request_id
  return typeof value === 'string' ? value.trim() : ''
}

/**
 * A terminal successful SSE may omit text deltas even though the assistant row
 * has already been committed. In that case the transcript endpoint is the
 * authority. Failed/stopped turns and persisted clarification cards are not
 * answer-content recovery candidates.
 */
export function shouldReloadCompletedEmptyAssistant(
  message,
  { sawDone = false, sawCompletedTurnState = false } = {},
) {
  if (!message || typeof message !== 'object' || message.role !== 'assistant') return false
  if (hasTextContent(message) || hasClarificationPresentation(message) || message.stopped === true) return false

  const delivery = normalizedStatus(message.delivery_status)
  const persistence = normalizedStatus(message.persistence_status)
  const turnStatus = normalizedStatus(message.turn_status ?? message.status)
  if (
    message.retryable === true
    || FAILED_STATUSES.has(delivery)
    || FAILED_STATUSES.has(persistence)
    || FAILED_STATUSES.has(turnStatus)
  ) return false

  const completionObserved = sawDone
    || sawCompletedTurnState
    || COMPLETED_STATUSES.has(turnStatus)
    || persistence === 'completed'
  return Boolean(
    completionObserved
    && (
      turnStatus === 'completed'
      || persistence === 'completed'
      || delivery === 'delivered'
    ),
  )
}

/**
 * Decide what an empty assistant card may render. A spinner is reserved for
 * the one request that is actively streaming; stale, terminal and failed
 * cards always resolve to readable, non-animated state text.
 */
export function emptyAssistantPresentation(
  message,
  { isStreaming = false, activeRequestId = '' } = {},
) {
  if (!message || typeof message !== 'object' || message.role !== 'assistant' || hasTextContent(message)) {
    return { kind: 'hidden', text: '' }
  }
  if (hasClarificationPresentation(message)) return { kind: 'hidden', text: '' }

  const delivery = normalizedStatus(message.delivery_status)
  const persistence = normalizedStatus(message.persistence_status)
  const turnStatus = normalizedStatus(message.turn_status ?? message.status)
  if (message.stopped === true || delivery === 'stopped') {
    return { kind: 'stopped', text: '已停止生成' }
  }
  if (
    message.retryable === true
    || FAILED_STATUSES.has(delivery)
    || FAILED_STATUSES.has(persistence)
    || FAILED_STATUSES.has(turnStatus)
  ) {
    const streamError = Array.isArray(message.stream_errors)
      ? message.stream_errors.find(value => typeof value === 'string' && value.trim())
      : ''
    return { kind: 'failed', text: streamError?.trim() || '本次回答未完成，请重试。' }
  }
  if (
    turnStatus === 'completed'
    || persistence === 'completed'
    || delivery === 'delivered'
  ) {
    return { kind: 'completed-empty', text: '回答已完成，但内容暂未加载。请重新打开当前会话。' }
  }

  const normalizedActiveRequestId = typeof activeRequestId === 'string' ? activeRequestId.trim() : ''
  const active = Boolean(
    isStreaming
    && normalizedActiveRequestId
    && messageRequestIdentity(message) === normalizedActiveRequestId,
  )
  if (active) return { kind: 'thinking', text: '思考中...' }
  if (IN_PROGRESS_STATUSES.has(turnStatus) || ['streaming', 'pending'].includes(delivery)) {
    return { kind: 'inactive', text: '回答仍在处理中，请稍后获取结果。' }
  }
  return { kind: 'inactive', text: '该回答暂无可显示内容。' }
}

/**
 * Prefer structured lifecycle fields. The exact legacy message is retained as
 * a compatibility contract for deployments that predate persistence_status.
 */
export function isPersistenceFailureEvent(event) {
  if (!event || typeof event !== 'object') return false
  const persistence = normalizedStatus(eventValue(event, 'persistence_status'))
  const turnStatus = normalizedStatus(eventValue(event, 'turn_status'))
  if (PERSISTENCE_FAILURE_STATUSES.has(persistence) || turnStatus === 'persist_failed' || turnStatus === 'save_failed') {
    return true
  }
  return event.type === 'error'
    && typeof event.message === 'string'
    && event.message.trim() === LEGACY_PERSISTENCE_FAILURE_MESSAGE
}

function identifier(value) {
  return typeof value === 'string' ? value.trim() : ''
}

/**
 * An error event may re-open a picker only when it explicitly says the same
 * pending route state is still active.  Absence is uncertainty, not consent;
 * callers must verify through the history endpoint instead.
 */
export function eventConfirmsPendingClarification(event, clarification) {
  if (!event || !clarification) return false
  const recovery = event.clarification_recovery && typeof event.clarification_recovery === 'object'
    ? event.clarification_recovery
    : {}
  const retained = event.pending_route_retained === true
    || event.clarification_retryable === true
    || recovery.active === true
  if (!retained) return false

  const eventStateId = identifier(
    event.pending_state_id || recovery.pending_state_id,
  )
  const eventMessageId = identifier(
    event.clarification_message_id || recovery.clarification_message_id,
  )
  const eventRevision = event.route_state_revision ?? recovery.route_state_revision
  if (eventStateId && eventStateId !== identifier(clarification.pending_state_id)) return false
  if (eventMessageId && eventMessageId !== identifier(clarification.clarification_message_id)) return false
  if (
    eventRevision !== undefined
    && eventRevision !== null
    && Number(eventRevision) !== Number(clarification.route_state_revision)
  ) return false
  return Boolean(
    clarification.status === 'active'
    && clarification.persisted
    && clarification.pending_state_id
    && clarification.clarification_message_id,
  )
}
