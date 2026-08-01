import { normalizeTraceId } from './chatRequest.js'
import { restoreHistoryMessageClarification } from './chatClarification.js'

const EVIDENCE_STATUSES = new Set([
  'skipped', 'hit', 'partial', 'version_mismatch', 'needs_clarification',
  'no_hit', 'unverified', 'error',
])

function objectValue(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null
}

function firstDefined(...values) {
  return values.find(value => value !== undefined && value !== null)
}

function normalizedStatus(value) {
  if (typeof value !== 'string') return ''
  const status = value.trim().toLowerCase()
  return EVIDENCE_STATUSES.has(status) ? status : ''
}

function normalizedBoolean(value) {
  return typeof value === 'boolean' ? value : null
}

function candidateSnapshot(message) {
  const source = objectValue(message)
  if (!source) return null
  return [
    source.search_snapshot,
    source.searchSnapshot,
    source.retrieval_snapshot,
    source.retrievalSnapshot,
    source.search_state,
    source.search_meta,
    source.searchMeta,
  ].map(objectValue).find(Boolean) || null
}

function normalizedResults(snapshot, message) {
  const raw = firstDefined(
    snapshot?.results,
    snapshot?.candidates,
    snapshot?.displayed_results,
    message?.retrieval_results,
  )
  return Array.isArray(raw) ? raw.filter(item => item && typeof item === 'object') : []
}

/**
 * Normalize the additive history contract.  Older servers only return
 * `sources`; newer servers may call the snapshot `search_snapshot` or
 * `retrieval_snapshot`.  Unknown fields are retained for forward-compatible
 * diagnostics, while status and booleans are fail-closed when absent.
 */
export function searchSnapshotFromHistoryMessage(message) {
  const source = objectValue(message)
  if (!source || source.role !== 'assistant') return null
  const raw = candidateSnapshot(source)
  const counters = objectValue(raw?.counters) || {}
  const hasExplicitState = Boolean(
    raw
    || source.evidence_status
    || source.retrieval_executed !== undefined
    || source.trace_id
    || source.traceId
    || source.delivery_status
    || source.persistence_status,
  )
  if (!hasExplicitState) return null

  const status = normalizedStatus(firstDefined(
    source.evidence_status,
    raw?.evidence_status,
    counters.evidence_status,
    raw?.status,
  ))
  const retrievalExecuted = normalizedBoolean(firstDefined(
    source.retrieval_executed,
    raw?.retrieval_executed,
    counters.retrieval_executed,
    raw?.retrievalExecuted,
  ))
  const traceId = normalizeTraceId(firstDefined(
    source.trace_id,
    source.traceId,
    raw?.trace_id,
    counters.trace_id,
    raw?.traceId,
  ))
  const results = normalizedResults(raw, source)
  const hasMessageClarification = Object.prototype.hasOwnProperty.call(source, 'clarification')
  const clarification = hasMessageClarification
    ? objectValue(source.clarification)
    : objectValue(raw?.clarification)
  const snapshot = {
    ...(raw || {}),
    ...counters,
    results,
    total: firstDefined(raw?.total, counters.total, counters.displayed_result_count, results.length),
    evidence_status: status || undefined,
    retrieval_executed: retrievalExecuted,
    trace_id: traceId || undefined,
    clarification: clarification || undefined,
    // The panel must never pretend that a historical snapshot is a live SSE
    // process.  SearchResultPanel uses this marker to render a static summary.
    historical: true,
    snapshot_available: true,
  }
  return snapshot
}

export function restoreHistoryMessageState(message) {
  const source = objectValue(message)
  if (!source) return source
  // Validate the separately returned pending clarification before it is merged
  // into the static search snapshot. This keeps stale/malformed history
  // fail-closed while allowing choices=[] refinement state to reach the panel.
  const strictSource = restoreHistoryMessageClarification({ ...source })
  const snapshot = searchSnapshotFromHistoryMessage(strictSource)
  const traceId = normalizeTraceId(firstDefined(
    strictSource.trace_id,
    strictSource.traceId,
    snapshot?.trace_id,
  ))
  const status = normalizedStatus(firstDefined(
    strictSource.evidence_status,
    snapshot?.evidence_status,
  ))
  const retrievalExecuted = normalizedBoolean(firstDefined(
    strictSource.retrieval_executed,
    snapshot?.retrieval_executed,
  ))
  const restored = {
    ...strictSource,
    trace_id: traceId || null,
    evidence_status: status || undefined,
    retrieval_executed: retrievalExecuted,
    delivery_status: firstDefined(strictSource.delivery_status, strictSource.deliveryState, null),
    persistence_status: firstDefined(strictSource.persistence_status, strictSource.persistenceState, null),
    search_snapshot: snapshot,
    search_meta: snapshot
      ? { ...(objectValue(strictSource.search_meta) || {}), ...snapshot, trace_id: traceId || undefined }
      : (objectValue(strictSource.search_meta) || strictSource.search_meta || null),
  }
  return restored
}

function restoredMessageIdentity(message) {
  const requestId = typeof message?.request_id === 'string' ? message.request_id.trim() : ''
  const role = typeof message?.role === 'string' ? message.role.trim() : ''
  if (requestId && role) return `request:${requestId}:${role}`
  const turnId = typeof message?.turn_id === 'string' ? message.turn_id.trim() : ''
  if (turnId && role) return `turn:${turnId}:${role}`
  return message?.id == null ? '' : `message:${String(message.id)}`
}

function messageAuthorityScore(message) {
  let score = 0
  if (typeof message?.content === 'string' && message.content.trim()) score += 8
  if (message?.clarification && typeof message.clarification === 'object') score += 4
  if (String(message?.turn_status || message?.status || '').trim().toLowerCase() === 'completed') score += 2
  if (String(message?.persistence_status || '').trim().toLowerCase() === 'completed') score += 1
  return score
}

/**
 * Restore one authoritative transcript and collapse duplicate rows for the
 * same durable request/role. The best persisted presentation replaces a blank
 * placeholder without changing that logical turn's original list position.
 */
export function restoreConversationMessages(rows) {
  if (!Array.isArray(rows)) return []
  const restored = []
  const indexByIdentity = new Map()
  for (const row of rows) {
    const message = restoreHistoryMessageState(row)
    if (!message || typeof message !== 'object') continue
    const identity = restoredMessageIdentity(message)
    if (!identity || !indexByIdentity.has(identity)) {
      if (identity) indexByIdentity.set(identity, restored.length)
      restored.push(message)
      continue
    }
    const index = indexByIdentity.get(identity)
    if (messageAuthorityScore(message) > messageAuthorityScore(restored[index])) {
      restored[index] = message
    }
  }
  return restored
}

export function hasSearchSnapshot(message) {
  return Boolean(searchSnapshotFromHistoryMessage(message))
}

export function searchSnapshotFromEvent(data, fallback = {}) {
  const event = objectValue(data) || {}
  const meta = objectValue(event.search_meta) || objectValue(event.meta) || {}
  const raw = {
    ...meta,
    ...event,
    results: Array.isArray(event.results) ? event.results : (Array.isArray(meta.results) ? meta.results : []),
  }
  const source = objectValue(fallback) || {}
  return {
    ...source,
    ...raw,
    search_meta: meta,
    historical: false,
    snapshot_available: true,
  }
}
