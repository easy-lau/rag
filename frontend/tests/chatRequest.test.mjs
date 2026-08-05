import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  captureAssistantPresentationForRetry,
  chatRequestHttpFailure,
  coalesceLogicalTurnMessages,
  createClientRequestId,
  normalizeClientRequestId,
  normalizeTraceId,
  normalizeTurnId,
  recoverableRetryRequestId,
  resetAssistantForLogicalRetry,
  restoreAssistantPresentationAfterUnconfirmedRetry,
  retryWithNewRequestIdForHttpStatus,
  streamEventConfirmsAssistantPresentation,
  traceIdFromResponse,
} from '../src/utils/chatRequest.js'

test('客户端 request id 为 UUID 且可在重试时原样复用', () => {
  const requestId = createClientRequestId()
  assert.match(requestId, /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
  assert.equal(normalizeClientRequestId(requestId), requestId)
})

test('请求和 Trace 标识拒绝空白、控制内容及过短值', () => {
  assert.equal(normalizeClientRequestId('question with spaces'), '')
  assert.equal(normalizeTraceId('x'), '')
  assert.equal(normalizeTraceId('trace-safe_001'), 'trace-safe_001')
  assert.equal(normalizeTurnId('3f49a3d0-6b35-4df0-8b5d-87c3f84d2ef6'), '3f49a3d0-6b35-4df0-8b5d-87c3f84d2ef6')
  assert.equal(normalizeTurnId('not-a-turn-id'), '')
})

test('响应头 Trace 可安全绑定，聊天 API 同时发送 request_id body/header', () => {
  assert.equal(traceIdFromResponse({
    headers: { get: name => name === 'X-RAG-Trace-ID' ? 'trace-header-001' : null },
  }), 'trace-header-001')

  const apiSource = readFileSync(new URL('../src/api/chat.js', import.meta.url), 'utf8')
  const storeSource = readFileSync(new URL('../src/stores/chat.js', import.meta.url), 'utf8')
  const viewSource = readFileSync(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')
  assert.ok(apiSource.includes("'X-Client-Request-ID': requestId"))
  assert.ok(apiSource.includes('request_id: requestId'))
  assert.ok(viewSource.includes('recoverableRetryRequestId(answerMessage, targetUser)'))
  assert.ok(storeSource.includes('responseHeader(res, \'X-RAG-Turn-ID\')'))
  assert.ok(storeSource.includes('turn_id: turnId || undefined'))
  assert.ok(storeSource.includes('pending_route_revision: clarificationIdentity?.route_state_revision'))
  assert.ok(storeSource.includes('pending_state_id: clarificationIdentity?.pending_state_id'))
})

test('保存或传输不确定时复用 request id，主动重新生成使用新逻辑请求', () => {
  const user = { request_id: 'user-request-001' }
  assert.equal(recoverableRetryRequestId({
    request_id: 'answer-request-001',
    failure_reason: 'stream_incomplete',
  }, user), 'answer-request-001')
  assert.equal(recoverableRetryRequestId({
    server_request_id: 'server-request-001',
    persistence_status: 'failed',
  }, user), 'server-request-001')
  assert.equal(recoverableRetryRequestId({
    request_id: 'recoverable-persist-request-001',
    turn_status: 'failed',
    persistence_status: 'failed',
    error_code: 'assistant_persistence_failed',
    same_request_recoverable: true,
    failure_reason: 'persistence_failed',
  }, user), 'recoverable-persist-request-001')
  assert.equal(recoverableRetryRequestId({
    request_id: 'pending-request-001',
    failure_reason: 'turn_in_progress',
  }, user), 'pending-request-001')
  assert.equal(recoverableRetryRequestId({
    request_id: 'unrecoverable-request-001',
    persistence_status: 'failed',
    same_request_recoverable: false,
    retry_with_new_request_id: true,
  }, user), '')
  assert.equal(recoverableRetryRequestId({
    request_id: 'historical-failed-request-001',
    turn_status: 'failed',
    persistence_status: 'failed',
    error_code: 'generated_payload_not_persisted',
  }, user), '')

  assert.equal(recoverableRetryRequestId({
    request_id: 'completed-request-001',
    turn_status: 'completed',
    persistence_status: 'completed',
  }, user), '')
  assert.equal(recoverableRetryRequestId({
    request_id: 'cancelled-request-001',
    failure_reason: 'stream_aborted',
  }, user), '')
  assert.equal(recoverableRetryRequestId({
    request_id: 'model-failed-request-001',
    failure_reason: 'server_error',
  }, user), '')
})

test('同一逻辑请求的重复本地气泡合并到第一组且连续恢复不增加数量', () => {
  const requestId = 'pending-request-duplicate-001'
  const firstUser = { id: 'user-first', role: 'user', request_id: requestId, content: '查询差旅标准' }
  const firstAssistant = {
    id: 'assistant-first',
    role: 'assistant',
    request_id: requestId,
    content: '同一请求仍在处理中',
    failure_reason: 'turn_in_progress',
  }
  let messages = [
    { id: 'older', role: 'assistant', content: '更早的消息' },
    firstUser,
    firstAssistant,
    { id: 'user-duplicate', role: 'user', request_id: requestId, content: '查询差旅标准' },
    { id: 'assistant-duplicate', role: 'assistant', request_id: requestId, content: '' },
    { id: 'newer', role: 'user', content: '无关消息' },
  ]

  for (let replay = 0; replay < 3; replay += 1) {
    const logicalTurn = coalesceLogicalTurnMessages(messages, requestId)
    messages = logicalTurn.messages
    assert.equal(logicalTurn.userMessage, firstUser)
    assert.equal(logicalTurn.assistantMessage, firstAssistant)
    resetAssistantForLogicalRetry(logicalTurn.assistantMessage, requestId)
  }

  assert.equal(messages.length, 4)
  assert.deepEqual(messages.map(message => message.id), [
    'older',
    'user-first',
    'assistant-first',
    'newer',
  ])
})

test('恢复请求在首个 SSE 事件前断开时还原原位回答和证据', () => {
  const assistant = {
    id: 'assistant-persist-failed',
    role: 'assistant',
    request_id: 'persist-request-001',
    content: '普通员工餐补为 100 元/天。',
    sources: [{ id: 'travel-policy' }],
    answer_provenance: 'general_model',
    stream_errors: ['回答保存失败'],
    search_snapshot: { evidence_status: 'no_hit', answer_provenance: 'general_model' },
    delivery_status: 'failed',
    persistence_status: 'failed',
    failure_reason: 'persistence_failed',
  }
  const snapshot = captureAssistantPresentationForRetry(assistant)

  resetAssistantForLogicalRetry(assistant, 'persist-request-001')
  assert.equal(assistant.content, '')
  assert.deepEqual(assistant.sources, [])
  assert.equal(assistant.answer_provenance, null)
  assert.equal(assistant.delivery_status, 'streaming')

  assert.equal(restoreAssistantPresentationAfterUnconfirmedRetry(assistant, snapshot), true)
  assert.equal(assistant.content, '普通员工餐补为 100 元/天。')
  assert.deepEqual(assistant.sources, [{ id: 'travel-policy' }])
  assert.equal(assistant.answer_provenance, 'general_model')
  assert.deepEqual(assistant.search_snapshot, {
    evidence_status: 'no_hit',
    answer_provenance: 'general_model',
  })
  // 传输状态不属于展示快照，调用方可继续标记本次恢复失败。
  assert.equal(assistant.delivery_status, 'streaming')
})

test('仅握手或进度 SSE 后断流仍保留旧回答，权威展示事件才放弃快照', () => {
  for (const progressEvent of [
    { type: 'conversation_started', conversation_id: 'conversation-1' },
    { type: 'intent', decision: { route: 'grounded_qa' } },
    { type: 'search_step', step: 'retrieve', status: 'running' },
    { type: 'turn_state', status: 'accepted' },
    { type: 'turn_state', status: 'generating' },
    { type: 'usage', total_tokens: 12 },
  ]) {
    assert.equal(streamEventConfirmsAssistantPresentation(progressEvent), false)
  }

  for (const authoritativeEvent of [
    { type: 'text_delta', content: '新回答' },
    { type: 'search_results', results: [] },
    { type: 'turn_state', status: 'completed' },
    { type: 'turn_state', status: 'failed' },
    { type: 'error', message: '生成失败' },
    { type: 'evidence_clarification_ack', persisted: true },
    { type: 'done', status: 'generating', replayed: true },
  ]) {
    assert.equal(streamEventConfirmsAssistantPresentation(authoritativeEvent), true)
  }

  const assistant = {
    role: 'assistant',
    request_id: 'handshake-only-request',
    content: '原回答仍可查看',
    sources: [{ id: 'old-source' }],
  }
  const snapshot = captureAssistantPresentationForRetry(assistant)
  resetAssistantForLogicalRetry(assistant, 'handshake-only-request')
  // conversation_started/accepted 均不会使调用方丢弃 snapshot。
  assert.equal(restoreAssistantPresentationAfterUnconfirmedRetry(assistant, snapshot), true)
  assert.equal(assistant.content, '原回答仍可查看')
  assert.deepEqual(assistant.sources, [{ id: 'old-source' }])
})

test('HTTP 重试优先采用服务端契约并兼容旧 409/503', () => {
  const conflict = chatRequestHttpFailure(409, {
    detail: {
      message: '待处理的澄清上下文已变化',
      error_code: 'pending_route_conflict',
    },
  })
  assert.equal(conflict.status, 409)
  assert.equal(conflict.publicMessage, '待处理的澄清上下文已变化')
  assert.deepEqual(conflict.detail, {
    message: '待处理的澄清上下文已变化',
    error_code: 'pending_route_conflict',
  })
  assert.equal(conflict.errorCode, 'pending_route_conflict')
  assert.equal(conflict.failureReason, 'request_conflict')
  assert.equal(conflict.retryWithNewRequestId, true)
  assert.equal(conflict.sameRequestRecoverable, false)
  assert.equal(retryWithNewRequestIdForHttpStatus(409), true)
  assert.equal(recoverableRetryRequestId({
    request_id: 'conflict-request-001',
    failure_reason: conflict.failureReason,
    retry_with_new_request_id: conflict.retryWithNewRequestId,
  }), '')

  const unavailable = chatRequestHttpFailure(503, {
    detail: '请求已接收但暂时无法保存，请使用相同 request_id 重试',
  })
  assert.equal(unavailable.failureReason, 'request_failed')
  assert.equal(unavailable.retryWithNewRequestId, false)
  assert.equal(unavailable.sameRequestRecoverable, true)
  assert.equal(recoverableRetryRequestId({
    request_id: 'unavailable-request-001',
    failure_reason: unavailable.failureReason,
    same_request_recoverable: unavailable.sameRequestRecoverable,
  }), 'unavailable-request-001')

  const reservationUncertain = chatRequestHttpFailure(503, {
    detail: {
      message: '请求已接收但暂时无法保存，请使用相同 request_id 重试',
      error_code: 'turn_reservation_persistence_uncertain',
      same_request_recoverable: true,
      retry_with_new_request_id: false,
    },
  })
  assert.equal(reservationUncertain.errorCode, 'turn_reservation_persistence_uncertain')
  assert.equal(reservationUncertain.retryWithNewRequestId, false)
  assert.equal(reservationUncertain.sameRequestRecoverable, true)

  const terminalRejection = chatRequestHttpFailure(503, {
    detail: {
      message: '请求执行合同校验失败，请重新发送',
      error_code: 'runner_contract_rejected',
      same_request_recoverable: false,
      retry_with_new_request_id: true,
    },
  })
  assert.equal(terminalRejection.errorCode, 'runner_contract_rejected')
  assert.equal(terminalRejection.retryWithNewRequestId, true)
  assert.equal(terminalRejection.sameRequestRecoverable, false)
  assert.equal(recoverableRetryRequestId({
    request_id: 'rejected-request-001',
    failure_reason: terminalRejection.failureReason,
    retry_with_new_request_id: terminalRejection.retryWithNewRequestId,
    same_request_recoverable: terminalRejection.sameRequestRecoverable,
  }), '')
})
