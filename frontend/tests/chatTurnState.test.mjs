import test from 'node:test'
import assert from 'node:assert/strict'

import {
  applyTurnLifecycleState,
  emptyAssistantPresentation,
  eventConfirmsPendingClarification,
  isPendingTurnReplay,
  isPersistenceFailureEvent,
  shouldReloadCompletedEmptyAssistant,
} from '../src/utils/chatTurnState.js'

test('SSE error 后的乱序 done 不能覆盖失败和未知持久化状态', () => {
  const message = { delivery_status: 'streaming', persistence_status: 'pending' }
  applyTurnLifecycleState(message, { type: 'error', status: 'failed' })
  applyTurnLifecycleState(message, {
    type: 'done',
    status: 'completed',
    delivery_status: 'delivered',
    persistence_status: 'completed',
  })

  assert.equal(message.delivery_status, 'failed')
  assert.equal(message.persistence_status, 'unknown')
  assert.equal(message.turn_status, 'failed')
  assert.equal(message.retryable, true)
})

test('persist_failed turn 即使随后 done 也保持保存失败', () => {
  const message = { delivery_status: 'streaming', persistence_status: 'pending' }
  applyTurnLifecycleState(message, { type: 'turn_state', status: 'persist_failed' })
  applyTurnLifecycleState(message, { type: 'done', status: 'persist_failed' })

  assert.equal(message.delivery_status, 'failed')
  assert.equal(message.persistence_status, 'failed')
  assert.equal(message.turn_status, 'failed')
})

test('正常 completed turn 恢复明确的交付与持久化状态', () => {
  const message = { delivery_status: 'streaming', persistence_status: 'pending' }
  applyTurnLifecycleState(message, { type: 'turn_state', status: 'completed' })
  applyTurnLifecycleState(message, { type: 'done', status: 'completed' })

  assert.equal(message.delivery_status, 'delivered')
  assert.equal(message.persistence_status, 'completed')
  assert.equal(message.turn_status, 'completed')
})

test('completed/done 正常结束但正文为空时要求从权威历史重载', () => {
  const message = {
    role: 'assistant',
    content: '   ',
    delivery_status: 'streaming',
    persistence_status: 'pending',
  }
  applyTurnLifecycleState(message, { type: 'done', status: 'completed' })
  assert.equal(shouldReloadCompletedEmptyAssistant(message, { sawDone: true }), true)
  assert.equal(shouldReloadCompletedEmptyAssistant({ ...message, content: '已收到回答' }, { sawDone: true }), false)
  assert.equal(shouldReloadCompletedEmptyAssistant({ ...message, retryable: true }, { sawDone: true }), false)
  assert.equal(shouldReloadCompletedEmptyAssistant({ ...message, stopped: true }, { sawDone: true }), false)
  assert.equal(shouldReloadCompletedEmptyAssistant({ ...message, clarification: { question: '请选择范围' } }, { sawDone: true }), false)
})

test('空回答只有当前活跃 request 可显示 spinner，终态和非活跃态都显示静态说明', () => {
  const active = {
    role: 'assistant',
    content: '',
    request_id: 'request-active',
    delivery_status: 'streaming',
    turn_status: 'generating',
  }
  assert.equal(emptyAssistantPresentation(active, {
    isStreaming: true,
    activeRequestId: 'request-active',
  }).kind, 'thinking')
  assert.equal(emptyAssistantPresentation(active, {
    isStreaming: true,
    activeRequestId: 'request-other',
  }).kind, 'inactive')

  assert.equal(emptyAssistantPresentation({
    ...active,
    delivery_status: 'failed',
    turn_status: 'failed',
    retryable: true,
    stream_errors: ['网络连接中断，请重试'],
  }, {
    isStreaming: true,
    activeRequestId: 'request-active',
  }).kind, 'failed')
  assert.equal(emptyAssistantPresentation({
    ...active,
    delivery_status: 'delivered',
    persistence_status: 'completed',
    turn_status: 'completed',
  }, {
    isStreaming: true,
    activeRequestId: 'request-active',
  }).kind, 'completed-empty')
  assert.equal(emptyAssistantPresentation({ ...active, stopped: true }).kind, 'stopped')
  assert.equal(emptyAssistantPresentation({
    ...active,
    clarification: { question: '请选择制度版本' },
  }).kind, 'hidden')
})

test('无状态 done 不会把前一条 completed turn_state 降级为未知', () => {
  const message = { delivery_status: 'streaming', persistence_status: 'pending' }
  applyTurnLifecycleState(message, { type: 'turn_state', status: 'completed' })
  applyTurnLifecycleState(message, { type: 'done' })

  assert.equal(message.delivery_status, 'delivered')
  assert.equal(message.persistence_status, 'completed')
  assert.equal(message.turn_status, 'completed')
})

test('replayed accepted/generating 结束传输时仍是处理中，不冒充已交付', () => {
  for (const status of ['accepted', 'generating']) {
    const message = { delivery_status: 'streaming', persistence_status: 'pending' }
    const event = { type: 'done', status, replayed: true }
    applyTurnLifecycleState(message, event)

    assert.equal(isPendingTurnReplay(event), true)
    assert.equal(message.replayed, true)
    assert.equal(message.delivery_status, 'pending')
    assert.equal(message.persistence_status, 'pending')
    assert.equal(message.turn_status, status)
  }
  assert.equal(isPendingTurnReplay({ type: 'done', status: 'completed', replayed: true }), false)
})

test('保存失败优先识别结构化状态，并兼容旧服务端精确错误契约', () => {
  assert.equal(isPersistenceFailureEvent({
    type: 'error',
    turn_status: 'persist_failed',
  }), true)
  assert.equal(isPersistenceFailureEvent({
    type: 'error',
    persistence_status: 'failed',
  }), true)
  assert.equal(isPersistenceFailureEvent({
    type: 'error',
    message: '回答已生成，但保存失败，请重试',
  }), true)
  assert.equal(isPersistenceFailureEvent({
    type: 'error',
    message: '模型服务暂时不可用',
  }), false)
  assert.equal(isPersistenceFailureEvent({
    type: 'error',
    persistence_status: 'cancelled',
  }), false)
})

test('生命周期保留服务端判定的 request id 恢复边界', () => {
  const message = {}
  applyTurnLifecycleState(message, {
    type: 'error',
    persistence_status: 'failed',
    same_request_recoverable: false,
    retry_with_new_request_id: true,
  })
  assert.equal(message.same_request_recoverable, false)
  assert.equal(message.retry_with_new_request_id, true)
})

test('只有明确确认同一 pending revision 的事件才允许恢复澄清', () => {
  const clarification = {
    status: 'active',
    persisted: true,
    pending_state_id: 'pending-travel',
    clarification_message_id: 'assistant-travel',
    route_state_revision: 7,
  }

  assert.equal(eventConfirmsPendingClarification({ type: 'error' }, clarification), false)
  assert.equal(eventConfirmsPendingClarification({
    type: 'error',
    pending_route_retained: true,
    pending_state_id: 'other-pending',
  }, clarification), false)
  assert.equal(eventConfirmsPendingClarification({
    type: 'error',
    pending_route_retained: true,
    pending_state_id: 'pending-travel',
    clarification_message_id: 'assistant-travel',
    route_state_revision: 7,
  }, clarification), true)
})
