import assert from 'node:assert/strict'
import test from 'node:test'

import {
  activateClarification,
  applyClarificationLifecycleEvent,
  attachClarification,
  clarificationFromSearchEvent,
  invalidateClarification,
  isClarificationActive,
  isClarificationSubmittable,
  lockMessageClarification,
  markClarificationSubmitted,
  normalizeClarification,
  restoreClarificationSubmissionForRetry,
  restoreHistoryMessageClarification,
} from '../src/utils/chatClarification.js'

const STATE_SCHEMA = 'rag_clarification_state.v1'

function choice(index, overrides = {}) {
  return {
    key: `scope${index}`,
    label: `平台A 版本 ${index}`,
    products: ['平台A'],
    versions: [String(index)],
    projects: [],
    filenames: [`平台A-${index}.md`],
    ...overrides,
  }
}

function proposed(overrides = {}) {
  return {
    type: 'clarification_state',
    schema_version: STATE_SCHEMA,
    status: 'proposed',
    persisted: false,
    needs_clarification: true,
    adapter: 'evidence',
    dimension: 'product_version',
    reason_code: 'multiple_authorized_versions',
    selection_mode: 'choice',
    choices: [choice(1), choice(2)],
    pending_state_id: null,
    clarification_message_id: null,
    route_state_revision: null,
    ...overrides,
  }
}

function active(overrides = {}) {
  return proposed({
    status: 'active',
    persisted: true,
    pending_state_id: 'pending-state-1',
    clarification_message_id: 'assistant-message-1',
    route_state_revision: 3,
    conversation_id: 'conversation-1',
    selected_kb_ids_snapshot: ['kb-1'],
    ...overrides,
  })
}

test('唯一澄清协议归一化候选并拒绝非当前 schema', () => {
  const normalized = normalizeClarification(proposed())
  assert.equal(normalized.schema_version, STATE_SCHEMA)
  assert.equal(normalized.adapter, 'evidence')
  assert.equal(normalized.selection_mode, 'choice')
  assert.deepEqual(normalized.choices.map(item => item.reply), ['scope1', 'scope2'])
  assert.equal(
    normalizeClarification({ ...proposed(), schema_version: 'unsupported.v1' }),
    null,
  )
  assert.equal(
    clarificationFromSearchEvent({
      clarification: { ...proposed(), schema_version: 'unsupported.v1' },
    }),
    null,
  )
})

test('proposed 只展示结构化候选，active 持久化事件才允许提交', () => {
  const message = { role: 'assistant', content: '', sources: [{ id: 'unsafe' }] }
  const proposedState = applyClarificationLifecycleEvent(message, proposed())
  assert.ok(proposedState)
  assert.equal(isClarificationActive(proposedState), false)
  assert.equal(isClarificationSubmittable(proposedState), false)

  const activeState = applyClarificationLifecycleEvent(message, active())
  assert.ok(activeState)
  assert.equal(isClarificationActive(activeState), true)
  assert.equal(isClarificationSubmittable(activeState), true)
  assert.deepEqual(message.clarification.selected_kb_ids_snapshot, ['kb-1'])
})

test('active 事件必须与 proposed 身份一致，乱序状态不能解锁', () => {
  const message = { role: 'assistant' }
  attachClarification(message, proposed({ pending_state_id: 'pending-state-1' }))
  assert.equal(activateClarification(message, active({ pending_state_id: 'other' })), null)
  assert.equal(isClarificationActive(message.clarification), false)
  assert.ok(activateClarification(message, active()))
  assert.equal(isClarificationActive(message.clarification), true)
})

test('semantic 自由补充不要求知识库快照，evidence 选择仍必须受范围约束', () => {
  const semanticMessage = { role: 'assistant' }
  applyClarificationLifecycleEvent(semanticMessage, proposed({
    adapter: 'semantic',
    dimension: 'query',
    reason_code: 'missing_context',
    selection_mode: 'refine',
    choices: [],
  }))
  assert.ok(activateClarification(semanticMessage, active({
    adapter: 'semantic',
    dimension: 'query',
    reason_code: 'missing_context',
    selection_mode: 'refine',
    choices: [],
    selected_kb_ids_snapshot: [],
  })))

  const evidenceMessage = { role: 'assistant' }
  applyClarificationLifecycleEvent(evidenceMessage, proposed())
  assert.equal(activateClarification(
    evidenceMessage,
    active({ selected_kb_ids_snapshot: [] }),
  ), null)
})

test('澄清状态锁定回答证据，模型无文字时结构化候选仍可展示', () => {
  const message = {
    role: 'assistant',
    content: '',
    sources: [{ id: 'candidate' }],
    search_meta: { evidence_status: 'hit', hit_count: 1 },
  }
  attachClarification(message, active())
  const clarification = lockMessageClarification(message)
  assert.ok(clarification)
  assert.deepEqual(message.sources, [])
  assert.equal(message.evidence_status, 'needs_clarification')
  assert.equal(message.search_meta.hit_count, 0)
  assert.equal(message.search_meta.clarification.choices.length, 2)
})

test('点击候选与自然语言补充共用同一提交生命周期', () => {
  const choiceMessage = { role: 'assistant' }
  applyClarificationLifecycleEvent(choiceMessage, active())
  assert.equal(markClarificationSubmitted(choiceMessage, 'scope2', { requestId: 'request-1' }), true)
  assert.equal(choiceMessage.clarification.submitted, true)
  assert.equal(choiceMessage.clarification.submission_request_id, 'request-1')
  assert.ok(restoreClarificationSubmissionForRetry(choiceMessage, 'request-1'))
  assert.equal(choiceMessage.clarification.retryable, true)

  const refineMessage = { role: 'assistant' }
  applyClarificationLifecycleEvent(refineMessage, active({
    adapter: 'semantic',
    dimension: 'query',
    reason_code: 'missing_context',
    selection_mode: 'refine',
    choices: [],
  }))
  assert.equal(
    markClarificationSubmitted(
      refineMessage,
      '我指的是平台A 版本 7',
      { allowFreeText: true, requestId: 'request-2' },
    ),
    true,
  )
})

test('错误、无 active 的 done 和主动失效都不会留下可点击候选', () => {
  for (const event of [
    { type: 'error' },
    { type: 'done' },
  ]) {
    const message = { role: 'assistant' }
    attachClarification(message, proposed())
    applyClarificationLifecycleEvent(message, event)
    assert.equal(isClarificationActive(message.clarification), false)
    assert.equal(message.clarification.invalidated, true)
  }

  const message = { role: 'assistant' }
  attachClarification(message, proposed())
  invalidateClarification(message, 'stream_aborted')
  assert.equal(message.clarification.invalid_reason, 'stream_aborted')
})

test('历史恢复只激活当前 assistant_message_id 对应的持久化状态', () => {
  const restored = restoreHistoryMessageClarification({
    id: 'assistant-message-1',
    role: 'assistant',
    content: '',
    sources: [{ id: 'old' }],
    clarification: active(),
  })
  assert.equal(isClarificationActive(restored.clarification), true)
  assert.deepEqual(restored.sources, [])

  const stale = restoreHistoryMessageClarification({
    id: 'another-message',
    role: 'assistant',
    clarification: active(),
  })
  assert.equal(stale.clarification, null)
})
