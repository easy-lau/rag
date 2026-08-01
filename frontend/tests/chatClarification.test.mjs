import test from 'node:test'
import assert from 'node:assert/strict'

import {
  acknowledgeEvidenceClarification,
  applyClarificationLifecycleEvent,
  attachEvidenceClarification,
  clarificationFromSearchEvent,
  invalidateEvidenceClarification,
  isClarificationActive,
  isClarificationSubmittable,
  lockMessageClarificationEvidence,
  markClarificationSubmitted,
  MAX_CLARIFICATION_CHOICES,
  normalizeEvidenceClarification,
  restoreHistoryMessageClarification,
} from '../src/utils/chatClarification.js'

function persistedAck(overrides = {}) {
  return {
    type: 'evidence_clarification_ack',
    schema_version: 'rag_evidence_clarification_ack.v1',
    persisted: true,
    pending_state_id: 'pending-1',
    clarification_message_id: 'assistant-1',
    route_state_revision: 3,
    conversation_id: 'conversation-1',
    ...overrides,
  }
}

function clarificationEvent(overrides = {}) {
  return {
    type: 'evidence_clarification',
    schema_version: 'rag_evidence_clarification.v1',
    needs_clarification: true,
    dimension: 'version',
    question: '请选择适用版本',
    choices: [{ key: 'c1', label: '云枢 8.6' }],
    ...overrides,
  }
}

test('结构化澄清保留有界选项并优先使用服务端 choice key', () => {
  const clarification = normalizeEvidenceClarification({
    schema_version: 'rag_evidence_clarification.v1',
    needs_clarification: true,
    dimension: 'document',
    question: '需要哪一篇？',
    choices: [
      { key: 'c1', label: '员工请假管理办法.docx' },
      { key: 'c2', label: '公司出差管理标准.docx' },
    ],
  })

  assert.equal(clarification.dimension, 'document')
  assert.deepEqual(
    clarification.choices.map(choice => ({ index: choice.index, label: choice.label, reply: choice.reply })),
    [
      { index: 1, label: '员工请假管理办法.docx', reply: 'c1' },
      { index: 2, label: '公司出差管理标准.docx', reply: 'c2' },
    ],
  )
  assert.equal(clarification.requires_refinement, false)
})

test('实时澄清在持久化 ack 前不可提交，收到合法 ack 后才解锁', () => {
  const payload = clarificationFromSearchEvent({
    type: 'search_results',
    evidence_status: 'needs_clarification',
    search_meta: {
      clarification: {
        schema_version: 'rag_evidence_clarification.v1',
        dimension: 'version',
        choices: [{ key: 'c1', label: 'CloudPivot 6' }],
      },
    },
  })
  const message = { role: 'assistant', content: '请选择版本' }

  assert.ok(attachEvidenceClarification(message, payload))
  assert.equal(message.clarification.choices[0].reply, 'c1')
  assert.equal(message.clarification.acknowledged, false)
  assert.equal(isClarificationSubmittable(message.clarification), false)
  assert.equal(markClarificationSubmitted(message, 'c1'), false)

  assert.ok(acknowledgeEvidenceClarification(message, persistedAck()))
  assert.equal(message.clarification.acknowledged, true)
  assert.equal(message.clarification.persisted, true)
  assert.equal(isClarificationSubmittable(message.clarification), true)
  assert.equal(markClarificationSubmitted(message, '不存在的选项'), false)
  assert.equal(markClarificationSubmitted(message, 'c1'), true)
  assert.equal(markClarificationSubmitted(message, 'c1'), false)
  assert.equal(message.clarification.submitted, true)
  assert.equal(message.clarification.submitted_reply, 'c1')
})

test('search_results 内嵌澄清也必须使用精确协议版本', () => {
  assert.equal(clarificationFromSearchEvent({
    type: 'search_results',
    clarification: {
      schema_version: 'rag_evidence_clarification.v2',
      choices: [{ key: 'c1', label: '错误协议候选' }],
    },
  }), null)
})

test('事件序列 clarification -> ack -> done 保留可提交的持久化 picker', () => {
  const message = {
    id: 'temporary-client-id',
    role: 'assistant',
    sources: [{ id: 'stale', evidence_role: 'direct' }],
  }

  let clarification = applyClarificationLifecycleEvent(message, clarificationEvent())
  lockMessageClarificationEvidence(message, clarification)
  assert.equal(isClarificationSubmittable(message.clarification), false)
  assert.deepEqual(message.sources, [])

  clarification = applyClarificationLifecycleEvent(message, persistedAck())
  lockMessageClarificationEvidence(message, clarification)
  assert.equal(isClarificationSubmittable(message.clarification), true)

  clarification = applyClarificationLifecycleEvent(message, {
    type: 'done',
    conversation_id: 'conversation-1',
  })
  lockMessageClarificationEvidence(message, clarification)
  assert.equal(message.clarification.invalidated, false)
  assert.equal(isClarificationSubmittable(message.clarification), true)
})

test('事件序列 clarification -> error -> done 无 ack 时永久失效', () => {
  const message = { role: 'assistant' }

  applyClarificationLifecycleEvent(message, clarificationEvent())
  applyClarificationLifecycleEvent(message, { type: 'error', message: '保存失败' })
  applyClarificationLifecycleEvent(message, { type: 'done' })

  assert.equal(message.clarification.acknowledged, false)
  assert.equal(message.clarification.invalidated, true)
  assert.equal(message.clarification.invalid_reason, 'server_error')
  assert.equal(isClarificationSubmittable(message.clarification), false)
  assert.equal(applyClarificationLifecycleEvent(message, persistedAck()), null)
})

test('同轮 error 早于 clarification 时也不能被后到 picker 和 ack 重新激活', () => {
  const message = { role: 'assistant' }

  applyClarificationLifecycleEvent(message, { type: 'error', message: '上游异常' })
  applyClarificationLifecycleEvent(message, clarificationEvent())

  assert.equal(message.clarification.invalidated, true)
  assert.equal(message.clarification.invalid_reason, 'server_error')
  assert.equal(applyClarificationLifecycleEvent(message, persistedAck()), null)
  assert.equal(isClarificationSubmittable(message.clarification), false)
})

test('ack 必须使用精确协议和完整持久化标识，畸形事件不会解锁', () => {
  for (const ack of [
    persistedAck({ schema_version: 'rag_evidence_clarification_ack.v2' }),
    persistedAck({ persisted: false }),
    persistedAck({ pending_state_id: '' }),
    persistedAck({ route_state_revision: null }),
  ]) {
    const message = { role: 'assistant' }
    attachEvidenceClarification(message, {
      choices: [{ key: 'c1', label: '云枢 8.6' }],
    })

    assert.equal(acknowledgeEvidenceClarification(message, ack), null)
    assert.equal(isClarificationSubmittable(message.clarification), false)
  }
})

test('未知版本的独立 clarification 事件不能与合法 ack 拼成可提交 picker', () => {
  const message = { role: 'assistant' }

  assert.equal(applyClarificationLifecycleEvent(
    message,
    clarificationEvent({ schema_version: 'rag_evidence_clarification.v2' }),
  ), null)
  assert.equal(applyClarificationLifecycleEvent(message, persistedAck()), null)
  assert.equal(isClarificationSubmittable(message.clarification), false)
})

test('错误或中止会永久失效且后到 ack 不能重新激活', () => {
  const message = { role: 'assistant' }
  attachEvidenceClarification(message, {
    choices: [{ key: 'c1', label: '公司出差管理标准.docx' }],
  })

  const invalidated = invalidateEvidenceClarification(message, 'server_error')
  assert.equal(invalidated.invalidated, true)
  assert.equal(invalidated.invalid_reason, 'server_error')
  assert.equal(acknowledgeEvidenceClarification(message, persistedAck()), null)
  assert.equal(markClarificationSubmitted(message, 'c1'), false)

  // done/finally 的兜底不得覆盖首个、更具体的失败原因。
  invalidateEvidenceClarification(message, 'missing_persistence_ack')
  assert.equal(message.clarification.invalid_reason, 'server_error')
})

test('历史仅恢复服务端确认且 message id 匹配的 active clarification', () => {
  const historyMessage = {
    id: 'assistant-1',
    role: 'assistant',
    content: '请选择适用版本',
    sources: [{ id: 'stale-source', evidence_role: 'direct' }],
    clarification: {
      schema_version: 'rag_evidence_clarification.v1',
      needs_clarification: true,
      choices: [{ key: 'c1', label: '云枢 8.6', versions: ['8.6'] }],
      acknowledged: true,
      persisted: true,
      pending_state_id: 'pending-1',
      clarification_message_id: 'assistant-1',
      route_state_revision: 3,
    },
  }

  const restored = restoreHistoryMessageClarification(historyMessage)
  assert.equal(isClarificationSubmittable(restored.clarification), true)
  assert.deepEqual(restored.sources, [])
  assert.equal(restored.evidence_status, 'needs_clarification')
  assert.equal(restored.search_meta.hit_count, 0)
  assert.deepEqual(restored.clarification.choices[0].versions, ['8.6'])
  assert.equal(markClarificationSubmitted(restored, restored.clarification.choices[0].reply), true)
  assert.equal(restored.clarification.submitted_reply, 'c1')

  const mismatched = restoreHistoryMessageClarification({
    ...historyMessage,
    id: 'different-assistant',
  })
  assert.equal(mismatched.clarification, null)

  const wrongSchema = restoreHistoryMessageClarification({
    ...historyMessage,
    clarification: {
      ...historyMessage.clarification,
      schema_version: 'rag_evidence_clarification.v2',
    },
  })
  assert.equal(wrongSchema.clarification, null)
})

test('无按钮的 refine 澄清可由手工补充消息关闭，但不是按钮可提交状态', () => {
  const message = {
    role: 'assistant',
    clarification: {
      needs_clarification: true,
      choices: [],
      acknowledged: true,
      persisted: true,
      pending_state_id: 'pending-1',
      clarification_message_id: 'assistant-1',
      route_state_revision: 3,
    },
  }

  assert.equal(isClarificationActive(message.clarification), true)
  assert.equal(isClarificationSubmittable(message.clarification), false)
  assert.equal(markClarificationSubmitted(message, '云枢 8.6', { allowFreeText: true }), true)
  assert.equal(message.clarification.submitted, true)
})

test('无 key 的兼容选项发送稳定序号', () => {
  const clarification = normalizeEvidenceClarification({
    dimension: 'version',
    choices: [
      { label: '6.0 版本' },
      { key: '包含 空格', label: '8.0 版本' },
    ],
  })

  assert.deepEqual(clarification.choices.map(choice => choice.reply), ['1', '2'])
})

test('重复 key 不会生成歧义按钮，后续项回退到序号', () => {
  const clarification = normalizeEvidenceClarification({
    choices: [
      { key: 'c1', label: '第一项' },
      { key: 'c1', label: '第二项' },
    ],
  })

  assert.deepEqual(clarification.choices.map(choice => choice.reply), ['c1', '2'])
})

test('超出上限或含畸形项时 fail closed 为补充范围提示', () => {
  const tooMany = normalizeEvidenceClarification({
    choices: Array.from(
      { length: MAX_CLARIFICATION_CHOICES + 1 },
      (_, index) => ({ key: `c${index + 1}`, label: `范围 ${index + 1}` }),
    ),
  })
  const malformed = normalizeEvidenceClarification({
    choices: [{ key: 'c1', label: '有效范围' }, { key: 'c2' }],
  })

  assert.deepEqual(tooMany.choices, [])
  assert.equal(tooMany.requires_refinement, true)
  assert.deepEqual(malformed.choices, [])
  assert.equal(malformed.requires_refinement, true)
})

test('choices 为空时仍保留澄清状态供 UI 提示用户补充范围', () => {
  const clarification = normalizeEvidenceClarification({
    needs_clarification: true,
    dimension: 'product_version',
    question: '请补充具体产品和版本。',
    choices: [],
  })

  assert.ok(clarification)
  assert.deepEqual(clarification.choices, [])
  assert.equal(clarification.requires_refinement, true)
})
