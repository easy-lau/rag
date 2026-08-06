import test from 'node:test'
import assert from 'node:assert/strict'

import {
  restoreConversationMessages,
  restoreHistoryMessageState,
  searchSnapshotFromHistoryMessage,
} from '../src/utils/chatHistory.js'

const SCENARIOS = [
  ['出差交通标准', '公司出差管理标准.docx'],
  ['员工请假天数', '员工请假管理办法.docx'],
  ['登录用户名枚举防护', '应用安全配置基线.md'],
  ['CloudPivot 版本差异', 'CloudPivot 8.6 运维手册.pdf'],
]

test('不同业务场景都可恢复 counters + candidates 历史检索快照', () => {
  for (const [index, [scenario, filename]] of SCENARIOS.entries()) {
    const traceId = `trace-scenario-${index + 1}`
    const restored = restoreHistoryMessageState({
      id: `assistant-${scenario}`,
      role: 'assistant',
      content: `${scenario}的回答`,
      trace_id: traceId,
      evidence_status: 'hit',
      retrieval_executed: true,
      delivery_status: 'delivered',
      persistence_status: 'completed',
      search_snapshot: {
        schema_version: 'rag_search_snapshot.v1',
        candidates: [{ id: `chunk-${scenario}`, filename, evidence_role: 'direct' }],
        counters: {
          total: 1,
          displayed_result_count: 1,
          evidence_status: 'hit',
          retrieval_executed: true,
          trace_id: traceId,
        },
      },
    })

    assert.equal(restored.search_snapshot.results[0].filename, filename)
    assert.equal(restored.search_snapshot.total, 1)
    assert.equal(restored.search_snapshot.evidence_status, 'hit')
    assert.equal(restored.search_snapshot.historical, true)
    assert.equal(restored.trace_id, traceId)
  }
})

test('没有服务端检索状态的旧历史消息不伪造右侧面板', () => {
  assert.equal(searchSnapshotFromHistoryMessage({
    id: 'legacy-assistant',
    role: 'assistant',
    content: '旧回答',
    sources: [{ id: 'citation-only' }],
  }), null)
})

test('权威历史按 durable request 去重并用已保存正文替换空占位', () => {
  const restored = restoreConversationMessages([
    {
      id: 'assistant-placeholder',
      role: 'assistant',
      request_id: 'request-travel-001',
      content: '',
      turn_status: 'completed',
      persistence_status: 'completed',
    },
    {
      id: 'assistant-authoritative',
      role: 'assistant',
      request_id: 'request-travel-001',
      content: '普通员工乘坐高铁二等座。',
      turn_status: 'completed',
      persistence_status: 'completed',
    },
  ])

  assert.equal(restored.length, 1)
  assert.equal(restored[0].id, 'assistant-authoritative')
  assert.equal(restored[0].content, '普通员工乘坐高铁二等座。')
})

test('缺少 durable identity 的不同历史消息不会被误合并', () => {
  const restored = restoreConversationMessages([
    { id: 'legacy-1', role: 'assistant', content: '第一条旧回答' },
    { id: 'legacy-2', role: 'assistant', content: '第二条旧回答' },
  ])
  assert.deepEqual(restored.map(message => message.id), ['legacy-1', 'legacy-2'])
})

test('历史失败状态与 Trace 保留，供回答卡片复制诊断', () => {
  const restored = restoreHistoryMessageState({
    id: 'failed-assistant',
    role: 'assistant',
    content: '[错误：回答保存失败]',
    trace_id: 'trace-save-failure-001',
    turn_status: 'persist_failed',
    delivery_status: 'failed',
    persistence_status: 'failed',
  })

  assert.equal(restored.trace_id, 'trace-save-failure-001')
  assert.equal(restored.delivery_status, 'failed')
  assert.equal(restored.persistence_status, 'failed')
})

test('历史快照保留完整证据状态合同，避免 evidence_status 在恢复时被丢弃', () => {
  for (const status of ['insufficient_evidence', 'scope_mismatch']) {
    const restored = restoreHistoryMessageState({
      id: `assistant-${status}`,
      role: 'assistant',
      content: '确定性状态提示',
      evidence_status: status,
      retrieval_executed: true,
      search_snapshot: {
        schema_version: 'rag_search_snapshot.v1',
        counters: { evidence_status: status, retrieval_executed: true },
      },
    })

    assert.equal(restored.evidence_status, status)
    assert.equal(restored.search_snapshot.evidence_status, status)
  }
})

test('通用模型回答来源从持久化 counters 恢复到消息与检索快照', () => {
  const restored = restoreHistoryMessageState({
    id: 'assistant-general-fallback',
    role: 'assistant',
    content: '通用参考回答',
    evidence_status: 'no_hit',
    retrieval_executed: true,
    search_snapshot: {
      schema_version: 'rag_search_snapshot.v1',
      candidates: [],
      answer_sources: [],
      counters: {
        evidence_status: 'no_hit',
        retrieval_executed: true,
        answer_provenance: 'general_model',
        general_fallback_mode: 'no_hit',
      },
    },
  })

  assert.equal(restored.answer_provenance, 'general_model')
  assert.equal(restored.search_snapshot.answer_provenance, 'general_model')
  assert.equal(restored.search_meta.answer_provenance, 'general_model')
  assert.deepEqual(restored.search_snapshot.answer_sources, [])
})

test('choices=[] 的已授权历史澄清同步进入检索快照，供面板提示自由补充', () => {
  const restored = restoreHistoryMessageState({
    id: 'assistant-refinement',
    role: 'assistant',
    content: '请补充具体制度名称。',
    evidence_status: 'needs_clarification',
    retrieval_executed: true,
    clarification: {
      type: 'clarification_state',
      schema_version: 'rag_clarification_state.v1',
      status: 'active',
      needs_clarification: true,
      adapter: 'semantic',
      dimension: 'document',
      reason_code: 'missing_document',
      selection_mode: 'refine',
      choices: [],
      persisted: true,
      pending_state_id: 'pending-refinement',
      clarification_message_id: 'assistant-refinement',
      route_state_revision: 5,
      conversation_id: 'conversation-1',
      selected_kb_ids_snapshot: ['kb-1'],
    },
    search_snapshot: {
      schema_version: 'rag_search_snapshot.v1',
      candidates: [],
      counters: {
        evidence_status: 'needs_clarification',
        retrieval_executed: true,
      },
    },
  })

  assert.deepEqual(restored.clarification.choices, [])
  assert.equal(restored.clarification.requires_refinement, true)
  assert.deepEqual(restored.search_snapshot.clarification.choices, [])
  assert.equal(restored.search_snapshot.clarification.requires_refinement, true)
})

test('畸形历史澄清不会藏在 search snapshot 中恢复为可操作状态', () => {
  const restored = restoreHistoryMessageState({
    id: 'assistant-stale',
    role: 'assistant',
    content: '旧澄清',
    evidence_status: 'needs_clarification',
    clarification: {
      type: 'clarification_state',
      schema_version: 'rag_clarification_state.v1',
      status: 'active',
      needs_clarification: true,
      adapter: 'evidence',
      dimension: 'document',
      reason_code: 'multiple_documents',
      selection_mode: 'choice',
      choices: [{ key: 'c1', label: '制度 A' }],
      persisted: true,
      pending_state_id: 'pending-stale',
      clarification_message_id: 'different-message',
      route_state_revision: 1,
      conversation_id: 'conversation-1',
      selected_kb_ids_snapshot: ['kb-1'],
    },
    search_snapshot: {
      schema_version: 'rag_search_snapshot.v1',
      clarification: { needs_clarification: true, choices: [{ key: 'c1', label: '制度 A' }] },
      counters: { evidence_status: 'needs_clarification' },
    },
  })

  assert.equal(restored.clarification, null)
  assert.equal(restored.search_snapshot.clarification, undefined)
})
