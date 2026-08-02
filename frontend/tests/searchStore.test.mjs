import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'

import { useSearchStore } from '../src/stores/search.js'
import {
  applyClarificationLifecycleEvent,
  lockMessageClarificationEvidence,
} from '../src/utils/chatClarification.js'

function freshStore() {
  setActivePinia(createPinia())
  return useSearchStore()
}

test('no_hit 的相近候选保留展示数量，但有效命中保持为零', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'no_hit',
    retrieval_executed: true,
    displayed_candidate_count: 2,
    hit_count: 0,
    direct_evidence_count: 0,
    related_reference_count: 2,
    results: [
      { id: 'a', evidence_role: 'related' },
      { id: 'b', evidence_role: 'related' },
    ],
  })

  assert.equal(store.results.length, 2)
  assert.equal(store.totalCount, 2)
  assert.equal(store.searchMeta.displayed_candidate_count, 2)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 2)
})

test('no_hit 会覆盖旧协议中自相矛盾的非零命中数', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'no_hit',
    retrieval_executed: true,
    displayed_result_count: 2,
    hit_count: 5,
    direct_evidence_count: 5,
    related_reference_count: 0,
    results: [
      { id: 'a', evidence_role: 'direct' },
      { id: 'b', evidence_role: 'related' },
    ],
  })

  assert.equal(store.searchMeta.displayed_candidate_count, 2)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 2)
  assert.deepEqual(store.results.map(item => item.evidence_role), ['related', 'related'])
})

test('insufficient_evidence 保留相关候选但绝不显示回答依据', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'insufficient_evidence',
    retrieval_executed: true,
    displayed_result_count: 2,
    hit_count: 2,
    direct_evidence_count: 2,
    related_reference_count: 0,
    context_evidence_count: 2,
    answer_source_count: 2,
    results: [
      { id: 'mapping', evidence_role: 'direct' },
      { id: 'policy', evidence_role: 'related' },
    ],
  })

  assert.equal(store.searchMeta.evidence_status, 'insufficient_evidence')
  assert.equal(store.searchMeta.retrieval_executed, true)
  assert.equal(store.searchMeta.displayed_candidate_count, 2)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.equal(store.searchMeta.answer_source_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 2)
  assert.deepEqual(store.results.map(item => item.evidence_role), ['related', 'related'])
})

test('skipped 和 error 不保留异常候选或旧版命中统计', () => {
  for (const evidenceStatus of ['skipped', 'error']) {
    const store = freshStore()
    store.setResults({
      evidence_status: evidenceStatus,
      retrieval_executed: evidenceStatus === 'skipped' ? false : true,
      displayed_result_count: 1,
      hit_count: 3,
      direct_evidence_count: 3,
      related_reference_count: 1,
      results: [{ id: evidenceStatus, evidence_role: 'direct' }],
    })

    assert.equal(store.searchMeta.evidence_status, evidenceStatus)
    assert.equal(store.searchMeta.displayed_candidate_count, 0)
    assert.equal(store.searchMeta.hit_count, 0)
    assert.equal(store.searchMeta.direct_evidence_count, 0)
    assert.equal(store.searchMeta.related_reference_count, 0)
    assert.deepEqual(store.results, [])
  }
})

test('scope_mismatch 会清空范围外候选，不能以相近资料形式泄漏回面板', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'scope_mismatch',
    retrieval_executed: true,
    displayed_result_count: 2,
    hit_count: 2,
    direct_evidence_count: 2,
    related_reference_count: 2,
    context_evidence_count: 2,
    results: [
      { id: 'wrong-project', evidence_role: 'direct' },
      { id: 'wrong-version', evidence_role: 'related' },
    ],
  })

  assert.equal(store.searchMeta.evidence_status, 'scope_mismatch')
  assert.equal(store.searchMeta.retrieval_executed, true)
  assert.equal(store.searchMeta.displayed_candidate_count, 0)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 0)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.deepEqual(store.results, [])
})

test('历史 scope_mismatch 快照恢复后仍会清空范围外候选', () => {
  const store = freshStore()
  const restored = store.restoreSnapshot({
    schema_version: 'rag_search_snapshot.v1',
    evidence_status: 'scope_mismatch',
    retrieval_executed: true,
    displayed_result_count: 1,
    candidates: [{ id: 'historical-wrong-version', evidence_role: 'related' }],
  })

  assert.equal(restored, true)
  assert.equal(store.contextMode, 'history')
  assert.equal(store.searchMeta.evidence_status, 'scope_mismatch')
  assert.equal(store.searchMeta.displayed_candidate_count, 0)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.deepEqual(store.results, [])
})

test('needs_clarification 表示检索已执行但选择前没有回答依据', () => {
  const store = freshStore()
  const clarification = {
    dimension: 'version',
    choices: [
      { id: 'c1', label: '云枢 6.0.1' },
      { id: 'c2', label: '云枢 8.2.75' },
    ],
  }
  store.setResults({
    evidence_status: 'needs_clarification',
    displayed_result_count: 2,
    hit_count: 2,
    direct_evidence_count: 2,
    related_reference_count: 0,
    context_evidence_count: 2,
    answer_source_count: 2,
    clarification,
    results: [
      { id: 'v6', evidence_role: 'direct' },
      { id: 'v8', evidence_role: 'related' },
    ],
  })

  assert.equal(store.searchMeta.evidence_status, 'needs_clarification')
  assert.equal(store.searchMeta.retrieval_executed, true)
  assert.equal(store.searchMeta.displayed_candidate_count, 2)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 2)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.equal(store.searchMeta.answer_source_count, 0)
  assert.deepEqual(store.searchMeta.clarification, clarification)
  assert.deepEqual(store.results.map(item => item.evidence_role), ['related', 'related'])
})

test('独立澄清事件可补写 search meta，等待后续 search_results 合并', () => {
  const store = freshStore()
  const clarification = {
    dimension: 'document',
    choices: [{ key: 'c1', label: '公司出差管理标准.docx' }],
  }

  store.setClarification(clarification)

  assert.equal(store.searchMeta.evidence_status, 'needs_clarification')
  assert.deepEqual(store.searchMeta.clarification, clarification)
})

test('独立澄清事件立即降级 direct 并清空全部回答依据统计', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'hit',
    retrieval_executed: true,
    displayed_result_count: 2,
    hit_count: 2,
    direct_evidence_count: 2,
    context_evidence_count: 2,
    answer_source_count: 2,
    coverage_status: 'complete',
    results: [
      { id: 'direct-1', evidence_role: 'direct' },
      { id: 'direct-2', evidence_role: 'direct' },
    ],
  })

  store.setClarification({
    dimension: 'document',
    choices: [{ key: 'c1', label: '公司出差管理标准.docx' }],
  })

  assert.deepEqual(store.results.map(item => item.evidence_role), ['related', 'related'])
  assert.equal(store.searchMeta.evidence_status, 'needs_clarification')
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.equal(store.searchMeta.answer_source_count, 0)
  assert.equal(store.searchMeta.related_reference_count, 2)
  assert.equal(store.searchMeta.coverage_status, 'insufficient')
})

test('clarification 锁定后乱序 hit search_results 仍保持 fail closed', () => {
  const store = freshStore()
  const message = {
    role: 'assistant',
    sources: [],
  }
  const clarification = applyClarificationLifecycleEvent(message, {
    type: 'evidence_clarification',
    schema_version: 'rag_evidence_clarification.v1',
    dimension: 'version',
    choices: [
      { key: 'c1', label: '云枢 6' },
      { key: 'c2', label: '云枢 8.6' },
    ],
  })
  lockMessageClarificationEvidence(message, clarification)
  store.setClarification(clarification)

  const contradictoryHit = {
    evidence_status: 'hit',
    retrieval_executed: true,
    hit_count: 1,
    direct_evidence_count: 1,
    context_evidence_count: 1,
    answer_source_count: 1,
    coverage_status: 'complete',
    answer_sources: [{ id: 'late-direct', evidence_role: 'direct' }],
    results: [{ id: 'late-direct', evidence_role: 'direct' }],
  }
  store.setResults(contradictoryHit)
  // This is the same final lock applied by the chat event path after it sees
  // an existing picker; even a contradictory answer_sources payload is gone.
  message.sources = contradictoryHit.answer_sources
  lockMessageClarificationEvidence(message, message.clarification)

  assert.deepEqual(message.sources, [])
  assert.equal(message.evidence_status, 'needs_clarification')
  assert.equal(message.search_meta.hit_count, 0)
  assert.equal(store.searchMeta.evidence_status, 'needs_clarification')
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.equal(store.searchMeta.context_evidence_count, 0)
  assert.equal(store.searchMeta.answer_source_count, 0)
  assert.equal(store.searchMeta.coverage_status, 'insufficient')
  assert.deepEqual(store.results.map(item => item.evidence_role), ['related'])
  assert.deepEqual(store.searchMeta.clarification, clarification)
})

test('旧协议明确未执行检索时优先判定为 skipped', () => {
  const store = freshStore()
  store.setResults({
    retrieval_executed: false,
    displayed_result_count: 1,
    hit_count: 3,
    direct_evidence_count: 3,
    results: [{ id: 'stale', evidence_role: 'direct' }],
  })

  assert.equal(store.searchMeta.evidence_status, 'skipped')
  assert.equal(store.searchMeta.displayed_candidate_count, 0)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.direct_evidence_count, 0)
  assert.deepEqual(store.results, [])
})

test('候选总数与 direct 回答依据分别统计', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'partial',
    displayed_result_count: 4,
    hit_count: 1,
    direct_evidence_count: 1,
    related_reference_count: 2,
    results: [
      { id: 'direct', evidence_role: 'direct' },
      { id: 'related-1', evidence_role: 'related' },
      { id: 'related-2', evidence_role: 'related' },
      { id: 'unknown' },
    ],
  })

  assert.equal(store.searchMeta.displayed_candidate_count, 4)
  assert.equal(store.searchMeta.hit_count, 1)
  assert.equal(store.searchMeta.direct_evidence_count, 1)
  assert.equal(store.searchMeta.unverified_reference_count, 1)
})

test('store 完整保留同一文档的多个候选片段，展示分组不能污染数据层', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'partial',
    displayed_result_count: 3,
    direct_evidence_count: 0,
    related_reference_count: 3,
    results: [
      { id: 'chunk-3', doc_id: 'doc-1', chunk_index: 2, evidence_role: 'related' },
      { id: 'chunk-2', doc_id: 'doc-1', chunk_index: 1, evidence_role: 'related' },
      { id: 'chunk-1', doc_id: 'doc-1', chunk_index: 0, evidence_role: 'related' },
    ],
  })

  assert.equal(store.totalCount, 3)
  assert.deepEqual(store.results.map(item => item.id), ['chunk-3', 'chunk-2', 'chunk-1'])
})

test('联合证据部分覆盖时保留生成上下文与缺失需求指标', () => {
  const store = freshStore()
  store.setResults({
    evidence_status: 'partial',
    coverage_status: 'partial',
    context_evidence_count: 2,
    answer_source_count: 2,
    missing_requirement_count: 1,
    expansion_attempted: true,
    joint_support_score: 0.76,
    results: [
      {
        id: 'bridge',
        evidence_role: 'related',
        jointly_selected: true,
        coverage_status: 'partial',
      },
      {
        id: 'detail',
        evidence_role: 'related',
        jointly_selected: true,
        coverage_status: 'partial',
      },
    ],
  })

  assert.equal(store.searchMeta.coverage_status, 'partial')
  assert.equal(store.searchMeta.context_evidence_count, 2)
  assert.equal(store.searchMeta.answer_source_count, 2)
  assert.equal(store.searchMeta.missing_requirement_count, 1)
  assert.equal(store.searchMeta.expansion_attempted, true)
  assert.equal(store.searchMeta.joint_support_score, 0.76)
})

test('仅有 results 的旧接口按待验证候选处理，不推导为有效命中', () => {
  const store = freshStore()
  store.setResults({ results: [{ id: 'legacy' }], total: 1 })

  assert.equal(store.searchMeta.evidence_status, 'unverified')
  assert.equal(store.searchMeta.displayed_candidate_count, 1)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.unverified_reference_count, 1)
})

test('新语义协议只使用后端编译合同决定检索策略', () => {
  const store = freshStore()
  store.setIntentDecision({
    action: 'chat',
    route_decision: {
      schema_version: 'rag_route_decision.v1',
      intent_code: 'knowledge_qa',
      relation: 'refinement',
      readiness: 'ready',
    },
    task_contract: {
      schema_version: 'rag_task_contract.v1',
      response_mode: 'grounded_qa',
      retrieval_policy: 'required',
      need_retrieval: true,
      decision_reason: 'compiled_knowledge_qa',
    },
  })

  assert.equal(store.intentDecision.intent_code, 'knowledge_qa')
  assert.equal(store.intentDecision.relation, 'refinement')
  assert.equal(store.intentDecision.response_mode, 'grounded_qa')
  assert.equal(store.intentDecision.retrieval_policy, 'required')
  assert.equal(store.intentDecision.need_retrieval, true)
  assert.equal(store.intentDecision.decision_reason, 'compiled_knowledge_qa')
})

test('新语义决定缺少编译合同时不从 action 猜测执行策略', () => {
  const store = freshStore()
  store.setIntentDecision({
    action: 'retrieve',
    route_decision: {
      schema_version: 'rag_route_decision.v1',
      intent_code: 'knowledge_qa',
      relation: 'followup',
      readiness: 'needs_clarification',
    },
  })

  assert.equal(store.intentDecision.response_mode, '')
  assert.equal(store.intentDecision.retrieval_policy, '')
  assert.equal(store.intentDecision.need_retrieval, null)
})

test('新语义决定缺少合同时忽略矛盾的顶层旧执行字段', () => {
  const store = freshStore()
  store.setIntentDecision({
    action: 'retrieve',
    response_mode: 'grounded_qa',
    retrieval_policy: 'required',
    need_retrieval: true,
    route_decision: {
      schema_version: 'rag_route_decision.v1',
      intent_code: 'knowledge_qa',
      relation: 'followup',
      readiness: 'needs_clarification',
    },
  })

  assert.equal(store.intentDecision.response_mode, '')
  assert.equal(store.intentDecision.retrieval_policy, '')
  assert.equal(store.intentDecision.need_retrieval, null)
})

test('无协议版本的旧意图事件继续兼容 action 映射', () => {
  const store = freshStore()
  store.setIntentDecision({ action: 'retrieve', intent_code: 'knowledge_qa' })

  assert.equal(store.intentDecision.response_mode, 'grounded_qa')
  assert.equal(store.intentDecision.retrieval_policy, 'required')
  assert.equal(store.intentDecision.need_retrieval, true)
})
