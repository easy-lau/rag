import test from 'node:test'
import assert from 'node:assert/strict'

import {
  answerSourcesFromSearchEvent,
  persistedAnswerSources,
} from '../src/utils/chatEvidence.js'

const direct = { id: 'direct', evidence_role: 'direct' }
const related = { id: 'related', evidence_role: 'related' }

test('新版事件只把 answer_sources 挂到回答，右侧 results 不参与', () => {
  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'hit',
    results: [direct, related],
    answer_sources: [direct],
  }), [direct])
})

test('同一文档有多个候选时仍只持久化实际采用片段', () => {
  const first = { id: 'chunk-1', doc_id: 'doc-1', evidence_role: 'related' }
  const second = { id: 'chunk-2', doc_id: 'doc-1', evidence_role: 'related' }
  const third = { id: 'chunk-3', doc_id: 'doc-1', evidence_role: 'related' }

  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'partial',
    results: [first, second, third],
    answer_sources: [first],
  }), [first])
})

test('同一文档确有两个采用片段时不能按 doc_id 去重', () => {
  const first = { id: 'chunk-1', doc_id: 'doc-1', evidence_role: 'direct' }
  const second = { id: 'chunk-2', doc_id: 'doc-1', evidence_role: 'direct' }

  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'hit',
    results: [first, second],
    answer_sources: [first, second],
  }), [first, second])
})

test('no_hit 即使携带 results 或异常 answer_sources 也不展示为回答依据', () => {
  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'no_hit',
    results: [related],
    answer_sources: [related],
  }), [])
})

test('skipped 和 error 即使携带异常 answer_sources 也不展示为回答依据', () => {
  for (const evidenceStatus of ['skipped', 'error']) {
    assert.deepEqual(answerSourcesFromSearchEvent({
      evidence_status: evidenceStatus,
      results: [direct],
      answer_sources: [direct],
    }), [])
  }
})

test('旧协议按 direct 角色和 context 数量保守恢复回答依据', () => {
  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'partial',
    context_evidence_count: 1,
    results: [direct, related],
  }), [direct])
})

test('旧协议允许恢复没有 direct 但明确进入上下文的版本差异资料', () => {
  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'version_mismatch',
    context_evidence_count: 1,
    results: [related, { id: 'other', evidence_role: 'related' }],
  }), [related])
})

test('旧协议 context 为零时不把 optional 检索候选挂到回答', () => {
  assert.deepEqual(answerSourcesFromSearchEvent({
    evidence_status: 'unverified',
    context_evidence_count: 0,
    results: [related],
  }), [])
})

test('历史消息会过滤每项标记为 no_hit 的旧来源', () => {
  assert.deepEqual(persistedAnswerSources({
    sources: [
      { id: 'old-related', evidence_status: 'no_hit', evidence_role: 'related' },
      { id: 'used', evidence_status: 'hit', evidence_role: 'direct' },
    ],
  }), [{ id: 'used', evidence_status: 'hit', evidence_role: 'direct' }])
})

test('消息级 no_hit 会隐藏所有旧来源', () => {
  assert.deepEqual(persistedAnswerSources({
    evidence_status: 'no_hit',
    sources: [related],
  }), [])
})
