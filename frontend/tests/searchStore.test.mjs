import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'

import { useSearchStore } from '../src/stores/search.js'

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

test('仅有 results 的旧接口按待验证候选处理，不推导为有效命中', () => {
  const store = freshStore()
  store.setResults({ results: [{ id: 'legacy' }], total: 1 })

  assert.equal(store.searchMeta.evidence_status, 'unverified')
  assert.equal(store.searchMeta.displayed_candidate_count, 1)
  assert.equal(store.searchMeta.hit_count, 0)
  assert.equal(store.searchMeta.unverified_reference_count, 1)
})
