import test from 'node:test'
import assert from 'node:assert/strict'

import {
  evidenceCandidatePolicy,
  isNonAnswerEvidenceStatus,
  normalizeEvidenceStatus,
} from '../src/utils/evidenceStatus.js'

test('共享证据状态合同区分历史兼容版本状态与 V2 范围不匹配', () => {
  assert.equal(normalizeEvidenceStatus(' Scope_Mismatch '), 'scope_mismatch')
  assert.equal(isNonAnswerEvidenceStatus('scope_mismatch'), true)
  assert.equal(evidenceCandidatePolicy('scope_mismatch'), 'clear')

  // Old history remains readable, but new V2 code must not emit this status.
  assert.equal(normalizeEvidenceStatus('version_mismatch'), 'version_mismatch')
  assert.equal(isNonAnswerEvidenceStatus('version_mismatch'), false)
})

test('所有新 V2 无法形成回答依据的终态都 fail closed', () => {
  for (const status of [
    'skipped',
    'scope_mismatch',
    'needs_clarification',
    'no_hit',
    'insufficient_evidence',
    'unverified',
    'error',
  ]) {
    assert.equal(isNonAnswerEvidenceStatus(status), true, status)
  }
})
