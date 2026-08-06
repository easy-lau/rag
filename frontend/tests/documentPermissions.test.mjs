import assert from 'node:assert/strict'
import test from 'node:test'

import {
  canDeleteDocumentRow,
  canReadDocumentRow,
  canUpdateDocumentRow,
  documentAllows,
} from '../src/utils/documentPermissions.js'


test('document permissions consume the backend decision', () => {
  const document = {
    permissions: { read: true, update: false, delete: true },
  }

  assert.equal(canReadDocumentRow(document), true)
  assert.equal(canUpdateDocumentRow(document), false)
  assert.equal(canDeleteDocumentRow(document), true)
  assert.equal(documentAllows(document, 'update'), false)
})


test('missing or malformed object permissions default to deny', () => {
  for (const document of [undefined, null, {}, { permissions: {} }]) {
    assert.equal(canReadDocumentRow(document), false)
    assert.equal(canUpdateDocumentRow(document), false)
    assert.equal(canDeleteDocumentRow(document), false)
  }

  assert.equal(documentAllows({ permissions: { update: 1 } }, 'update'), false)
})
