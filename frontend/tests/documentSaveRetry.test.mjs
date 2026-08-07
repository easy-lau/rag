import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

const documentViewSource = readFileSync(
  new URL('../src/views/DocumentView.vue', import.meta.url),
  'utf8',
)

test('手动文档首次创建后绑定同一文档，失败重试不会再次新建', () => {
  const createIndex = documentViewSource.indexOf(
    'doc = await createTextDocument(',
  )
  const bindIndex = documentViewSource.indexOf(
    'editingDocId.value = doc.id',
    createIndex,
  )
  const pollIndex = documentViewSource.indexOf(
    'await pollDocumentStatus(doc.id)',
    createIndex,
  )

  assert.ok(createIndex >= 0)
  assert.ok(bindIndex > createIndex)
  assert.ok(pollIndex > bindIndex)
  assert.match(documentViewSource, /updateTextDocument\(selectedKbId\.value, editingDocId\.value/)
})
