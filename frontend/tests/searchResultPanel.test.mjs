import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const panelSource = readFileSync(
  new URL('../src/components/search/SearchResultPanel.vue', import.meta.url),
  'utf8',
)

test('检索结果面板把待澄清状态显示为等待选择范围', () => {
  assert.match(
    panelSource,
    /needs_clarification:\s*\{\s*label:\s*'等待选择范围',\s*type:\s*'warning'\s*\}/,
  )
  assert.match(panelSource, /选择前这些片段不能作为回答依据/)
  assert.match(panelSource, /已检索，等待选择适用范围/)
})
