import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const panelSource = readFileSync(
  new URL('../src/components/search/SearchResultPanel.vue', import.meta.url),
  'utf8',
)

test('检索结果面板区分有限选项和需要自由补充的澄清', () => {
  assert.match(panelSource, /clarificationRequiresRefinement/)
  assert.match(panelSource, /等待补充范围/)
  assert.match(panelSource, /等待选择范围/)
  assert.match(panelSource, /选择前这些片段不能作为回答依据/)
  assert.match(panelSource, /已检索，等待选择适用范围/)
  assert.match(panelSource, /请在输入框补充具体产品、版本、项目或制度名称/)
  assert.match(panelSource, /回复序号、版本或“都对比”/)
})
