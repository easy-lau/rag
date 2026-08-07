import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const panelSource = readFileSync(
  new URL('../src/components/search/SearchResultPanel.vue', import.meta.url),
  'utf8',
)
const documentGroupSource = readFileSync(
  new URL('../src/components/search/DocumentEvidenceGroup.vue', import.meta.url),
  'utf8',
)
const chatMessageSource = readFileSync(
  new URL('../src/components/chat/ChatMessage.vue', import.meta.url),
  'utf8',
)

test('检索结果面板区分有限选项和需要自由补充的澄清', () => {
  assert.match(panelSource, /clarificationRequiresRefinement/)
  assert.match(panelSource, /等待补充范围/)
  assert.match(panelSource, /等待选择范围/)
  assert.match(panelSource, /选择前这些资料不能作为回答依据/)
  assert.match(panelSource, /已检索，等待选择适用范围/)
  assert.match(panelSource, /请在输入框补充具体产品、版本、项目或制度名称/)
  assert.match(panelSource, /回复序号、版本或“都对比”/)
})

test('授权候选存在但证据不足时展示文档确认而不是无资料', () => {
  assert.match(panelSource, /candidateConfirmation/)
  assert.match(panelSource, /answerability_status/)
  assert.match(panelSource, /已找到当前权限范围内的候选文章/)
  assert.match(panelSource, /确认后系统只会在所选文章内/)
  assert.match(panelSource, /确认前不会把候选资料当作已验证答案/)
})

test('右侧检索面板只按文章展示，不展开候选片段', () => {
  assert.match(panelSource, /document-only/)
  assert.match(panelSource, /\{\{ displayedDocumentCount \}\} 篇文章/)
  assert.doesNotMatch(panelSource, /fragment-label="候选片段"/)
  assert.doesNotMatch(panelSource, /个片段/)
  assert.match(documentGroupSource, /v-if="!documentOnly"[\s\S]*?class="evidence-document__fragments"/)
  assert.match(documentGroupSource, /v-else[\s\S]*?evidence-document__toggle--static/)
})

test('回答卡片单独提示相近文章，不冒充回答依据', () => {
  assert.match(chatMessageSource, /找到 \{\{ relatedCandidateGroups\.length \}\} 篇相近文章/)
  assert.match(chatMessageSource, /未作为本条回答的已验证依据/)
  assert.match(chatMessageSource, /visibleRelatedCandidateGroups/)
  assert.match(chatMessageSource, /@click="\$emit\('inspect', message\)"/)
  assert.match(chatMessageSource, /answerDocumentKeys/)
})

test('回答依据只按文章展示，不展开实际采用片段', () => {
  assert.match(chatMessageSource, /本条回答使用的知识库文章/)
  assert.match(chatMessageSource, /这里只展示实际进入回答上下文的文章，不展开命中片段/)
  assert.match(chatMessageSource, /<DocumentEvidenceGroup[\s\S]*?document-only[\s\S]*?id-prefix=/)
  assert.doesNotMatch(chatMessageSource, /fragment-label="采用片段"/)
  assert.doesNotMatch(chatMessageSource, /\{\{ sourceDocumentCount \}\} 篇 · \{\{ sources\.length \}\} 个片段/)
})

test('回答依据和右侧检索中的全文按钮在文章行内居中并留出边距', () => {
  assert.match(documentGroupSource, /\.evidence-document__header\s*\{[\s\S]*?align-items:\s*center/)
  assert.match(documentGroupSource, /\.evidence-document__full-button\s*\{[\s\S]*?align-self:\s*center/)
  assert.match(documentGroupSource, /\.evidence-document__full-button\s*\{[\s\S]*?justify-content:\s*center/)
  assert.match(documentGroupSource, /margin:\s*0 8px 0 4px/)
  assert.match(documentGroupSource, /border-radius:\s*var\(--ui-radius-control\)/)
})
