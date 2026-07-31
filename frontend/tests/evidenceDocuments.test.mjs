import test from 'node:test'
import assert from 'node:assert/strict'

import {
  evidenceAnchorCandidates,
  evidenceFragmentContent,
  evidenceFragmentLabel,
  evidenceSectionLabel,
  groupEvidenceByDocument,
  matchingEvidenceBlockIndexes,
  safeExternalSourceUrl,
} from '../src/utils/evidenceDocuments.js'

function source(overrides = {}) {
  return {
    id: 'chunk-1',
    kb_id: 'kb-1',
    doc_id: 'doc-1',
    filename: '公司出差管理标准.docx',
    file_type: 'docx',
    chunk_index: 0,
    content: '默认内容',
    evidence_role: 'related',
    topic_relevance: 0.8,
    answer_support: 0.2,
    ...overrides,
  }
}

test('同一文档的三个片段归为一个文档组且保持后端排序和独立分数', () => {
  const input = [
    source({ id: 'chunk-3', chunk_index: 2, topic_relevance: 1, answer_support: 0.55 }),
    source({ id: 'chunk-2', chunk_index: 1, topic_relevance: 0.85, answer_support: 0.2 }),
    source({ id: 'chunk-1', chunk_index: 0, topic_relevance: 0.95, answer_support: 0.1 }),
  ]
  const snapshot = structuredClone(input)

  const groups = groupEvidenceByDocument(input)

  assert.equal(groups.length, 1)
  assert.equal(groups[0].fragment_count, 3)
  assert.deepEqual(groups[0].items.map(item => item.id), ['chunk-3', 'chunk-2', 'chunk-1'])
  assert.deepEqual(groups[0].items.map(item => item.answer_support), [0.55, 0.2, 0.1])
  assert.equal(groups[0].best_topic_relevance, 1)
  assert.equal(groups[0].best_answer_support, 0.55)
  assert.deepEqual(input, snapshot, '展示分组不能原地修改 store 数据')
})

test('同名但 doc_id 不同的文档绝不合并', () => {
  const groups = groupEvidenceByDocument([
    source({ id: 'a', doc_id: 'doc-a', filename: '同名文档.docx' }),
    source({ id: 'b', doc_id: 'doc-b', filename: '同名文档.docx' }),
  ])

  assert.equal(groups.length, 2)
  assert.deepEqual(groups.map(group => group.doc_id), ['doc-a', 'doc-b'])
})

test('只去掉完全相同的 chunk，不删除同文档其他片段', () => {
  const groups = groupEvidenceByDocument([
    source({ id: 'same', chunk_index: 0 }),
    source({ id: 'same', chunk_index: 0, answer_support: 0.9 }),
    source({ id: 'other', chunk_index: 1 }),
  ])

  assert.equal(groups.length, 1)
  assert.deepEqual(groups[0].items.map(item => item.id), ['same', 'other'])
})

test('旧协议缺少 doc_id 时不会仅凭文件名误合并', () => {
  const groups = groupEvidenceByDocument([
    source({ id: 'legacy-a', doc_id: null, filename: '同名.md' }),
    source({ id: 'legacy-b', doc_id: null, filename: '同名.md' }),
  ])

  assert.equal(groups.length, 2)
})

test('片段展示去掉检索上下文前缀并保留片段编号和章节', () => {
  const item = source({
    chunk_index: 2,
    content: '【公司出差管理标准 > 职级分类】\n普通员工、专员属于 D 级。',
    metadata: { heading: '公司出差管理标准 > 职级分类' },
  })

  assert.equal(evidenceFragmentContent(item), '普通员工、专员属于 D 级。')
  assert.equal(evidenceFragmentLabel(item), '片段 3')
  assert.equal(evidenceSectionLabel(item), '职级分类')
})

test('全文定位可以跨换行和 Markdown 表格符号匹配目标行', () => {
  const item = source({
    content: [
      '【公司出差管理标准 > 职级分类】',
      '| 职级 | 适用人员 |',
      '| --- | --- |',
      '| D级 | 普通员工、专员 |',
    ].join('\n'),
  })
  const blockTexts = [
    '公司出差管理标准',
    '职级 适用人员',
    'D级    普通员工、专员',
    '其他内容',
  ]

  assert.ok(evidenceAnchorCandidates(item).includes('| D级 | 普通员工、专员 |'))
  assert.deepEqual(matchingEvidenceBlockIndexes(blockTexts, item), [1, 2])
})

test('全文没有对应内容时返回空定位结果', () => {
  const item = source({ content: '【文档 > 章节】\n完全不存在的目标内容' })
  assert.deepEqual(matchingEvidenceBlockIndexes(['另一段正文', '其他表格行'], item), [])
})

test('分散在全文不同章节的通用词不能拼成定位成功', () => {
  const item = source({
    content: [
      '费用标准',
      '住宿标准为一线城市每晚不超过四百五十元',
      '餐饮补贴为每天一百元',
    ].join('\n'),
  })
  const blockTexts = [
    '费用标准',
    '无关内容一',
    '无关内容二',
    '无关内容三',
    '无关内容四',
    '无关内容五',
    '住宿标准为一线城市每晚不超过四百五十元',
    '另一章节',
    '其他说明一',
    '其他说明二',
    '其他说明三',
    '其他说明四',
    '其他说明五',
    '餐饮补贴为每天一百元',
  ]

  assert.deepEqual(matchingEvidenceBlockIndexes(blockTexts, item), [])
})

test('重复表头时选择与目标数据行连续出现的表格位置', () => {
  const item = source({
    content: [
      '| 职级 | 适用人员 |',
      '| --- | --- |',
      '| D级 | 普通员工、专员 |',
    ].join('\n'),
  })
  const blockTexts = [
    '职级 适用人员',
    'A级 董事长、总经理',
    '其他章节一',
    '其他章节二',
    '其他章节三',
    '其他章节四',
    '职级 适用人员',
    'D级 普通员工、专员',
  ]

  assert.deepEqual(matchingEvidenceBlockIndexes(blockTexts, item), [6, 7])
})

test('外部来源链接只允许 http 和 https 协议', () => {
  assert.equal(safeExternalSourceUrl('javascript:alert(1)'), '')
  assert.equal(safeExternalSourceUrl('data:text/html,hello'), '')
  assert.equal(safeExternalSourceUrl('/relative/path'), '')
  assert.equal(safeExternalSourceUrl('https://example.com/docs?a=1'), 'https://example.com/docs?a=1')
  assert.equal(safeExternalSourceUrl('http://example.com/guide'), 'http://example.com/guide')
})
