const EVIDENCE_ROLE_PRIORITY = {
  direct: 3,
  related: 2,
  '': 1,
  irrelevant: 0,
}

function finiteNumber(value) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function normalizedRole(value) {
  const role = typeof value === 'string' ? value.trim().toLowerCase() : ''
  return Object.prototype.hasOwnProperty.call(EVIDENCE_ROLE_PRIORITY, role) ? role : ''
}

function chunkIdentity(item, index) {
  const chunkId = item?.id || item?.chunk_id
  if (chunkId) return `chunk:${chunkId}`

  const chunkIndex = finiteNumber(item?.chunk_index)
  if (item?.doc_id && chunkIndex !== null) {
    return `document-chunk:${item?.kb_id || 'unknown-kb'}:${item.doc_id}:${chunkIndex}`
  }

  // 旧协议既没有 chunk id，也没有文档定位信息时，必须逐项隔离；不能仅凭
  // 同名文件把未知来源误合并。
  return `legacy-fragment:${index}`
}

function documentIdentity(item, chunkKey) {
  if (!item?.doc_id) return chunkKey
  return `document:${item?.kb_id || 'unknown-kb'}:${item.doc_id}`
}

/**
 * 只在展示层按文档归组。每个 chunk 及其独立分数都会保留，不能把同文档的
 * 多个证据片段去重成一条。
 */
export function groupEvidenceByDocument(items) {
  if (!Array.isArray(items) || !items.length) return []

  const groups = []
  const groupByKey = new Map()

  items.forEach((rawItem, index) => {
    if (!rawItem || typeof rawItem !== 'object') return

    const chunkKey = chunkIdentity(rawItem, index)
    const groupKey = documentIdentity(rawItem, chunkKey)
    let group = groupByKey.get(groupKey)
    if (!group) {
      group = {
        key: groupKey,
        kb_id: rawItem.kb_id || null,
        doc_id: rawItem.doc_id || null,
        filename: rawItem.filename || '未命名文档',
        file_type: rawItem.file_type || '',
        source_url: rawItem.source_url || null,
        first_rank: index + 1,
        evidence_role: '',
        best_topic_relevance: null,
        best_answer_support: null,
        items: [],
        _chunkKeys: new Set(),
      }
      groupByKey.set(groupKey, group)
      groups.push(group)
    }

    // 同一个 chunk 可能因兼容载荷重复到达；只去掉完全相同的 chunk，绝不按
    // doc_id 删除其他片段。
    if (group._chunkKeys.has(chunkKey)) return
    group._chunkKeys.add(chunkKey)

    const item = {
      ...rawItem,
      _chunkKey: chunkKey,
      _resultRank: index + 1,
    }
    group.items.push(item)

    const role = normalizedRole(item.evidence_role)
    if (
      EVIDENCE_ROLE_PRIORITY[role]
      > EVIDENCE_ROLE_PRIORITY[group.evidence_role]
    ) {
      group.evidence_role = role
    }

    const topic = finiteNumber(item.topic_relevance ?? item.rerank_score)
    const support = finiteNumber(item.answer_support)
    if (topic !== null && (group.best_topic_relevance === null || topic > group.best_topic_relevance)) {
      group.best_topic_relevance = topic
    }
    if (support !== null && (group.best_answer_support === null || support > group.best_answer_support)) {
      group.best_answer_support = support
    }
  })

  return groups.map(({ _chunkKeys, ...group }) => ({
    ...group,
    fragment_count: group.items.length,
  }))
}

/** 去掉解析器为了检索补入的“【文件名 > 标题】”上下文，只显示真实片段正文。 */
export function evidenceFragmentContent(source) {
  const original = String(source?.content || '').replace(/\r\n?/g, '\n').trim()
  if (!original) return ''
  const withoutContext = original.replace(/^【[^\n】]+】\s*(?:\n|$)/, '').trim()
  return withoutContext || original
}

export function evidenceFragmentLabel(source, fallbackIndex = 0) {
  const chunkIndex = finiteNumber(source?.chunk_index)
  const safeFallback = finiteNumber(fallbackIndex)
  const sequence = chunkIndex === null
    ? Math.max(1, Math.trunc(safeFallback ?? 0) + 1)
    : Math.max(0, Math.trunc(chunkIndex)) + 1
  return `片段 ${sequence}`
}

export function evidenceSectionLabel(source) {
  const heading = typeof source?.metadata?.heading === 'string'
    ? source.metadata.heading.trim()
    : ''
  if (!heading) return ''

  const parts = heading.split(/\s*(?:>|›|→)\s*/).filter(Boolean)
  const leaf = parts.at(-1) || ''
  const filename = String(source?.filename || '').replace(/\.[^.]+$/, '').trim()
  return leaf && leaf !== filename ? leaf : ''
}

export function normalizeEvidenceText(value) {
  return String(value || '')
    .replace(/\[([^\]]+)]\([^)]+\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[\s|#*_>`~\-\[\](){}:：,，。；;！？!?、“”"'·]+/g, '')
    .toLocaleLowerCase()
}

function plainAnchorLine(value) {
  return String(value || '')
    .replace(/^\s*#{1,6}\s+/, '')
    .replace(/^\s*(?:[-*+]\s+|\d+[.)、]\s*)/, '')
    .replace(/^\s*>\s?/, '')
    .replace(/\[([^\]]+)]\([^)]+\)/g, '$1')
    .replace(/[*_`]/g, '')
    .trim()
}

function evidenceAnchorRecords(source) {
  const content = evidenceFragmentContent(source)
  const heading = evidenceSectionLabel(source)
  const lines = content.split('\n')
  if (heading) lines.unshift(heading)

  const seen = new Set()
  const anchors = []
  for (const line of lines) {
    if (/^\s*\|?\s*:?[-]{3,}/.test(line) || /^\s*```/.test(line)) continue
    const text = plainAnchorLine(line)
    const normalized = normalizeEvidenceText(text)
    if (normalized.length < 4 || seen.has(normalized)) continue
    seen.add(normalized)
    anchors.push({ text, normalized })
  }

  return anchors
}

/** 生成用于在渲染后全文中定位的候选文本，不把 Markdown 表格分隔线当证据。 */
export function evidenceAnchorCandidates(source, limit = 12) {
  const anchors = evidenceAnchorRecords(source)

  return anchors
    .sort((a, b) => b.normalized.length - a.normalized.length)
    .slice(0, Math.max(1, limit))
    .map(item => item.text)
}

/**
 * 在一组渲染后块级文本中寻找最可能属于目标 chunk 的块。DOM 操作留在页面，
 * 这里保持纯函数，便于覆盖换行、表格和无法定位等回归场景。
 */
export function matchingEvidenceBlockIndexes(blockTexts, source, limit = 4) {
  if (!Array.isArray(blockTexts) || !blockTexts.length) return []
  const blocks = blockTexts.map(normalizeEvidenceText)
  const anchors = evidenceAnchorRecords(source).map(item => item.normalized)
  if (!anchors.length) return []

  const matches = (block, anchor) => Boolean(
    block
    && anchor
    && (
      block.includes(anchor)
      || (anchor.length >= 12 && block.length >= 6 && anchor.includes(block))
    )
  )
  const totalAnchorChars = anchors.reduce((sum, anchor) => sum + anchor.length, 0)
  const maxBlockGap = 4
  let best = null

  // 以每个“锚点 + 原文块”匹配作为起点，只接受后续顺序一致且相距较近的
  // 匹配。这样重复表头会与它附近的目标行组成证据窗口，不会把全文中相隔很远
  // 的几个通用词拼成一次“定位成功”。
  anchors.forEach((anchor, anchorStart) => {
    blocks.forEach((block, blockStart) => {
      if (!matches(block, anchor)) return

      const pairs = [{ anchorIndex: anchorStart, blockIndex: blockStart }]
      let lastBlockIndex = blockStart
      for (let anchorIndex = anchorStart + 1; anchorIndex < anchors.length; anchorIndex += 1) {
        let matchedBlockIndex = -1
        const end = Math.min(blocks.length - 1, lastBlockIndex + maxBlockGap)
        for (let blockIndex = lastBlockIndex; blockIndex <= end; blockIndex += 1) {
          if (matches(blocks[blockIndex], anchors[anchorIndex])) {
            matchedBlockIndex = blockIndex
            break
          }
        }
        if (matchedBlockIndex >= 0) {
          pairs.push({ anchorIndex, blockIndex: matchedBlockIndex })
          lastBlockIndex = matchedBlockIndex
        }
      }

      const matchedAnchorIndexes = [...new Set(pairs.map(pair => pair.anchorIndex))]
      const matchedChars = matchedAnchorIndexes.reduce(
        (sum, index) => sum + anchors[index].length,
        0,
      )
      const charCoverage = totalAnchorChars ? matchedChars / totalAnchorChars : 0
      const countCoverage = matchedAnchorIndexes.length / anchors.length
      const strongestMatch = Math.max(...matchedAnchorIndexes.map(index => anchors[index].length))
      const blockIndexes = [...new Set(pairs.map(pair => pair.blockIndex))].sort((a, b) => a - b)
      const span = blockIndexes.at(-1) - blockIndexes[0]
      const accepted = anchors.length === 1
        ? strongestMatch >= 4
        : strongestMatch >= 6 && (charCoverage >= 0.6 || countCoverage >= 0.67)
      if (!accepted) return

      const score = charCoverage * 2 + countCoverage - span * 0.01
      if (
        !best
        || score > best.score
        || (score === best.score && span < best.span)
        || (score === best.score && span === best.span && blockIndexes[0] > best.blockIndexes[0])
      ) {
        best = { score, span, blockIndexes }
      }
    })
  })

  if (!best) return []
  const safeLimit = Math.max(1, Math.trunc(Number(limit) || 1))
  if (best.blockIndexes.length <= safeLimit) return best.blockIndexes

  // 片段很长时只高亮起始位置附近的有限块，避免整页染色；匹配覆盖率已经在
  // 截断前完成校验，因此不会改变“是否定位成功”的结论。
  return best.blockIndexes.slice(0, safeLimit)
}

/** 外部资料链接只允许浏览器可安全跳转的 HTTP(S) 协议。 */
export function safeExternalSourceUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return ''
  try {
    const url = new URL(value.trim())
    return ['http:', 'https:'].includes(url.protocol) ? url.href : ''
  } catch {
    return ''
  }
}
