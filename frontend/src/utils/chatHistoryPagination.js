/** Fetch every page of the additive conversation-history API safely. */
export async function fetchAllConversationPages(fetchPage, { pageSize = 100, isCurrent = () => true } = {}) {
  const normalizedPageSize = Math.max(1, Math.min(Number(pageSize) || 100, 100))
  const rows = []
  const seenConversationIds = new Set()
  let page = 1

  while (true) {
    const pageRows = await fetchPage({ page, page_size: normalizedPageSize })
    if (!isCurrent()) return null
    if (!Array.isArray(pageRows)) throw new TypeError('对话历史接口返回格式无效')

    let addedCount = 0
    for (const conversation of pageRows) {
      const conversationId = typeof conversation?.id === 'string'
        ? conversation.id.trim()
        : ''
      if (!conversationId || seenConversationIds.has(conversationId)) continue
      seenConversationIds.add(conversationId)
      rows.push(conversation)
      addedCount += 1
    }

    // A short page is terminal. A duplicate-only full page is a defensive
    // terminal condition for an old proxy/service that ignores ``page``.
    if (pageRows.length < normalizedPageSize || addedCount === 0) return rows
    page += 1
  }
}
