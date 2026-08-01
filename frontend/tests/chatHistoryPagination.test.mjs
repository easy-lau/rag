import test from 'node:test'
import assert from 'node:assert/strict'
import { fetchAllConversationPages } from '../src/utils/chatHistoryPagination.js'

test('逐页加载超过首个 20 条限制的全部对话', async () => {
  const calls = []
  const rows = await fetchAllConversationPages(async params => {
    calls.push(params)
    if (params.page === 1) return Array.from({ length: 100 }, (_, index) => ({ id: `a-${index}` }))
    return Array.from({ length: 23 }, (_, index) => ({ id: `b-${index}` }))
  })

  assert.equal(rows.length, 123)
  assert.deepEqual(calls, [
    { page: 1, page_size: 100 },
    { page: 2, page_size: 100 },
  ])
})

test('重复的完整页面会停止，避免旧服务忽略页码时无限请求', async () => {
  let calls = 0
  const page = Array.from({ length: 100 }, (_, index) => ({ id: `item-${index}` }))
  const rows = await fetchAllConversationPages(async () => {
    calls += 1
    return page
  })

  assert.equal(rows.length, 100)
  assert.equal(calls, 2)
})

test('较早的加载请求失效时不会返回半页结果覆盖新请求', async () => {
  const rows = await fetchAllConversationPages(
    async () => [{ id: 'item-1' }],
    { isCurrent: () => false },
  )

  assert.equal(rows, null)
})
