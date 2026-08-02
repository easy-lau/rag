import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const apiSource = readFileSync(new URL('../src/api/chat.js', import.meta.url), 'utf8')
const storeSource = readFileSync(new URL('../src/stores/chat.js', import.meta.url), 'utf8')
const sidebarSource = readFileSync(new URL('../src/layout/ChatSidebar.vue', import.meta.url), 'utf8')

test('批量删除通过一个有界请求提交会话 id 集合', () => {
  assert.match(apiSource, /http\.post\('\/chat\/batch-delete', \{[\s\S]*conversation_ids: conversationIds/)
  assert.ok(storeSource.includes('async function removeConversations(convIds)'))
  assert.ok(storeSource.includes('deleteConversationsRequest(ids)'))
  assert.ok(storeSource.includes('historyRequestId += 1'))
  assert.ok(storeSource.includes('deletedIds.has(String(currentConvId.value))'))
})

test('侧栏使用多选管理和统一危险确认，不提供无确认的一键清空', () => {
  assert.ok(sidebarSource.includes('aria-label="批量管理对话历史"'))
  assert.ok(sidebarSource.includes('aria-label="全选对话"'))
  assert.ok(sidebarSource.includes(':indeterminate="someConversationsSelected"'))
  assert.ok(sidebarSource.includes('title="批量删除对话？"'))
  assert.ok(sidebarSource.includes('这些对话中的全部问答内容都会被永久删除，且无法恢复。'))
  assert.ok(sidebarSource.includes('await chatStore.removeConversations(ids)'))
  assert.doesNotMatch(sidebarSource, /清空全部对话/)
})
