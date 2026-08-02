import http from '@/utils/request'
import { createClientRequestId, normalizeClientRequestId } from '@/utils/chatRequest'

export const getChatHistory = (params) => http.get('/chat/history', { params })
export const getMessages = (convId) => http.get(`/chat/${convId}/messages`)
export const renameConversation = (convId, title) => http.patch(`/chat/${convId}`, { title })
export const deleteConversation = (convId) => http.delete(`/chat/${convId}`)
export const deleteConversations = (conversationIds) => http.post('/chat/batch-delete', {
  conversation_ids: conversationIds,
})

export function createChatStream(payload, options = {}) {
  const ctrl = new AbortController()
  const requestId = normalizeClientRequestId(options.requestId || payload?.request_id)
    || createClientRequestId()
  const headers = {
    'Content-Type': 'application/json',
    // The body is the source of truth for older servers; the header lets
    // gateways correlate a failed stream before they parse the JSON body.
    'X-Client-Request-ID': requestId,
  }
  const token = localStorage.getItem('token')
  if (token) headers.Authorization = `Bearer ${token}`
  const requestPayload = { ...(payload || {}), request_id: requestId }
  const promise = fetch('/api/chat/send', {
    method: 'POST',
    headers,
    body: JSON.stringify(requestPayload),
    signal: ctrl.signal
  })
  return { promise, abort: () => ctrl.abort(), requestId }
}
