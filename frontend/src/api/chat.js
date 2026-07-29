import http from '@/utils/request'

export const getChatHistory = (params) => http.get('/chat/history', { params })
export const getMessages = (convId) => http.get(`/chat/${convId}/messages`)
export const renameConversation = (convId, title) => http.patch(`/chat/${convId}`, { title })
export const deleteConversation = (convId) => http.delete(`/chat/${convId}`)

export function createChatStream(payload) {
  const ctrl = new AbortController()
  const headers = { 'Content-Type': 'application/json' }
  const token = localStorage.getItem('token')
  if (token) headers.Authorization = `Bearer ${token}`
  const promise = fetch('/api/chat/send', {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: ctrl.signal
  })
  return { promise, abort: () => ctrl.abort() }
}
