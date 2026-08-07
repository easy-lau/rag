import http from '@/utils/request'

// 智能路由策略
export const getIntentRoutingConfig = () => http.get('/intent-routing/config')
export const updateIntentRoutingConfig = (data) => http.put('/intent-routing/config', data)

// 意图分类
export const getIntentCategories = () => http.get('/intent-routing/categories')
export const createIntentCategory = (data) => http.post('/intent-routing/categories', data)
export const updateIntentCategory = (id, data) => http.put(`/intent-routing/categories/${id}`, data)
export const deleteIntentCategory = (id) => http.delete(`/intent-routing/categories/${id}`)

// 在线调试。question 保留给旧后端，current_input/context_messages/
// selected_kb_count 是新语义合同的调试上下文；服务端只做路由，不应据此读取真实知识库。
export const testIntentRouting = (data = {}) => {
  const currentInput = String(data.current_input ?? data.question ?? '').trim()
  const contextMessages = Array.isArray(data.context_messages)
    ? data.context_messages
      .filter(item => item && ['user', 'assistant'].includes(item.role))
      .map(item => ({ role: item.role, content: String(item.content || '').trim() }))
      .filter(item => item.content)
      .slice(-6)
    : []
  const selectedKbCount = Math.min(100, Math.max(0, Math.trunc(Number(data.selected_kb_count) || 0)))
  return http.post('/intent-routing/test', {
    ...data,
    question: currentInput,
    current_input: currentInput,
    context_messages: contextMessages,
    selected_kb_count: selectedKbCount,
  })
}
