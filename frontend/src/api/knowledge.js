import http from '@/utils/request'

export const getKnowledgeBases = () => http.get('/knowledge/list')

// 文档标签去重列表；kbIds 为空则取全部可访问知识库。后端要求重复的 kb_ids 查询参数
export const getDocumentTags = (kbIds = []) => {
  const qs = kbIds.map(id => `kb_ids=${encodeURIComponent(id)}`).join('&')
  return http.get(`/knowledge/tags${qs ? `?${qs}` : ''}`)
}
export const createKnowledgeBase = (data) => http.post('/knowledge/create', data)
export const updateKnowledgeBase = (id, data) => http.put(`/knowledge/${id}`, data)
export const deleteKnowledgeBase = (id) => http.delete(`/knowledge/${id}`)
