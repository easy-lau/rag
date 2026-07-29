import http from '@/utils/request'

// 智能路由策略
export const getIntentRoutingConfig = () => http.get('/intent-routing/config')
export const updateIntentRoutingConfig = (data) => http.put('/intent-routing/config', data)

// 意图分类
export const getIntentCategories = () => http.get('/intent-routing/categories')
export const createIntentCategory = (data) => http.post('/intent-routing/categories', data)
export const updateIntentCategory = (id, data) => http.put(`/intent-routing/categories/${id}`, data)
export const deleteIntentCategory = (id) => http.delete(`/intent-routing/categories/${id}`)

// 在线调试与路由日志
export const testIntentRouting = (data) => http.post('/intent-routing/test', data)
export const getIntentRouteLogs = (params) => http.get('/intent-routing/logs', { params })
export const submitIntentRouteFeedback = (id, feedback) =>
  http.post(`/intent-routing/logs/${id}/feedback`, { feedback })
