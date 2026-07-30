import http from '@/utils/request'

export const getSettings = () => http.get('/settings')
export const updateSettings = (data) => http.put('/settings', data)

// 使用当前表单值验证模型服务；接口不会保存或返回 API Key。
export const testModelConnection = (data) => http.post('/settings/test-connection', data)

// 读取兼容 OpenAI API 的服务商可用模型。候选 API Key 仅随本次请求发送，不会保存或回显。
export const getAvailableModels = (data) => http.post('/settings/models', data)

// 公开品牌信息（无需鉴权）
export const getSiteSettings = () => http.get('/settings/site')

// 上传站点图标（multipart），返回 { url }
export const uploadLogo = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  return http.post('/settings/logo', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}
