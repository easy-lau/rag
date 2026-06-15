import http from '@/utils/request'

export const getSettings = () => http.get('/settings')
export const updateSettings = (data) => http.put('/settings', data)

// 公开品牌信息（无需鉴权）
export const getSiteSettings = () => http.get('/settings/site')

// 上传站点图标（multipart），返回 { url }
export const uploadLogo = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  return http.post('/settings/logo', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}
