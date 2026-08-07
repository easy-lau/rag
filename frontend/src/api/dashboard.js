import http from '@/utils/request'

export const getDashboardOverview = (days = 7) =>
  http.get('/admin/dashboard/overview', { params: { days } })

export const getDashboardAiReport = (days = 7) =>
  http.get('/admin/dashboard/report', { params: { days } })
