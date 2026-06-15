import http from '@/utils/request'

export const getLoginLogs = (params) => http.get('/login-logs', { params })
