import http from '@/utils/request'

export const getOperationLogs = (params) => http.get('/operation-logs', { params })
