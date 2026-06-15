import http from '@/utils/request'

export const searchTest = (data) => http.post('/search/test', data)
