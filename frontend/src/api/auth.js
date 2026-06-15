import http from '@/utils/request'

export const login = (data) => http.post('/auth/login', data)
export const getMe = () => http.get('/auth/me')
export const changePassword = (data) => http.post('/auth/change-password', data)
