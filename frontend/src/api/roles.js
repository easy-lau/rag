import http from '@/utils/request'

export const getRoles = () => http.get('/roles')
export const createRole = (data) => http.post('/roles', data)
export const updateRole = (id, data) => http.put(`/roles/${id}`, data)
export const deleteRole = (id) => http.delete(`/roles/${id}`)
export const getPermissionCatalog = () => http.get('/roles/permission-catalog')
