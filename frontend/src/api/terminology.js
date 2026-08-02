import http from '@/utils/request'

// 受控术语注册表始终隶属于一个知识库。调用方只能传入已经由
// /knowledge/list 返回的可访问知识库 ID；对象级权限仍由后端校验。
const basePath = kbId => `/knowledge/${encodeURIComponent(kbId)}/terminology`

export const getTerminologyRegistry = (kbId, { includeInactive = true } = {}) =>
  http.get(basePath(kbId), { params: { include_inactive: includeInactive } })

export const createTerminologyConcept = (kbId, data) =>
  http.post(`${basePath(kbId)}/concepts`, data)

export const updateTerminologyConcept = (kbId, conceptId, data) =>
  http.put(`${basePath(kbId)}/concepts/${encodeURIComponent(conceptId)}`, data)

export const createTerminologyTerm = (kbId, conceptId, data) =>
  http.post(`${basePath(kbId)}/concepts/${encodeURIComponent(conceptId)}/terms`, data)

export const updateTerminologyTerm = (kbId, conceptId, termId, data) =>
  http.put(
    `${basePath(kbId)}/concepts/${encodeURIComponent(conceptId)}/terms/${encodeURIComponent(termId)}`,
    data,
  )

export const createTerminologyBinding = (kbId, data) =>
  http.post(`${basePath(kbId)}/bindings`, data)

export const updateTerminologyBinding = (kbId, bindingId, data) =>
  http.put(`${basePath(kbId)}/bindings/${encodeURIComponent(bindingId)}`, data)
