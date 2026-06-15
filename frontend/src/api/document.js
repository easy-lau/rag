import http from '@/utils/request'

export const getDocuments = (kbId, params) => http.get(`/knowledge/${kbId}/documents`, { params })

// 拉取某知识库下全部文档（分批绕过后端单页上限 100），供前端做分页/筛选
export const getAllDocuments = async (kbId) => {
  const pageSize = 100
  const all = []
  for (let page = 1; ; page++) {
    const batch = await getDocuments(kbId, { page, page_size: pageSize })
    all.push(...batch)
    if (batch.length < pageSize) break
  }
  return all
}

export const uploadDocument = (kbId, file, onProgress, tags = []) => {
  const form = new FormData()
  form.append('file', file)
  if (tags.length) form.append('tags', JSON.stringify(tags))
  return http.post(`/knowledge/${kbId}/documents`, form, {
    onUploadProgress: e => onProgress?.(Math.round(e.loaded / e.total * 100))
  })
}

export const uploadImageDocument = (kbId, file, tags = []) => {
  const form = new FormData()
  form.append('file', file)
  if (tags.length) form.append('tags', JSON.stringify(tags))
  return http.post(`/knowledge/${kbId}/documents/image`, form)
}

export const createTextDocument = (kbId, title, content, sourceUrl = null, tags = []) =>
  http.post(`/knowledge/${kbId}/documents/text`, { title, content, source_url: sourceUrl, tags })

export const getDocument = (kbId, docId) => http.get(`/knowledge/${kbId}/documents/${docId}`)

export const updateTextDocument = (kbId, docId, title, content, sourceUrl = null, tags = []) =>
  http.put(`/knowledge/${kbId}/documents/${docId}`, { title, content, source_url: sourceUrl, tags })

export const updateDocumentTags = (kbId, docId, tags) =>
  http.patch(`/knowledge/${kbId}/documents/${docId}/tags`, { tags })

export const toggleDocument = (kbId, docId) => http.patch(`/knowledge/${kbId}/documents/${docId}/toggle`)

export const deleteDocument = (kbId, docId) => http.delete(`/knowledge/${kbId}/documents/${docId}`)
