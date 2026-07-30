import http from '@/utils/request'

export const getRagTraces = params => http.get('/rag-traces', { params })
export const getRagTraceDetail = (traceId, params) => (
  http.get(`/rag-traces/${encodeURIComponent(traceId)}`, { params })
)
export const downloadRagTrace = traceId => (
  http.get(`/rag-traces/${encodeURIComponent(traceId)}/export`, {
    responseType: 'blob',
    returnFullResponse: true,
  })
)
