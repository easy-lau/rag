const MAX_PUBLIC_ERROR_CHARS = 180

function publicErrorText(value, fallback) {
  const text = typeof value === 'string'
    ? value.replace(/\s+/g, ' ').trim()
    : ''
  return (text || fallback).slice(0, MAX_PUBLIC_ERROR_CHARS)
}

export function splitCompleteSseEvents(buffer) {
  const events = String(buffer || '').split(/\r?\n\r?\n/)
  return {
    complete: events.slice(0, -1),
    remainder: events.at(-1) || '',
  }
}

export function parseSseDataEvent(rawEvent) {
  const rawData = String(rawEvent || '')
    .split(/\r?\n/)
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trimStart())
    .join('\n')
  if (!rawData) return null

  const parsed = JSON.parse(rawData)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new TypeError('SSE data must be a JSON object')
  }
  return parsed
}

export function appendUniqueStreamError(message, errorMessage) {
  if (!message || typeof message !== 'object') return false
  const normalized = publicErrorText(errorMessage, '响应处理异常，部分内容可能未显示')
  const previous = Array.isArray(message.stream_errors)
    ? message.stream_errors.filter(value => typeof value === 'string')
    : []
  if (previous.includes(normalized)) return false

  const notice = `[错误：${normalized}]`
  message.stream_errors = [...previous, normalized]
  if (!String(message.content || '').includes(notice)) {
    const current = String(message.content || '').trimEnd()
    message.content = `${current}${current ? '\n\n' : ''}${notice}`
  }
  return true
}

export function appendStreamText(message, content) {
  if (!message || typeof message !== 'object' || typeof content !== 'string') return false
  const errors = Array.isArray(message.stream_errors)
    ? message.stream_errors.filter(value => typeof value === 'string')
    : []
  if (!errors.length) {
    message.content = `${String(message.content || '')}${content}`
    return true
  }

  // A recoverable protocol error can occur between two valid deltas. Keep the
  // public notices at the bottom instead of concatenating later answer text
  // after "[错误：…]".
  const noticeBlock = errors.map(error => `[错误：${error}]`).join('\n\n')
  const current = String(message.content || '')
  let answer = current
  if (current === noticeBlock) answer = ''
  else if (current.endsWith(`\n\n${noticeBlock}`)) {
    answer = current.slice(0, -(noticeBlock.length + 2))
  }
  const nextAnswer = `${answer}${content}`
  message.content = `${nextAnswer}${nextAnswer ? '\n\n' : ''}${noticeBlock}`
  return true
}

export function publicRequestError(error) {
  if (typeof error?.publicMessage === 'string') {
    return publicErrorText(error.publicMessage, '请求失败，请稍后重试')
  }
  return error?.name === 'TypeError'
    ? '网络连接中断，请检查连接后重试'
    : '请求处理失败，请稍后重试'
}

export const SSE_PARSE_ERROR_MESSAGE = '收到无法解析的响应事件，部分内容可能未显示'
export const SSE_HANDLER_ERROR_MESSAGE = '响应事件处理异常，部分内容可能未显示'
