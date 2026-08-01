import test from 'node:test'
import assert from 'node:assert/strict'

import {
  appendStreamText,
  appendUniqueStreamError,
  parseSseDataEvent,
  splitCompleteSseEvents,
} from '../src/utils/chatStream.js'

test('SSE 分包只交付完整事件并保留残片', () => {
  const first = splitCompleteSseEvents('data: {"type":"text_')
  assert.deepEqual(first.complete, [])
  assert.equal(first.remainder, 'data: {"type":"text_')

  const second = splitCompleteSseEvents(`${first.remainder}delta","content":"回答"}\n\ndata: {"type":"do`)
  assert.equal(second.complete.length, 1)
  assert.deepEqual(parseSseDataEvent(second.complete[0]), {
    type: 'text_delta',
    content: '回答',
  })
  assert.equal(second.remainder, 'data: {"type":"do')
})

test('SSE 支持多 data 行并拒绝非对象 JSON', () => {
  assert.deepEqual(
    parseSseDataEvent('data: {"type":"usage",\ndata: "total_tokens":12}'),
    { type: 'usage', total_tokens: 12 },
  )
  assert.throws(() => parseSseDataEvent('data: [1,2,3]'), /JSON object/)
  assert.throws(() => parseSseDataEvent('data: {not-json}'), SyntaxError)
})

test('流事件错误追加到已有回答且同类错误只显示一次', () => {
  const message = { content: '已经生成的回答' }

  assert.equal(appendUniqueStreamError(message, '响应事件处理异常'), true)
  assert.equal(appendUniqueStreamError(message, '响应事件处理异常'), false)
  assert.equal(
    message.content,
    '已经生成的回答\n\n[错误：响应事件处理异常]',
  )
  assert.deepEqual(message.stream_errors, ['响应事件处理异常'])
})

test('可恢复错误后的后续增量继续组成回答，错误提示始终留在底部', () => {
  const message = { content: '前半段' }
  appendUniqueStreamError(message, '某个事件损坏')

  assert.equal(appendStreamText(message, '后半段'), true)
  assert.equal(message.content, '前半段后半段\n\n[错误：某个事件损坏]')

  const initiallyBroken = { content: '' }
  appendUniqueStreamError(initiallyBroken, '首个事件损坏')
  appendStreamText(initiallyBroken, '后续有效回答')
  assert.equal(initiallyBroken.content, '后续有效回答\n\n[错误：首个事件损坏]')
})

test('空回答也能显示公开错误且不会暴露无限长服务端文本', () => {
  const message = { content: '' }
  appendUniqueStreamError(message, 'x'.repeat(500))

  assert.match(message.content, /^\[错误：x+\]$/)
  assert.ok(message.content.length < 220)
})
