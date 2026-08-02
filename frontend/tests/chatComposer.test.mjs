import test from 'node:test'
import assert from 'node:assert/strict'

import { normalizeSingleLinePaste } from '../src/utils/chatComposer.js'

test('单行问题复制时的块级边界换行会被清理', () => {
  assert.equal(
    normalizeSingleLinePaste('\n\n我现在想二开发送钉钉工作通知\n'),
    '我现在想二开发送钉钉工作通知',
  )
  assert.equal(
    normalizeSingleLinePaste('\r\n查询差旅标准\r\n'),
    '查询差旅标准',
  )
})

test('真实多行粘贴保留原样', () => {
  const multiline = '第一步：配置接口\n第二步：发送通知'
  assert.equal(normalizeSingleLinePaste(multiline), multiline)
})

test('非字符串内容不做转换', () => {
  assert.equal(normalizeSingleLinePaste(null), null)
})
