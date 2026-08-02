import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const routingViewSource = readFileSync(
  new URL('../src/views/admin/IntentRoutingView.vue', import.meta.url),
  'utf8',
)

test('意图路由测试页把执行层未解决项翻译成用户可读中文', () => {
  assert.match(routingViewSource, /query_execution: '查询执行条件'/)
  assert.match(routingViewSource, /missing: '缺少信息'/)
  assert.match(routingViewSource, /ambiguous: '存在歧义'/)
  assert.match(routingViewSource, /map\(unresolvedSlotLabel\)/)
})

test('意图路由测试页不直接把内部 role/reason 代码拼进澄清文案', () => {
  assert.doesNotMatch(
    routingViewSource,
    /map\(item => `\$\{item\.role \|\| '未命名'\}.*item\.reason/,
  )
})
