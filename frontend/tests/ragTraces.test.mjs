import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const traceViewSource = readFileSync(
  new URL('../src/views/admin/RagTracesView.vue', import.meta.url),
  'utf8',
)

test('调用链页面为待澄清任务过期事件提供明确中文名称', () => {
  assert.match(
    traceViewSource,
    /'intent\.clarification_expired': '待澄清任务已过期'/,
  )
})

test('待澄清任务过期使用警告语义而不是成功或错误语义', () => {
  assert.ok(traceViewSource.includes("event === 'chat.cancelled'"))
  assert.ok(traceViewSource.includes("event === 'intent.clarification_expired'"))
})

test('待澄清任务完成继续使用成功语义', () => {
  assert.ok(traceViewSource.includes("event === 'intent.clarification_resolved'"))
  assert.match(traceViewSource, /\) return 'is-success'/)
})

test('调用链页面展示证据范围澄清的创建、重复与完成事件', () => {
  for (const [event, label] of [
    ['evidence.clarification_created', '保存证据范围选项'],
    ['evidence.clarification_repeated', '重复证据范围选项'],
    ['evidence.clarification_resolved', '完成证据范围选择'],
    ['evidence.scope_filter_applied', '应用证据范围过滤'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
  assert.match(
    traceViewSource,
    /event === 'evidence\.clarification_repeated'/,
  )
  assert.match(
    traceViewSource,
    /event === 'evidence\.clarification_resolved'/,
  )
  assert.match(
    traceViewSource,
    /needs_clarification: '等待选择范围'/,
  )
})

test('调用链页面展示 V2 执行器、规划、证据覆盖和任务恢复阶段', () => {
  for (const [event, label] of [
    ['chat.pipeline_selected', '选择问答执行器'],
    ['chat.turn_reclaimed', '恢复过期问答任务'],
    ['direct.plan', '制定直答计划'],
    ['query.plan', '制定查询与证据计划'],
    ['evidence.coverage_assessed', '评估证据覆盖情况'],
    ['retrieval.expansion_error', '证据补检失败'],
    ['generation.error', '回答生成失败'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
})
