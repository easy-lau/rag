import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const traceViewSource = readFileSync(
  new URL('../src/views/admin/RagTracesView.vue', import.meta.url),
  'utf8',
)
const evidenceStatusSource = readFileSync(
  new URL('../src/utils/evidenceStatus.js', import.meta.url),
  'utf8',
)

test('调用链页面为待澄清任务过期事件提供明确中文名称', () => {
  assert.match(
    traceViewSource,
    /'clarification\.expired': '待澄清状态已过期'/,
  )
})

test('待澄清任务过期使用警告语义而不是成功或错误语义', () => {
  assert.ok(traceViewSource.includes("name === 'chat.cancelled'"))
  assert.ok(traceViewSource.includes("name === 'clarification.expired'"))
})

test('待澄清任务完成继续使用成功语义', () => {
  assert.ok(traceViewSource.includes("name === 'clarification.resolved'"))
  assert.match(traceViewSource, /\) return 'is-success'/)
})

test('调用链页面展示统一澄清的创建、重复与完成事件', () => {
  for (const [event, label] of [
    ['clarification.created', '保存待澄清状态'],
    ['clarification.repeated', '重复待澄清状态'],
    ['clarification.resolved', '完成澄清'],
    ['evidence.scope_filter_applied', '应用证据范围过滤'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
  assert.match(
    traceViewSource,
    /name === 'clarification\.repeated'/,
  )
  assert.match(
    traceViewSource,
    /name === 'clarification\.resolved'/,
  )
  assert.match(traceViewSource, /evidenceStatusLabel/)
  assert.match(evidenceStatusSource, /needs_clarification:\s*\{[\s\S]*?label: '等待选择范围'/)
})

test('调用链证据状态从共享合同读取，避免后台与历史页面各自维护枚举', () => {
  assert.match(traceViewSource, /from '@\/utils\/evidenceStatus'/)
  assert.match(evidenceStatusSource, /scope_mismatch:/)
})

test('调用链页面展示 V2 执行器、规划、证据覆盖和任务恢复阶段', () => {
  for (const [event, label] of [
    ['chat.pipeline_selected', '选择问答执行器'],
    ['chat.turn_reclaimed', '恢复过期问答任务'],
    ['direct.plan', '制定直答计划'],
    ['query.plan', '制定查询与证据计划'],
    ['evidence.coverage_assessed', '评估证据覆盖情况'],
    ['evidence.unverified_fallback', '保留待验证候选上下文'],
    ['retrieval.expansion_error', '证据补检失败'],
    ['generation.error', '回答生成失败'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
})

test('通用模型兜底在调用链中使用明确名称和警告语义', () => {
  assert.ok(
    traceViewSource.includes(
      "'generation.general_fallback': '启用通用模型兜底'",
    ),
  )
  assert.match(traceViewSource, /name === 'generation\.general_fallback'/)
})

test('调用链页面展示统一查询执行闸门', () => {
  for (const [event, label] of [
    ['query.execution', '校验查询执行基线'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
})

test('查询执行闸门关闭时在调用链中使用警告语义', () => {
  assert.match(traceViewSource, /name === 'query\.execution'/)
  assert.match(traceViewSource, /event\?\.payload\?\.state === 'needs_clarification'/)
})
