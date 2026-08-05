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
    /'intent\.clarification_expired': '待澄清任务已过期'/,
  )
})

test('待澄清任务过期使用警告语义而不是成功或错误语义', () => {
  assert.ok(traceViewSource.includes("name === 'chat.cancelled'"))
  assert.ok(traceViewSource.includes("name === 'intent.clarification_expired'"))
})

test('待澄清任务完成继续使用成功语义', () => {
  assert.ok(traceViewSource.includes("name === 'intent.clarification_resolved'"))
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
    /name === 'evidence\.clarification_repeated'/,
  )
  assert.match(
    traceViewSource,
    /name === 'evidence\.clarification_resolved'/,
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

test('调用链页面区分查询执行闸门与可选大模型结构化理解阶段', () => {
  for (const [event, label] of [
    ['query.execution', '校验查询执行基线'],
    ['query.analysis.requested', '请求大模型结构化理解'],
    ['query.analysis.validated', '校验大模型结构化理解'],
    ['query.analysis.compiled', '编译结构化检索计划'],
    ['query.analysis.execution_decision', '确定结构化理解执行结果'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
  assert.match(traceViewSource, /模型结构化理解 JSON（已通过协议校验）/)
  assert.match(traceViewSource, /后端编译后的执行计划 JSON/)
})

test('查询执行闸门关闭时在调用链中使用警告语义', () => {
  assert.match(traceViewSource, /name === 'query\.execution'/)
  assert.match(traceViewSource, /event\?\.payload\?\.state === 'needs_clarification'/)
})

test('调用链页面展示 V3 语义入口闸门，并区分放行、延后理解和硬性拦截', () => {
  assert.ok(
    traceViewSource.includes("'intent.semantic_entry_gate': '校验 V3 语义入口闸门'"),
  )
  assert.match(traceViewSource, /semanticEntryDisposition\(event\) === 'blocked'/)
  assert.match(traceViewSource, /\['dispatch', 'defer_to_v3'\]\.includes\(semanticEntryDisposition\(event\)\)/)
  assert.match(traceViewSource, /后端已重建当前轮的受限执行合同，交由 V3 选择可信片段/)
  assert.match(traceViewSource, /V3 不会绕过该入口闸门/)
})

test('调用链页面展示 V3 受限模型理解及后端可信编译阶段', () => {
  for (const [event, label] of [
    ['query.understanding.v3.requested', '请求 V3 受限 Span 结构理解'],
    ['query.understanding.v3.completed', '收到 V3 结构理解结果'],
    ['query.understanding.v3.validated', '校验 V3 Span 选择协议'],
    ['query.understanding.v3.execution_validated', '校验 V3 可信编译边界'],
    ['query.understanding.v3.compiled', '编译 V3 可信执行计划'],
    ['query.understanding.v3.execution_decision', '确定 V3 执行结果'],
    ['query.understanding.v3.fallback', 'V3 回退确定性计划'],
    ['query.understanding.v3.cancelled', '取消 V3 结构理解'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
  assert.match(traceViewSource, /服务器签发的 V3 Span 目录 JSON（开发正文）/)
  assert.match(traceViewSource, /V3 模型原始结构化响应 JSON（开发正文）/)
  assert.match(traceViewSource, /V3 Span 选择摘要（生产环境未记录正文）/)
  assert.match(traceViewSource, /后端可信编译摘要（生产环境未记录正文）/)
})

test('调用链页面说明严格本地追问 Span 的安全边界与执行状态', () => {
  assert.ok(
    traceViewSource.includes("'query.understanding.v3.deterministic_contextual_ellipsis': '解析严格本地追问 Span'"),
  )
  assert.match(traceViewSource, /deterministicContextualApplied\(event\)/)
  assert.match(traceViewSource, /deterministicContextualStatus\(event\)/)
  assert.match(traceViewSource, /已选出严格的原文 Span，正在绑定当前请求的 V3 目录/)
  assert.match(traceViewSource, /未读取助手回答、未拼接历史问题/)
  assert.match(traceViewSource, /当前追问未满足严格继承条件，系统不会猜测历史主体/)
  assert.match(traceViewSource, /selection_failed', 'binding_rejected', 'binding_failed/)
})

test('V3 编译成功、回退和取消使用符合实际执行状态的调用链语义', () => {
  assert.match(traceViewSource, /name === 'query\.understanding\.v3\.compiled'/)
  assert.match(traceViewSource, /name === 'query\.understanding\.v3\.fallback'/)
  assert.match(traceViewSource, /name === 'query\.understanding\.v3\.cancelled'/)
  assert.match(traceViewSource, /\['fallback', 'clarification', 'skipped'\]\.includes\(event\?\.payload\?\.decision\)/)
})

test('调用链页面展示 V3 请求版本围栏和原问题锚点预取，并说明其并非最终证据', () => {
  for (const [event, label] of [
    ['query.understanding.v3.revision_fence', '校验 V3 请求版本围栏'],
    ['retrieval.anchor_preflight.completed', '完成原问题锚点预取'],
    ['retrieval.anchor_preflight.reused', '复用原问题锚点预取'],
    ['retrieval.anchor_preflight.rejected', '拒绝不匹配的原问题锚点预取'],
  ]) {
    assert.ok(
      traceViewSource.includes(`'${event}': '${label}'`),
      `${event} 应有明确中文名称`,
    )
  }
  assert.match(traceViewSource, /尚未成为证据，最终任务图仍会重新校验授权、范围和相关性/)
  assert.match(traceViewSource, /通过请求版本围栏后才允许进入后续 V2 证据任务图/)
})

test('锚点预取降级和版本围栏拒绝在调用链中使用警告语义', () => {
  assert.match(traceViewSource, /name === 'retrieval\.anchor_preflight\.rejected'/)
  assert.match(traceViewSource, /\['timeout', 'unavailable'\]\.includes\(anchorPreflightStatus\(event\)\)/)
  assert.match(traceViewSource, /name === 'query\.understanding\.v3\.revision_fence'/)
  assert.match(traceViewSource, /isV3FenceRejected\(event\)/)
})
