import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const chatStoreSource = readFileSync(
  new URL('../src/stores/chat.js', import.meta.url),
  'utf8',
)
const chatMessageSource = readFileSync(
  new URL('../src/components/chat/ChatMessage.vue', import.meta.url),
  'utf8',
)
const chatViewSource = readFileSync(
  new URL('../src/views/ChatView.vue', import.meta.url),
  'utf8',
)
const chatInputSource = readFileSync(
  new URL('../src/components/chat/ChatInput.vue', import.meta.url),
  'utf8',
)
const searchTestSource = readFileSync(
  new URL('../src/views/SearchTestView.vue', import.meta.url),
  'utf8',
)

test('chat store 同时消费结构化事件和 search_results 内嵌澄清', () => {
  assert.ok(chatStoreSource.includes("data.type === 'evidence_clarification'"))
  assert.ok(chatStoreSource.includes("data.type === 'evidence_clarification_ack'"))
  assert.ok(chatStoreSource.includes('clarificationFromSearchEvent('))
  assert.ok(chatStoreSource.includes('attachEvidenceClarification(aiMsg'))
  assert.ok(chatStoreSource.includes('applyClarificationLifecycleEvent(aiMsg, data)'))
  assert.ok(chatStoreSource.includes('aiMsg = reactive(aiMsg)'))
})

test('澄清选项使用原生 button 并提供序号、aria 与都对比操作', () => {
  assert.match(
    chatMessageSource,
    /<button[\s\S]*v-for="choice in clarification\.choices"[\s\S]*type="button"/,
  )
  assert.ok(chatMessageSource.includes(':aria-label="`选择第 ${choice.index} 项：${choice.label}`"'))
  assert.ok(chatMessageSource.includes("selectClarification('都对比')"))
  assert.ok(chatMessageSource.includes('当前候选范围较多，无法安全列成有限选项'))
  assert.ok(chatMessageSource.includes(':disabled="!clarificationCanSubmit"'))
  assert.ok(chatMessageSource.includes('以下选项已失效。请重新提问'))
})

test('组件选择事件经 ChatView 回到 store 的统一 sendMessage 链', () => {
  assert.ok(chatMessageSource.includes("emit('clarify', { message: props.message, reply })"))
  assert.ok(chatViewSource.includes('@clarify="handleClarification"'))
  assert.ok(chatViewSource.includes('chatStore.submitClarification(message, reply)'))
  assert.ok(chatStoreSource.includes('void sendMessage(reply, {'))
  assert.ok(chatStoreSource.includes('clarificationSource: target'))
  assert.ok(chatStoreSource.includes('knowledgeBaseIds: clarification.selected_kb_ids_snapshot'))
  assert.ok(chatStoreSource.includes("content: displayQuestion"))
  assert.ok(chatStoreSource.includes("? '选择：都对比'"))
  assert.ok(chatStoreSource.includes('`选择：${selected.label}`'))
})

test('未 ack、错误、中止和无 ack done 都会保持 picker 失效', () => {
  assert.match(
    chatStoreSource,
    /data\.type === 'error'[\s\S]*applyClarificationLifecycleEvent\(aiMsg, data\)/,
  )
  assert.ok(chatStoreSource.includes("invalidateClarification(lastAssistantMessage, searchStore, 'stream_aborted')"))
  assert.ok(chatStoreSource.includes("invalidateClarification(aiMsg, searchStore, 'missing_persistence_ack')"))
  assert.ok(chatStoreSource.includes("? 'stream_aborted'"))
  assert.ok(chatStoreSource.includes(": (e.failureReason || 'request_failed')"))
})

test('clarification 锁定后乱序 search_results 不恢复消息来源或 hit 状态', () => {
  assert.ok(chatStoreSource.includes('aiMsg.sources = clarification ? [] : answerSourcesFromSearchEvent'))
  assert.ok(chatStoreSource.includes("? { ...data, evidence_status: 'needs_clarification', clarification }"))
  assert.ok(chatStoreSource.includes('lockClarificationEvidence(aiMsg, searchStore, attached)'))
})

test('历史消息通过服务端持久化状态归一化后恢复 picker', () => {
  assert.ok(chatStoreSource.includes('restoreConversationMessages(loadedMessages)'))
})

test('SSE 解析和事件异常都有公开反馈，不再使用空 catch 丢弃', () => {
  assert.ok(chatStoreSource.includes('appendUniqueStreamError(aiMsg, SSE_PARSE_ERROR_MESSAGE)'))
  assert.ok(chatStoreSource.includes('appendUniqueStreamError(aiMsg, SSE_HANDLER_ERROR_MESSAGE)'))
  assert.doesNotMatch(
    chatStoreSource,
    /handleEvent\(JSON\.parse\(rawData\)[\s\S]{0,80}catch \{\}/,
  )
})

test('Trace 诊断、历史检索入口和响应式键盘语义完整', () => {
  assert.ok(chatMessageSource.includes('aria-label="复制错误追踪 ID"'))
  assert.ok(chatMessageSource.includes('aria-label="查看本条回答的检索摘要"'))
  assert.ok(chatMessageSource.includes("$emit('inspect', message)"))
  assert.ok(chatMessageSource.includes('.message-trace'))
  assert.ok(chatMessageSource.includes('var(--ui-danger-subtle)'))
  assert.ok(chatMessageSource.includes('@media (max-width: 639px)'))
  assert.ok(chatMessageSource.includes(':focus-visible'))
  assert.ok(chatViewSource.includes("useMediaQuery('(min-width: 1280px)')"))
  assert.ok(chatViewSource.includes(':width="ui.isMobile ? 320 : 360"'))
  assert.ok(chatViewSource.includes('@inspect="inspectMessageSearch"'))
})

test('保存失败和处理中回放使用同一 request id 恢复而不是重复生成', () => {
  assert.ok(chatStoreSource.includes('isPersistenceFailureEvent(data)'))
  assert.ok(chatStoreSource.includes('isPendingTurnReplay(data)'))
  assert.ok(chatStoreSource.includes("aiMsg.failure_reason = 'turn_in_progress'"))
  assert.ok(chatStoreSource.includes("没有可恢复的已生成回答"))
  assert.ok(chatStoreSource.includes("aiMsg.replayed === true"))
  assert.ok(chatStoreSource.includes('restoreConversationMessages(replayedMessages)'))
  assert.ok(chatStoreSource.includes('shouldReloadCompletedEmptyAssistant(aiMsg'))
  assert.ok(chatStoreSource.includes('String(currentConvId.value) === authoritativeConversationId'))
  assert.ok(chatMessageSource.includes("return '获取结果'"))
  assert.ok(chatMessageSource.includes("return '重新发送'"))
  assert.ok(chatMessageSource.includes("return '恢复回答'"))
  const handleEventIndex = chatStoreSource.indexOf('handleEvent(data, aiMsg, searchStore, runId')
  const presentationConfirmationIndex = chatStoreSource.indexOf(
    'streamEventConfirmsAssistantPresentation(data)',
    handleEventIndex,
  )
  assert.ok(handleEventIndex >= 0 && presentationConfirmationIndex > handleEventIndex)
  assert.ok(chatStoreSource.includes('!sawDone && !sawCompletedTurnState'))
  assert.ok(chatStoreSource.includes('(aiMsg.replayed === true || (!sawDone && sawCompletedTurnState))'))
})

test('空回答 spinner 仅由活跃 request 状态驱动，终态与失败态使用静态说明', () => {
  assert.ok(chatMessageSource.includes('emptyAssistantPresentation(props.message'))
  assert.ok(chatMessageSource.includes("emptyPresentation.kind === 'thinking'"))
  assert.ok(chatMessageSource.includes("emptyPresentation.kind !== 'hidden'"))
  assert.doesNotMatch(chatMessageSource, /<div v-else class="flex items-center gap-2 text-gray-400 text-sm">\s*<n-spin/)
  assert.ok(chatStoreSource.includes("activeRequestId.value = requestId"))
  assert.ok(chatStoreSource.includes("if (activeRequestId.value === requestId) activeRequestId.value = ''"))
})

test('回答依据仅使用 answer_support，不把召回排序伪装成匹配率', () => {
  assert.match(
    chatMessageSource,
    /function sourceSupportScore\(source\) \{[\s\S]*return finiteScore\(source\.answer_support\)\n\}/,
  )
  assert.doesNotMatch(
    chatMessageSource,
    /return finiteScore\(source\.answer_support\) \?\? finiteScore\(source\.effective_score\)/,
  )
  assert.ok(chatMessageSource.includes("const level = score === null ? '已验证'"))
})

test('聊天与检索测试不再提供或发送用户标签检索偏好', () => {
  assert.doesNotMatch(chatInputSource, /标签筛选|searchConfig\.tags|getDocumentTags|PricetagOutline/)
  assert.doesNotMatch(searchTestSource, /config\.tags|getDocumentTags|标签（软加权|标签筛选/)
  assert.doesNotMatch(chatStoreSource, /searchConfig = ref\([^\n]*tags:/)
})

test('助手消息在底部展示本次思考回答耗时，流式完成后写入时长', () => {
  assert.ok(chatMessageSource.includes('思考回答耗时 {{ responseDurationText }}'))
  assert.ok(chatMessageSource.includes('const responseDurationText = computed'))
  assert.ok(chatStoreSource.includes('duration_ms: null'))
  assert.ok(chatStoreSource.includes('answer_started_at_ms: Date.now()'))
  assert.ok(chatStoreSource.includes('const serverDuration = Number(data.duration_ms)'))
})
