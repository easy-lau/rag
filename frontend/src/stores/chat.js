import { defineStore } from 'pinia'
import { reactive, ref, watch } from 'vue'
import {
  createChatStream,
  deleteConversation,
  deleteConversations as deleteConversationsRequest,
  getChatHistory,
  getMessages,
  renameConversation as renameConversationRequest,
} from '@/api/chat'
import { useSearchStore } from './search'
import { answerSourcesFromSearchEvent } from '@/utils/chatEvidence'
import {
  restoreConversationMessages,
  restoreHistoryMessageState,
  searchSnapshotFromEvent,
  searchSnapshotFromHistoryMessage,
} from '@/utils/chatHistory'
import {
  applyClarificationLifecycleEvent,
  attachEvidenceClarification,
  clarificationFromSearchEvent,
  invalidateEvidenceClarification,
  isClarificationActive,
  isClarificationSubmittable,
  lockMessageClarificationEvidence,
  markClarificationSubmitted,
  normalizeEvidenceClarification,
  restoreClarificationSubmissionForRetry,
} from '@/utils/chatClarification'
import {
  captureAssistantPresentationForRetry,
  chatRequestHttpFailure,
  coalesceLogicalTurnMessages,
  createClientRequestId,
  normalizeClientRequestId,
  normalizeTraceId,
  normalizeTurnId,
  resetAssistantForLogicalRetry,
  responseHeader,
  restoreAssistantPresentationAfterUnconfirmedRetry,
  streamEventConfirmsAssistantPresentation,
  traceIdFromResponse,
} from '@/utils/chatRequest'
import {
  applyTurnLifecycleState,
  shouldReloadCompletedEmptyAssistant,
  eventConfirmsPendingClarification,
  isPendingTurnReplay,
  isPersistenceFailureEvent,
} from '@/utils/chatTurnState'
import {
  appendStreamText,
  appendUniqueStreamError,
  parseSseDataEvent,
  publicRequestError,
  splitCompleteSseEvents,
  SSE_HANDLER_ERROR_MESSAGE,
  SSE_PARSE_ERROR_MESSAGE,
} from '@/utils/chatStream'
import { fetchAllConversationPages } from '@/utils/chatHistoryPagination'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const conversations = ref([])
  const currentConvId = ref(null)
  const isStreaming = ref(false)
  const activeRequestId = ref('')
  const isConversationLoading = ref(false)
  // 加载失败时保留当前会话 ID 与深链，供页面提供明确的重试/返回选择；
  // 不能把网络或服务端临时错误误当成“会话不存在”。
  const conversationLoadError = ref(null)
  const searchConfig = ref({ method: 'hybrid', rerank: true, top_k: 5 })

  const _savedKbIds = localStorage.getItem('selectedKbIds')
  const selectedKbIds = ref(_savedKbIds ? JSON.parse(_savedKbIds) : [])

  watch(selectedKbIds, val => localStorage.setItem('selectedKbIds', JSON.stringify(val)), { deep: true })

  let abortFn = null
  let aborted = false
  // 用户快速切换历史对话时，只接收最后一次请求的响应，避免旧响应覆盖当前会话。
  let conversationRequestId = 0
  // 侧栏可能由桌面/移动布局同时触发加载。只有最后一次完整分页结果可以
  // 覆盖当前列表，避免较早请求在后返回时截断已经加载到的历史会话。
  let historyRequestId = 0
  // 停止生成后使旧 SSE 事件失效，避免其 done 事件把已新建的空会话切回旧会话。
  let streamRunId = 0
  // Only one stream can be active.  Keeping its source outside the async
  // stack lets Stop/Abort restore a choice immediately without allowing an
  // older stream callback to revive it later.
  let activeTurnContext = null

  function latestPendingClarificationMessage() {
    return [...messages.value].reverse().find(message => {
      if (message?.role !== 'assistant') return false
      const clarification = normalizeEvidenceClarification(message.clarification)
      return isClarificationActive(clarification)
    }) || null
  }

  function lockClarificationEvidence(aiMsg, searchStore, clarification) {
    const lockedClarification = lockMessageClarificationEvidence(aiMsg, clarification)
    if (lockedClarification) searchStore.setClarification(lockedClarification)
  }

  function invalidateClarification(aiMsg, searchStore, reason) {
    const clarification = invalidateEvidenceClarification(aiMsg, reason)
    if (clarification) lockClarificationEvidence(aiMsg, searchStore, clarification)
    return clarification
  }

  function bindTraceId(aiMsg, searchStore, value) {
    const traceId = normalizeTraceId(value)
    if (!traceId) return ''
    aiMsg.trace_id = traceId
    aiMsg.search_meta = { ...(aiMsg.search_meta || {}), trace_id: traceId }
    if (aiMsg.search_snapshot) aiMsg.search_snapshot.trace_id = traceId
    searchStore.setTraceId(traceId)
    return traceId
  }

  function restoreSubmittedClarification(
    context,
    searchStore,
    reason,
    { reuseRequestId = true } = {},
  ) {
    const clarification = restoreClarificationSubmissionForRetry(
      context?.clarificationSource,
      reuseRequestId ? context?.requestId : '',
      reason,
    )
    if (clarification) lockClarificationEvidence(context.clarificationSource, searchStore, clarification)
    return clarification
  }

  function markTurnFailed(aiMsg, context, searchStore, reason) {
    applyTurnLifecycleState(aiMsg, {
      type: 'error',
      turn_status: 'failed',
      persistence_status: reason === 'persistence_failed' ? 'failed' : 'unknown',
    })
    aiMsg.failure_reason = reason
    // Do not reopen a picker solely because fetch/SSE failed.  The pending
    // route may have been consumed server-side; recovery verifies it below.
  }

  function maybeRestoreClarificationFromEvent(context, searchStore, event, reason) {
    const sourceClarification = normalizeEvidenceClarification(context?.clarificationSource?.clarification)
    if (!eventConfirmsPendingClarification(event, sourceClarification)) return false
    return Boolean(restoreSubmittedClarification(context, searchStore, reason))
  }

  async function recoverClarificationFromHistory(
    context,
    searchStore,
    reason,
    { reuseRequestId = true } = {},
  ) {
    const source = context?.clarificationSource
    const attempted = normalizeEvidenceClarification(source?.clarification)
    const conversationId = currentConvId.value == null ? '' : String(currentConvId.value)
    if (!source || !attempted?.submission_pending || !conversationId) return false
    try {
      const rows = await getMessages(conversationId)
      const authoritative = (Array.isArray(rows) ? rows : [])
        .map(restoreHistoryMessageState)
        .find(message => {
          const clarification = normalizeEvidenceClarification(message?.clarification)
          return Boolean(
            clarification
            && isClarificationActive(clarification)
            && clarification.pending_state_id === attempted.pending_state_id
            && clarification.clarification_message_id === attempted.clarification_message_id
            && clarification.route_state_revision === attempted.route_state_revision
          )
        })
      if (!authoritative) {
        source.clarification = {
          ...attempted,
          submitted: false,
          submitted_reply: null,
          submission_pending: false,
          invalidated: true,
          invalid_reason: 'pending_state_not_active',
        }
        lockClarificationEvidence(source, searchStore, source.clarification)
        return false
      }
      // The history endpoint has rechecked current KB/document permissions and
      // confirmed the same pending revision.  Only now is it safe to reopen.
      source.clarification = {
        ...authoritative.clarification,
        submitted: true,
        submitted_reply: attempted.submitted_reply,
        submission_pending: true,
        submission_request_id: reuseRequestId ? context.requestId : null,
        last_submission_request_id: reuseRequestId
          ? authoritative.clarification?.last_submission_request_id || null
          : null,
      }
      return Boolean(restoreSubmittedClarification(
        context,
        searchStore,
        reason,
        { reuseRequestId },
      ))
    } catch (error) {
      console.warn('[chat] 无法复验待澄清状态', { conversationId, error })
      return false
    }
  }

  async function sendMessage(question, options = {}) {
    const normalizedQuestion = typeof question === 'string' ? question.trim() : ''
    if (isStreaming.value || !normalizedQuestion) return
    const displayQuestion = typeof options.displayContent === 'string'
      ? options.displayContent.trim() || normalizedQuestion
      : normalizedQuestion
    const runId = ++streamRunId
    const requestedRequestId = normalizeClientRequestId(options.requestId)
    const requestId = requestedRequestId || createClientRequestId()
    const logicalTurn = requestedRequestId
      ? coalesceLogicalTurnMessages(messages.value, requestId)
      : null
    if (logicalTurn?.removedCount) messages.value = logicalTurn.messages
    const priorTurn = requestedRequestId
      ? [...messages.value].reverse().find(message => (
          (
            normalizeClientRequestId(message?.request_id)
            || normalizeClientRequestId(message?.server_request_id)
          ) === requestId
          && normalizeTurnId(message?.turn_id)
        ))
      : null
    const turnId = requestedRequestId
      ? (normalizeTurnId(options.turnId) || normalizeTurnId(priorTurn?.turn_id))
      : ''

    // Any new user turn closes the currently displayed picker. A button click
    // marks it first; a manually typed selection/new question reaches the same
    // state here, so stale choices cannot be submitted again later.
    const explicitSource = options.clarificationSource
      ? messages.value.find(item => item === options.clarificationSource
        || (options.clarificationSource?.id != null && item?.id === options.clarificationSource.id))
      : null
    const pendingClarification = explicitSource || latestPendingClarificationMessage()
    let clarificationSource = null
    if (pendingClarification) {
      const marked = markClarificationSubmitted(
        pendingClarification,
        normalizedQuestion,
        {
          allowFreeText: !explicitSource || options.allowFreeText === true,
          requestId,
        },
      )
      if (!marked && explicitSource) return
      if (marked) clarificationSource = pendingClarification
    }
    const clarificationIdentity = normalizeEvidenceClarification(clarificationSource?.clarification)
    const turnContext = { requestId, clarificationSource }
    activeTurnContext = turnContext
    activeRequestId.value = requestId

    const searchStore = useSearchStore()
    searchStore.resetSteps()
    isStreaming.value = true
    aborted = false

    const now = new Date()
    const createUserMessage = () => ({
      id: `client-user-${requestId}-${runId}`,
      role: 'user',
      // c1/c2 are transport tokens, never user-facing conversation text.
      content: displayQuestion,
      request_id: requestId,
      delivery_status: 'sent',
      created_at: now,
    })
    const createAssistantMessage = () => ({
      id: `client-assistant-${requestId}-${runId}`,
      role: 'assistant',
      content: '',
      sources: [],
      stopped: false,
      tokens: null,
      clarification: null,
      stream_errors: [],
      request_id: requestId,
      trace_id: null,
      evidence_status: null,
      answer_provenance: null,
      retrieval_executed: null,
      delivery_status: 'streaming',
      persistence_status: 'pending',
      duration_ms: null,
      answer_started_at_ms: Date.now(),
      retryable: false,
      search_snapshot: null,
      search_meta: null,
      clarification_parent_message_id: clarificationSource?.id || null,
      created_at: now,
    })
    let userMsg = logicalTurn?.userMessage || null
    let aiMsg = logicalTurn?.assistantMessage || null
    const reusedAssistant = Boolean(aiMsg)
    if (requestedRequestId && (userMsg || aiMsg)) {
      const nextMessages = [...messages.value]
      if (!userMsg) {
        userMsg = createUserMessage()
        const assistantIndex = nextMessages.indexOf(aiMsg)
        nextMessages.splice(assistantIndex >= 0 ? assistantIndex : nextMessages.length, 0, userMsg)
      }
      if (!aiMsg) {
        aiMsg = createAssistantMessage()
        const userIndex = nextMessages.indexOf(userMsg)
        nextMessages.splice(userIndex >= 0 ? userIndex + 1 : nextMessages.length, 0, aiMsg)
      }
      messages.value = nextMessages
    } else {
      userMsg = createUserMessage()
      aiMsg = createAssistantMessage()
      messages.value.push(userMsg, aiMsg)
    }

    // `ref([])` exposes array entries as reactive proxies, but the local
    // variable created above still points at the original plain object.  SSE
    // writes through that raw reference do not invalidate computed consumers
    // such as ChatMessage's clarification picker.  Reuse Vue's cached proxy so
    // live clarification -> ack -> done mutations render immediately, just as
    // the same message does after a history reload.
    aiMsg = reactive(aiMsg)

    const requestKnowledgeBaseIds = Array.isArray(options.knowledgeBaseIds)
      ? options.knowledgeBaseIds
      : selectedKbIds.value
    const { promise, abort } = createChatStream({
      question: normalizedQuestion,
      conversation_id: currentConvId.value,
      knowledge_base_ids: requestKnowledgeBaseIds,
      search_config: searchConfig.value,
      request_id: requestId,
      turn_id: turnId || undefined,
      pending_route_revision: clarificationIdentity?.route_state_revision ?? undefined,
      pending_state_id: clarificationIdentity?.pending_state_id || undefined,
    }, { requestId })
    abortFn = abort
    let sawDone = false
    let sawCompletedTurnState = false
    let presentationConfirmed = false
    let retryPresentation = null
    let retryPresentationActive = false
    const restoreUnconfirmedAssistant = () => {
      if (!retryPresentationActive || presentationConfirmed) return false
      const currentErrors = Array.isArray(aiMsg.stream_errors) ? [...aiMsg.stream_errors] : []
      const restored = restoreAssistantPresentationAfterUnconfirmedRetry(aiMsg, retryPresentation)
      retryPresentationActive = false
      if (restored) currentErrors.forEach(error => appendUniqueStreamError(aiMsg, error))
      return restored
    }
    turnContext.restoreUnconfirmedAssistant = restoreUnconfirmedAssistant
    turnContext.hasConfirmedPresentation = () => presentationConfirmed

    try {
      const res = await promise
      const responseTraceId = traceIdFromResponse(res)
      const responseTurnId = responseHeader(res, 'X-RAG-Turn-ID')
      const responseRequestId = normalizeClientRequestId(responseHeader(res, 'X-RAG-Request-ID')) || requestId
      if (!res.ok) {
        let body = null
        try {
          body = await res.json()
        } catch {}
        const failure = chatRequestHttpFailure(res.status, body)
        bindTraceId(aiMsg, searchStore, responseTraceId)
        aiMsg.turn_id = responseTurnId || aiMsg.turn_id || null
        aiMsg.server_request_id = responseRequestId
        const publicError = new Error('chat request failed')
        publicError.publicMessage = failure.publicMessage
        publicError.httpStatus = failure.status
        publicError.errorDetail = failure.detail
        publicError.errorCode = failure.errorCode
        publicError.failureReason = failure.failureReason
        publicError.retryWithNewRequestId = failure.retryWithNewRequestId
        publicError.sameRequestRecoverable = failure.sameRequestRecoverable
        throw publicError
      }

      // Do not erase a visible answer while fetch is still unconfirmed. Once
      // the server accepts the replay, clear transient output in place; if the
      // connection then dies before an authoritative replacement event, the
      // snapshot below is restored so the prior answer never disappears.
      if (reusedAssistant) {
        retryPresentation = captureAssistantPresentationForRetry(aiMsg)
        retryPresentationActive = Boolean(retryPresentation)
        resetAssistantForLogicalRetry(aiMsg, requestId)
      }
      bindTraceId(aiMsg, searchStore, responseTraceId)
      aiMsg.turn_id = responseTurnId || aiMsg.turn_id || null
      aiMsg.server_request_id = responseRequestId

      // 新会话在后端已提交；先从响应头绑定 ID，避免用户刚开始生成就停止时丢失会话上下文。
      const startedConversationId = res.headers.get('X-Conversation-ID')
      if (startedConversationId && runId === streamRunId) currentConvId.value = startedConversationId

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let sseBuffer = ''

      // fetch 的每个 read() 不保证对应一条完整 SSE 事件；保留残片，避免 JSON 被拆包后静默丢失。
      const processSseEvent = (rawEvent) => {
        let data = null
        try {
          data = parseSseDataEvent(rawEvent)
        } catch (error) {
          console.warn('[chat] SSE 数据解析失败', { error })
          invalidateClarification(aiMsg, searchStore, 'protocol_parse_error')
          markTurnFailed(aiMsg, turnContext, searchStore, 'protocol_parse_error')
          appendUniqueStreamError(aiMsg, SSE_PARSE_ERROR_MESSAGE)
          return
        }
        if (!data) return
        try {
          handleEvent(data, aiMsg, searchStore, runId, turnContext, () => { sawDone = true })
          if (
            data.type === 'turn_state'
            && String(data.turn_status ?? data.status ?? '').trim().toLowerCase() === 'completed'
          ) {
            sawCompletedTurnState = true
          }
          if (streamEventConfirmsAssistantPresentation(data)) {
            presentationConfirmed = true
            retryPresentationActive = false
          }
        } catch (error) {
          console.error('[chat] SSE 事件处理失败', { eventType: data.type || 'unknown', error })
          invalidateClarification(aiMsg, searchStore, 'protocol_handler_error')
          markTurnFailed(aiMsg, turnContext, searchStore, 'protocol_handler_error')
          appendUniqueStreamError(aiMsg, SSE_HANDLER_ERROR_MESSAGE)
        }
      }
      const flushCompleteSseEvents = () => {
        const { complete, remainder } = splitCompleteSseEvents(sseBuffer)
        sseBuffer = remainder
        complete.forEach(processSseEvent)
      }

      while (true) {
        const { done, value } = await reader.read()
        if (value) {
          sseBuffer += decoder.decode(value, { stream: true })
          flushCompleteSseEvents()
        }
        if (done) {
          sseBuffer += decoder.decode()
          if (sseBuffer.trim()) processSseEvent(sseBuffer)
          break
        }
      }
    } catch (e) {
      if (runId !== streamRunId) return
      restoreUnconfirmedAssistant()
      const failureReason = e.name === 'AbortError'
        ? 'stream_aborted'
        : (e.failureReason || 'request_failed')
      markTurnFailed(
        aiMsg,
        turnContext,
        searchStore,
        failureReason,
      )
      if (Number.isFinite(Number(e.httpStatus))) aiMsg.http_status = Number(e.httpStatus)
      if (e.errorDetail !== undefined) aiMsg.error_detail = e.errorDetail
      if (typeof e.errorCode === 'string' && e.errorCode) aiMsg.error_code = e.errorCode
      if (typeof e.sameRequestRecoverable === 'boolean') {
        aiMsg.same_request_recoverable = e.sameRequestRecoverable
      }
      if (e.retryWithNewRequestId === true) {
        aiMsg.retry_with_new_request_id = true
        aiMsg.same_request_recoverable = false
      }
      invalidateClarification(
        aiMsg,
        searchStore,
        failureReason,
      )
      if (e.name !== 'AbortError') appendUniqueStreamError(aiMsg, publicRequestError(e))
    } finally {
      if (runId !== streamRunId) return
      restoreUnconfirmedAssistant()
      if (!sawDone && !sawCompletedTurnState && !aborted && !aiMsg.retryable) {
        markTurnFailed(aiMsg, turnContext, searchStore, 'stream_incomplete')
        appendUniqueStreamError(aiMsg, '响应未完整结束，请重试')
      }
      if (!aborted && aiMsg.retryable && !String(aiMsg.content || '').trim()) {
        const retryNotice = aiMsg.failure_reason === 'persistence_unrecoverable'
          ? '请求未能完成，且没有可恢复的已生成回答。请重新发送。'
          : (aiMsg.failure_reason === 'persistence_failed'
              ? '回答保存尚未完成，请点击“恢复回答”。'
              : (aiMsg.failure_reason === 'request_conflict'
                  ? '请求上下文已变化，请点击“重新发送”。'
                  : '请求未能完成，请重试。'))
        appendUniqueStreamError(aiMsg, retryNotice)
      }
      const clarification = normalizeEvidenceClarification(aiMsg.clarification)
      if (clarification && !clarification.acknowledged && !clarification.invalidated) {
        invalidateClarification(aiMsg, searchStore, 'missing_persistence_ack')
      }
      if (
        aiMsg.retryable
        && aiMsg.failure_reason !== 'turn_in_progress'
        && turnContext.clarificationSource
      ) {
        await recoverClarificationFromHistory(
          turnContext,
          searchStore,
          aiMsg.failure_reason || 'turn_failed',
          { reuseRequestId: aiMsg.retry_with_new_request_id !== true },
        )
      }
      isStreaming.value = false
      abortFn = null
      searchStore.finishSteps()
      if (aborted) {
        // 用户主动停止：标记为已停止，模板据此显示"已停止生成"，避免一直卡在"思考中"
        aiMsg.stopped = true
        aiMsg.delivery_status = 'stopped'
        aborted = false
      } else {
        const shouldReloadCompletedTurn = shouldReloadCompletedEmptyAssistant(aiMsg, {
          sawDone,
          sawCompletedTurnState,
        })
        if (
          (
            shouldReloadCompletedTurn
            || (
              (aiMsg.replayed === true || (!sawDone && sawCompletedTurnState))
              && aiMsg.turn_status === 'completed'
            )
          )
          && currentConvId.value
        ) {
          // A completed replay is authoritative, but its compact SSE does not
          // necessarily repeat an active clarification payload. Rehydrate the
          // transcript so duplicate local bubbles disappear and any currently
          // authorized picker/search snapshot is restored from durable state.
          try {
            const authoritativeConversationId = String(currentConvId.value)
            const replayedMessages = await getMessages(authoritativeConversationId)
            if (
              runId === streamRunId
              && String(currentConvId.value) === authoritativeConversationId
            ) {
              messages.value = Array.isArray(replayedMessages)
                ? restoreConversationMessages(replayedMessages)
                : messages.value
              restoreLatestMessageSearch()
            }
          } catch (error) {
            console.warn('[chat] 已完成回答无法刷新权威历史，保留当前消息状态', { error })
          }
        }
        await loadHistory().catch(() => {})
      }
      if (activeTurnContext === turnContext) activeTurnContext = null
      if (activeRequestId.value === requestId) activeRequestId.value = ''
    }
  }

  function handleEvent(data, aiMsg, searchStore, runId, turnContext = {}, markDone = () => {}) {
    if (runId !== streamRunId) return
    bindTraceId(aiMsg, searchStore, data.trace_id || data.search_meta?.trace_id || data.meta?.trace_id)
    if (data.type === 'conversation_started') {
      if (data.conversation_id) currentConvId.value = data.conversation_id
      if (data.turn_id) aiMsg.turn_id = data.turn_id
    } else if (data.type === 'turn_state') {
      if (data.turn_id) aiMsg.turn_id = data.turn_id
      if (data.request_id) aiMsg.server_request_id = data.request_id
      if (typeof data.error_code === 'string' && data.error_code) aiMsg.error_code = data.error_code
      if (data.evidence_status) aiMsg.evidence_status = data.evidence_status
      if (typeof data.retrieval_executed === 'boolean') aiMsg.retrieval_executed = data.retrieval_executed
      const serverDuration = Number(data.duration_ms)
      if (Number.isFinite(serverDuration) && serverDuration >= 0) {
        aiMsg.duration_ms = Math.round(serverDuration)
      }
      applyTurnLifecycleState(aiMsg, data)
      if (aiMsg.retryable) {
        aiMsg.failure_reason = data.status === 'persist_failed' ? 'persistence_failed' : 'server_error'
      }
    } else if (data.type === 'intent') {
      aiMsg.intent_decision = data.decision || data
      searchStore.setIntentDecision(aiMsg.intent_decision)
    } else if (data.type === 'search_process') {
      searchStore.setProcessPlan(data)
    } else if (data.type === 'search_step') {
      searchStore.updateStep(data.step, data.status, data.reason)
    } else if (data.type === 'search_results') {
      // 搜索事件现在会返回实际执行与证据状态；搜索配置只作为旧版本接口
      // 未携带 method/top_k 时的展示兜底，不能代替服务端执行结论。
      const existingClarification = normalizeEvidenceClarification(aiMsg.clarification)
      const incomingClarification = clarificationFromSearchEvent(data)
      const clarification = existingClarification || incomingClarification
      searchStore.setResults(
        clarification
          ? { ...data, evidence_status: 'needs_clarification', clarification }
          : data,
        searchConfig.value,
      )
      // 右侧面板展示完整 results；回答卡片只绑定真正进入生成上下文的
      // answer_sources。旧协议由工具函数按证据状态和 context 数量保守兼容。
      aiMsg.sources = clarification ? [] : answerSourcesFromSearchEvent(data, 20)
      const eventMeta = data.search_meta || data.meta || {}
      aiMsg.retrieval_executed = data.retrieval_executed ?? eventMeta.retrieval_executed
      aiMsg.evidence_status = clarification
        ? 'needs_clarification'
        : (data.evidence_status ?? eventMeta.evidence_status)
      aiMsg.answer_provenance = clarification
        ? null
        : (data.answer_provenance ?? eventMeta.answer_provenance ?? null)
      aiMsg.search_meta = {
        ...(aiMsg.search_meta || {}),
        ...eventMeta,
        retrieval_executed: aiMsg.retrieval_executed,
        evidence_status: aiMsg.evidence_status,
        answer_provenance: aiMsg.answer_provenance,
      }
      aiMsg.search_snapshot = searchSnapshotFromEvent(data, {
        trace_id: aiMsg.trace_id,
        evidence_status: aiMsg.evidence_status,
        answer_provenance: aiMsg.answer_provenance,
        retrieval_executed: aiMsg.retrieval_executed,
        intent_decision: aiMsg.intent_decision,
      })
      if (clarification) {
        const attached = attachEvidenceClarification(aiMsg, clarification)
        lockClarificationEvidence(aiMsg, searchStore, attached)
      }
    } else if (data.type === 'evidence_clarification') {
      const clarification = applyClarificationLifecycleEvent(aiMsg, data)
      if (clarification) lockClarificationEvidence(aiMsg, searchStore, clarification)
    } else if (data.type === 'evidence_clarification_ack') {
      const ackConversationId = typeof data.conversation_id === 'string'
        ? data.conversation_id.trim()
        : ''
      const currentConversationId = currentConvId.value == null
        ? ''
        : String(currentConvId.value)
      if (currentConversationId && ackConversationId !== currentConversationId) {
        invalidateClarification(aiMsg, searchStore, 'ack_conversation_mismatch')
        return
      }
      const clarification = applyClarificationLifecycleEvent(aiMsg, data)
      if (clarification) {
        if (!currentConversationId) currentConvId.value = ackConversationId
        lockClarificationEvidence(aiMsg, searchStore, clarification)
      }
    } else if (data.type === 'text_delta') {
      if (typeof data.content !== 'string') throw new TypeError('text_delta.content must be a string')
      appendStreamText(aiMsg, data.content)
    } else if (data.type === 'usage') {
      aiMsg.tokens = data.total_tokens
    } else if (data.type === 'done') {
      markDone()
      const serverDuration = Number(data.duration_ms)
      const localStartedAt = Number(aiMsg.answer_started_at_ms)
      if (Number.isFinite(serverDuration) && serverDuration >= 0) {
        aiMsg.duration_ms = Math.round(serverDuration)
      } else if (Number.isFinite(localStartedAt) && localStartedAt > 0) {
        aiMsg.duration_ms = Math.max(0, Date.now() - localStartedAt)
      }
      if (data.conversation_id) currentConvId.value = data.conversation_id
      const clarification = applyClarificationLifecycleEvent(aiMsg, data)
      if (clarification) lockClarificationEvidence(aiMsg, searchStore, clarification)
      applyTurnLifecycleState(aiMsg, data)
      if (isPendingTurnReplay(data)) {
        aiMsg.failure_reason = 'turn_in_progress'
        aiMsg.retryable = true
        if (!String(aiMsg.content || '').trim()) {
          aiMsg.content = '同一请求仍在处理中，请稍后点击“获取结果”。'
        }
      }
      if (aiMsg.retryable) {
        maybeRestoreClarificationFromEvent(turnContext, searchStore, data, 'turn_failed')
      }
      searchStore.finishSteps()
    } else if (data.type === 'error') {
      searchStore.failActiveStep(data.message || '')
      // 正常管线会先发送 search_results；若异常发生得更早，则显式标记为
      // “检索状态失败”，避免结果面板一直误显示为等待中。
      if (!searchStore.hasResultEvent) {
        searchStore.setResults({
          results: [],
          total: 0,
          retrieval_executed: null,
          evidence_status: 'error',
          decision_reason: searchStore.intentDecision?.decision_reason || '',
        }, searchConfig.value)
      }
      const clarification = applyClarificationLifecycleEvent(aiMsg, data)
      if (clarification) lockClarificationEvidence(aiMsg, searchStore, clarification)
      if (typeof data.error_code === 'string' && data.error_code) aiMsg.error_code = data.error_code
      const persistenceFailed = isPersistenceFailureEvent(data)
      applyTurnLifecycleState(aiMsg, persistenceFailed
        ? { ...data, turn_status: 'persist_failed', persistence_status: 'failed' }
        : data)
      aiMsg.failure_reason = persistenceFailed
        ? (data.same_request_recoverable === false ? 'persistence_unrecoverable' : 'persistence_failed')
        : 'server_error'
      maybeRestoreClarificationFromEvent(turnContext, searchStore, data, aiMsg.failure_reason)
      appendUniqueStreamError(aiMsg, data.message || '服务端处理失败，请稍后重试')
    }
  }

  function submitClarification(message, reply) {
    if (isStreaming.value) return false
    const target = messages.value.find(item => (
      item === message || (message?.id != null && item?.id === message.id)
    ))
    if (!target || target !== latestPendingClarificationMessage()) return false
    const clarification = normalizeEvidenceClarification(target.clarification)
    if (!isClarificationSubmittable(clarification)) return false
    const selected = clarification.choices.find(choice => choice.reply === reply)
    const displayContent = reply === '都对比'
      ? '选择：都对比'
      : (selected ? `选择：${selected.label}` : reply)

    // Reuse the exact same request/conversation/SSE path as typed questions.
    void sendMessage(reply, {
      clarificationSource: target,
      allowFreeText: false,
      displayContent,
      // A picker is a continuation of its originating evidence scope, rather
      // than a new query under a possibly changed global KB filter.  Backend
      // authorization remains authoritative and rechecks this snapshot.
      knowledgeBaseIds: clarification.selected_kb_ids_snapshot,
      requestId: clarification.retryable
        ? clarification.last_submission_request_id
        : null,
    })
    return true
  }

  function stopStreaming() {
    if (!isStreaming.value) return
    const turnContext = activeTurnContext
    aborted = true
    streamRunId += 1
    abortFn?.()
    abortFn = null
    isStreaming.value = false
    activeRequestId.value = ''
    if (!turnContext?.hasConfirmedPresentation?.()) turnContext?.restoreUnconfirmedAssistant?.()
    const lastAssistantMessage = [...messages.value].reverse().find(m => m.role === 'assistant')
    const searchStore = useSearchStore()
    if (lastAssistantMessage) {
      lastAssistantMessage.stopped = true
      invalidateClarification(lastAssistantMessage, searchStore, 'stream_aborted')
      applyTurnLifecycleState(lastAssistantMessage, {
        type: 'error',
        turn_status: 'failed',
        persistence_status: 'unknown',
      })
      lastAssistantMessage.delivery_status = 'stopped'
    }
    if (turnContext?.clarificationSource) {
      void recoverClarificationFromHistory(turnContext, searchStore, 'stream_aborted')
    }
    activeTurnContext = null
    searchStore.finishSteps()
    // 后端在流式开始前已保存会话；停止后刷新侧栏，保留这次未完成的记录入口。
    loadHistory().catch(() => {})
  }

  async function loadHistory() {
    const requestId = ++historyRequestId
    const rows = await fetchAllConversationPages(getChatHistory, {
      pageSize: 100,
      isCurrent: () => requestId === historyRequestId,
    })
    if (rows !== null && requestId === historyRequestId) conversations.value = rows
  }

  function restoreMessageSearch(message) {
    const snapshot = searchSnapshotFromHistoryMessage(message)
    const searchStore = useSearchStore()
    if (!snapshot) {
      searchStore.resetSteps()
      return false
    }
    return searchStore.restoreSnapshot(snapshot)
  }

  function restoreLatestMessageSearch() {
    const latest = [...messages.value].reverse().find(message => (
      message?.role === 'assistant' && searchSnapshotFromHistoryMessage(message)
    ))
    return latest ? restoreMessageSearch(latest) : (useSearchStore().resetSteps(), false)
  }

  async function loadConversation(convId) {
    if (isStreaming.value) return
    const conversationId = String(convId)
    const requestId = ++conversationRequestId
    currentConvId.value = conversationId
    messages.value = []
    isConversationLoading.value = true
    conversationLoadError.value = null
    // 切换时先清空，随后只从服务端持久化的最后一轮 snapshot 恢复；
    // 没有 snapshot 时保持空面板，不能沿用上一个会话的命中片段。
    useSearchStore().resetSteps()
    try {
      const loadedMessages = await getMessages(conversationId)
      if (requestId !== conversationRequestId || currentConvId.value !== conversationId) return
      messages.value = restoreConversationMessages(loadedMessages)
      restoreLatestMessageSearch()
      conversationLoadError.value = null
    } catch (error) {
      // 仅处理当前仍被选中的请求；旧请求失败不应打断后来已切换的会话。
      if (requestId !== conversationRequestId || currentConvId.value !== conversationId) return
      messages.value = []
      conversationLoadError.value = {
        status: Number.isFinite(Number(error?.response?.status)) ? Number(error.response.status) : null,
        detail: typeof error?.response?.data?.detail === 'string' ? error.response.data.detail : '',
        code: typeof error?.code === 'string' ? error.code : '',
      }
      throw error
    } finally {
      if (requestId === conversationRequestId) isConversationLoading.value = false
    }
  }

  function newConversation() {
    if (isStreaming.value) return
    conversationRequestId += 1
    currentConvId.value = null
    messages.value = []
    isConversationLoading.value = false
    activeRequestId.value = ''
    conversationLoadError.value = null
    const searchStore = useSearchStore()
    searchStore.resetSteps()
  }

  async function renameConversation(convId, title) {
    if (isStreaming.value) return
    const updatedConversation = await renameConversationRequest(convId, title)
    const conversation = conversations.value.find(item => item.id === convId)
    if (conversation) Object.assign(conversation, updatedConversation)
    return updatedConversation
  }

  async function removeConversation(convId) {
    if (isStreaming.value) return
    await deleteConversation(convId)
    historyRequestId += 1
    conversations.value = conversations.value.filter(c => c.id !== convId)
    if (currentConvId.value === convId) newConversation()
  }

  async function removeConversations(convIds) {
    if (isStreaming.value) return
    const ids = [...new Set(
      (Array.isArray(convIds) ? convIds : [])
        .map(value => String(value || '').trim())
        .filter(Boolean),
    )]
    if (!ids.length) return { deleted_count: 0, deleted_ids: [] }

    const result = await deleteConversationsRequest(ids)
    historyRequestId += 1
    const deletedIds = new Set(ids)
    conversations.value = conversations.value.filter(c => !deletedIds.has(String(c.id)))
    if (currentConvId.value && deletedIds.has(String(currentConvId.value))) newConversation()
    return result
  }

  return {
    messages, conversations, currentConvId, isStreaming, activeRequestId, isConversationLoading, conversationLoadError,
    searchConfig, selectedKbIds,
    sendMessage, submitClarification, stopStreaming, loadHistory, loadConversation,
    newConversation, renameConversation, removeConversation, removeConversations,
    restoreMessageSearch, restoreLatestMessageSearch,
  }
})
