import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { createChatStream, getChatHistory, getMessages, renameConversation as renameConversationRequest, deleteConversation } from '@/api/chat'
import { useSearchStore } from './search'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const conversations = ref([])
  const currentConvId = ref(null)
  const isStreaming = ref(false)
  const isConversationLoading = ref(false)
  const searchConfig = ref({ method: 'hybrid', rerank: true, top_k: 5, tags: [] })

  const _savedKbIds = localStorage.getItem('selectedKbIds')
  const selectedKbIds = ref(_savedKbIds ? JSON.parse(_savedKbIds) : [])

  watch(selectedKbIds, val => localStorage.setItem('selectedKbIds', JSON.stringify(val)), { deep: true })

  let abortFn = null
  let aborted = false
  // 用户快速切换历史对话时，只接收最后一次请求的响应，避免旧响应覆盖当前会话。
  let conversationRequestId = 0
  // 停止生成后使旧 SSE 事件失效，避免其 done 事件把已新建的空会话切回旧会话。
  let streamRunId = 0

  async function sendMessage(question) {
    if (isStreaming.value || !question.trim()) return
    const runId = ++streamRunId

    const searchStore = useSearchStore()
    searchStore.resetSteps()
    isStreaming.value = true
    aborted = false

    messages.value.push({ id: Date.now(), role: 'user', content: question, created_at: new Date() })
    messages.value.push({ id: Date.now() + 1, role: 'assistant', content: '', sources: [], stopped: false, tokens: null, created_at: new Date() })
    const aiMsg = messages.value[messages.value.length - 1]

    const { promise, abort } = createChatStream({
      question,
      conversation_id: currentConvId.value,
      knowledge_base_ids: selectedKbIds.value,
      search_config: searchConfig.value,
    })
    abortFn = abort

    try {
      const res = await promise
      if (!res.ok) {
        let detail = '请求失败，请稍后重试'
        try {
          const body = await res.json()
          detail = body?.detail || detail
        } catch {}
        throw new Error(detail)
      }

      // 新会话在后端已提交；先从响应头绑定 ID，避免用户刚开始生成就停止时丢失会话上下文。
      const startedConversationId = res.headers.get('X-Conversation-ID')
      if (startedConversationId && runId === streamRunId) currentConvId.value = startedConversationId

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let sseBuffer = ''

      // fetch 的每个 read() 不保证对应一条完整 SSE 事件；保留残片，避免 JSON 被拆包后静默丢失。
      const processSseEvent = (rawEvent) => {
        const rawData = rawEvent
          .split(/\r?\n/)
          .filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trimStart())
          .join('\n')
        if (!rawData) return
        try {
          handleEvent(JSON.parse(rawData), aiMsg, searchStore, runId)
        } catch {}
      }
      const flushCompleteSseEvents = () => {
        const events = sseBuffer.split(/\r?\n\r?\n/)
        sseBuffer = events.pop() || ''
        events.forEach(processSseEvent)
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
      if (e.name !== 'AbortError') aiMsg.content += `\n\n[${e.message || '请求出错，请重试'}]`
    } finally {
      if (runId !== streamRunId) return
      isStreaming.value = false
      abortFn = null
      searchStore.finishSteps()
      if (aborted) {
        // 用户主动停止：标记为已停止，模板据此显示"已停止生成"，避免一直卡在"思考中"
        aiMsg.stopped = true
        aborted = false
      } else {
        await loadHistory().catch(() => {})
      }
    }
  }

  function handleEvent(data, aiMsg, searchStore, runId) {
    if (runId !== streamRunId) return
    if (data.type === 'conversation_started') {
      if (data.conversation_id) currentConvId.value = data.conversation_id
    } else if (data.type === 'intent') {
      searchStore.setIntentDecision(data.decision || data)
    } else if (data.type === 'search_step') {
      searchStore.updateStep(data.step, data.status)
    } else if (data.type === 'search_results') {
      searchStore.setResults(data)
      aiMsg.sources = data.results?.slice(0, 5) || []
    } else if (data.type === 'text_delta') {
      aiMsg.content += data.content
    } else if (data.type === 'usage') {
      aiMsg.tokens = data.total_tokens
    } else if (data.type === 'done') {
      if (data.conversation_id) currentConvId.value = data.conversation_id
      searchStore.finishSteps()
    } else if (data.type === 'error') {
      aiMsg.content += `\n\n[错误：${data.message}]`
    }
  }

  function stopStreaming() {
    if (!isStreaming.value) return
    aborted = true
    streamRunId += 1
    abortFn?.()
    abortFn = null
    isStreaming.value = false
    const lastAssistantMessage = [...messages.value].reverse().find(m => m.role === 'assistant')
    if (lastAssistantMessage) lastAssistantMessage.stopped = true
    useSearchStore().finishSteps()
    // 后端在流式开始前已保存会话；停止后刷新侧栏，保留这次未完成的记录入口。
    loadHistory().catch(() => {})
  }

  async function loadHistory() {
    conversations.value = await getChatHistory()
  }

  async function loadConversation(convId) {
    if (isStreaming.value) return
    const requestId = ++conversationRequestId
    currentConvId.value = convId
    messages.value = []
    isConversationLoading.value = true
    try {
      const loadedMessages = await getMessages(convId)
      if (requestId !== conversationRequestId || currentConvId.value !== convId) return
      messages.value = loadedMessages
    } catch (error) {
      // 仅处理当前仍被选中的请求；旧请求失败不应打断后来已切换的会话。
      if (requestId !== conversationRequestId || currentConvId.value !== convId) return
      currentConvId.value = null
      messages.value = []
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
    conversations.value = conversations.value.filter(c => c.id !== convId)
    if (currentConvId.value === convId) newConversation()
  }

  return {
    messages, conversations, currentConvId, isStreaming, isConversationLoading,
    searchConfig, selectedKbIds,
    sendMessage, stopStreaming, loadHistory, loadConversation,
    newConversation, renameConversation, removeConversation,
  }
})
