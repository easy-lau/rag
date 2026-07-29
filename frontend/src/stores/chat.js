import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { createChatStream, getChatHistory, getMessages, deleteConversation } from '@/api/chat'
import { useSearchStore } from './search'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const conversations = ref([])
  const currentConvId = ref(null)
  const isStreaming = ref(false)
  const searchConfig = ref({ method: 'hybrid', rerank: true, top_k: 5, tags: [] })

  const _savedKbIds = localStorage.getItem('selectedKbIds')
  const selectedKbIds = ref(_savedKbIds ? JSON.parse(_savedKbIds) : [])

  watch(selectedKbIds, val => localStorage.setItem('selectedKbIds', JSON.stringify(val)), { deep: true })

  let abortFn = null
  let aborted = false

  async function sendMessage(question) {
    if (isStreaming.value || !question.trim()) return

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

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const lines = decoder.decode(value).split('\n')
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            handleEvent(data, aiMsg, searchStore)
          } catch {}
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') aiMsg.content += `\n\n[${e.message || '请求出错，请重试'}]`
    } finally {
      isStreaming.value = false
      abortFn = null
      searchStore.finishSteps()
      if (aborted) {
        // 用户主动停止：标记为已停止，模板据此显示"已停止生成"，避免一直卡在"思考中"
        aiMsg.stopped = true
        aborted = false
      } else {
        await loadHistory()
      }
    }
  }

  function handleEvent(data, aiMsg, searchStore) {
    if (data.type === 'intent') {
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
    aborted = true
    abortFn?.()
    isStreaming.value = false
  }

  async function loadHistory() {
    conversations.value = await getChatHistory()
    if (conversations.value.length && !currentConvId.value) {
      await loadConversation(conversations.value[0].id)
    }
  }

  async function loadConversation(convId) {
    currentConvId.value = convId
    messages.value = await getMessages(convId)
  }

  function newConversation() {
    currentConvId.value = null
    messages.value = []
    const searchStore = useSearchStore()
    searchStore.resetSteps()
  }

  async function removeConversation(convId) {
    await deleteConversation(convId)
    conversations.value = conversations.value.filter(c => c.id !== convId)
    if (currentConvId.value === convId) newConversation()
  }

  return {
    messages, conversations, currentConvId, isStreaming,
    searchConfig, selectedKbIds,
    sendMessage, stopStreaming, loadHistory, loadConversation,
    newConversation, removeConversation,
  }
})
