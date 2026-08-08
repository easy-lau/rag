<template>
  <article class="chat-message" :class="isUser ? 'chat-message--user' : 'chat-message--assistant'">
    <!-- AI avatar -->
    <div v-if="!isUser" class="chat-message__avatar chat-message__avatar--assistant">
      <n-icon><HardwareChipOutline /></n-icon>
    </div>

    <div class="chat-message__content">
      <!-- User bubble -->
      <div
        v-if="isUser"
        class="chat-message__user-bubble"
        @copy="handleUserCopy"
      >
        {{ message.content }}
      </div>

      <!-- AI card -->
      <div v-else class="chat-message__assistant-card">
        <div
          v-if="isGeneralModelAnswer"
          class="message-general-fallback"
          role="status"
          aria-label="通用大模型回答，未经知识库验证"
        >
          <n-icon :size="16" aria-hidden="true"><WarningOutline /></n-icon>
          <div>
            <strong>通用大模型回答</strong>
            <span>未经知识库验证，请勿视为企业制度或内部事实依据。</span>
          </div>
        </div>
        <div
          v-if="message.content"
          class="markdown-body text-sm text-gray-800 dark:text-gray-200"
          v-html="rendered"
          @click="handleMarkdownClick"
        />
        <div v-else-if="emptyPresentation.kind === 'stopped'" class="flex items-center gap-2 text-gray-400 text-sm">
          <n-icon><StopCircleOutline /></n-icon><span>已停止生成</span>
        </div>
        <div v-else-if="emptyPresentation.kind === 'thinking'" class="flex items-center gap-2 text-gray-400 text-sm" role="status" aria-live="polite">
          <n-spin size="small" /><span>{{ emptyPresentation.text }}</span>
        </div>
        <div
          v-else-if="emptyPresentation.kind !== 'hidden'"
          class="flex items-center gap-2 text-gray-400 text-sm"
          role="status"
          aria-live="polite"
        >
          <span>{{ emptyPresentation.text }}</span>
        </div>

        <section
          v-if="clarification"
          class="message-clarification"
          aria-label="补充问题条件"
        >
          <div class="message-clarification__heading">
            <span class="message-clarification__eyebrow">{{ clarificationEyebrow }}</span>
          </div>

          <p
            v-if="clarification.status === 'active' && !clarification.invalidated && !clarification.submitted"
            class="message-clarification__hint"
          >
            {{ clarificationHint }}
          </p>
          <p
            v-if="clarification.submitted"
            class="message-clarification__submitted"
            aria-live="polite"
          >
            {{ clarificationSubmittedText }}
          </p>
          <p
            v-else-if="clarificationStatusText"
            :id="clarificationStatusId"
            class="message-clarification__status"
            :class="{ 'message-clarification__status--invalid': clarification.invalidated }"
            aria-live="polite"
          >
            {{ clarificationStatusText }}
          </p>

          <template v-if="clarification.choices.length">
            <div
              v-if="!clarification.submitted"
              class="message-clarification__choices"
              role="group"
              aria-label="可选适用范围"
              :aria-describedby="clarificationStatusText ? clarificationStatusId : undefined"
            >
              <button
                v-for="choice in clarification.choices"
                :key="choice.id"
                type="button"
                class="message-clarification__choice"
                :disabled="!clarificationCanSubmit"
                :aria-label="`选择第 ${choice.index} 项：${choice.label}`"
                :title="clarificationDisabledReason"
                @click="selectClarification(choice.reply)"
              >
                <span class="message-clarification__index" aria-hidden="true">{{ choice.index }}</span>
                <span>{{ choice.label }}</span>
              </button>
              <button
                v-if="clarification.allowed_actions.includes('select_all') && clarification.choices.length > 1"
                type="button"
                class="message-clarification__choice message-clarification__choice--compare"
                :disabled="!clarificationCanSubmit"
                aria-label="对比全部适用范围"
                :title="clarificationDisabledReason"
                @click="selectClarification('都对比')"
              >
                都对比
              </button>
            </div>
          </template>

          <p
            v-else-if="clarification.status === 'active' && !clarification.invalidated && !clarification.submitted"
            class="message-clarification__refinement"
            aria-live="polite"
          >
            可直接在输入框补充条件。
          </p>
        </section>

        <!-- 历史消息直接使用自身持久化的 sources，不依赖右侧“本次检索”状态。 -->
        <div v-if="hasPersistedEvidence" class="message-evidence">
          <button
            type="button"
            class="message-evidence__toggle"
            :class="`message-evidence__toggle--${evidenceSummary.tone}`"
            :aria-expanded="evidenceExpanded"
            :aria-controls="evidencePanelId"
            :aria-label="`${evidenceSummary.ariaLabel}，${evidenceExpanded ? '收起检索资料' : evidenceSummary.actionLabel}`"
            @click="evidenceExpanded = !evidenceExpanded"
          >
            <n-icon :size="15" aria-hidden="true"><DocumentTextOutline /></n-icon>
            <span class="message-evidence__summary">
              <span>{{ evidenceSummary.label }}</span>
              <span v-if="evidenceSummary.level" aria-hidden="true">·</span>
              <strong v-if="evidenceSummary.level">{{ evidenceSummary.level }}</strong>
              <span v-if="evidenceSummary.percent !== null" class="message-evidence__score">
                {{ evidenceSummary.percent }}%
              </span>
            </span>
            <span class="message-evidence__action">{{ evidenceExpanded ? '收起' : evidenceSummary.actionLabel }}</span>
            <n-icon
              :size="14"
              aria-hidden="true"
              class="message-evidence__chevron"
              :class="{ 'message-evidence__chevron--open': evidenceExpanded }"
            >
              <ChevronDownOutline />
            </n-icon>
          </button>

          <section
            v-show="evidenceExpanded"
            :id="evidencePanelId"
            class="message-evidence__panel"
            role="region"
            :aria-label="`本条回答实际使用了 ${sourceDocumentCount} 篇知识库文章`"
          >
            <div class="message-evidence__panel-head">
              <div>
                <p class="message-evidence__panel-title">本条回答使用的知识库文章</p>
                <p class="message-evidence__panel-hint">这里只展示实际进入回答上下文的文章，不展开命中片段。</p>
              </div>
              <span class="message-evidence__count">{{ sourceDocumentCount }} 篇文章</span>
            </div>

            <div class="message-evidence__results">
              <DocumentEvidenceGroup
                v-for="(group, index) in sourceGroups"
                :key="group.key"
                :group="group"
                :rank="index + 1"
                :preview-enabled="canPreviewSources"
                document-only
                :id-prefix="`message-${message.id || 'streaming'}`"
                @preview="openSource"
              />
            </div>

            <!-- 外部链接只表达资料原始地址，不等同于回答依据。 -->
            <div v-if="urlSources.length" class="message-evidence__links">
              <span>原始链接</span>
              <a
                v-for="source in urlSources"
                :key="`${source._evidenceKey}-url`"
                :href="source.source_url"
                target="_blank"
                rel="noopener noreferrer"
                :title="source.source_url"
              >
                <n-icon :size="13" aria-hidden="true"><LinkOutline /></n-icon>
                <span>{{ source.filename || '未命名文档' }}</span>
              </a>
            </div>
          </section>
        </div>

        <section
          v-if="relatedCandidateGroups.length"
          class="message-related-discovery"
          role="status"
          aria-label="本轮找到的相近文章，未作为回答依据"
        >
          <n-icon :size="16" aria-hidden="true"><DocumentTextOutline /></n-icon>
          <div class="message-related-discovery__content">
            <strong>找到 {{ relatedCandidateGroups.length }} 篇相近文章</strong>
            <span>这些资料未作为本条回答的已验证依据。</span>
            <ul class="message-related-discovery__titles">
              <li v-for="group in visibleRelatedCandidateGroups" :key="group.key">
                {{ group.filename }}
              </li>
            </ul>
            <span v-if="hiddenRelatedCandidateCount" class="message-related-discovery__more">
              还有 {{ hiddenRelatedCandidateCount }} 篇
            </span>
          </div>
          <n-button
            text
            size="tiny"
            class="message-related-discovery__action"
            aria-label="查看本条回答找到的相近文章"
            @click="$emit('inspect', message)"
          >
            查看文章
          </n-button>
        </section>

        <!-- Actions -->
        <div v-if="message.content" class="chat-message__actions">
          <n-button class="chat-message__action" quaternary size="tiny" @click="copy">
            <template #icon><n-icon><CopyOutline /></n-icon></template>
            复制
          </n-button>
          <n-button class="chat-message__action" quaternary size="tiny" :disabled="chatStore.isStreaming" @click="$emit('retry', message)">
            <template #icon><n-icon><RefreshOutline /></n-icon></template>
            {{ retryActionLabel }}
          </n-button>
          <n-button
            v-if="canInspectSearch"
            quaternary
            size="tiny"
            class="chat-message__action"
            aria-label="查看本条回答的检索摘要"
            @click="$emit('inspect', message)"
          >
            <template #icon><n-icon><DocumentTextOutline /></n-icon></template>
            检索摘要
          </n-button>
        </div>

        <div
          v-if="showTraceDiagnostic"
          class="message-trace"
          role="status"
          aria-live="polite"
        >
          <span class="message-trace__label">错误追踪 ID</span>
          <code :title="traceId">{{ traceId }}</code>
          <n-button
            text
            size="tiny"
            class="message-trace__copy"
            aria-label="复制错误追踪 ID"
            @click="copyTraceId"
          >
            <template #icon><n-icon><CopyOutline /></n-icon></template>
            复制
          </n-button>
        </div>
      </div>

      <div class="chat-message__meta">
        {{ formatTime(message.created_at) }}
        <span v-if="!isUser && responseDurationText"> · 思考回答耗时 {{ responseDurationText }}</span>
        <span v-if="message.tokens"> · {{ message.tokens }} tokens</span>
      </div>
    </div>

    <!-- User avatar -->
    <div v-if="isUser" class="chat-message__avatar chat-message__avatar--user">
      <n-icon><PersonOutline /></n-icon>
    </div>
  </article>
</template>

<script setup>
import { computed, ref } from 'vue'
import { NButton, NIcon, NSpin, useMessage } from 'naive-ui'
import { ChevronDownOutline, CopyOutline, RefreshOutline, HardwareChipOutline, PersonOutline, DocumentTextOutline, StopCircleOutline, LinkOutline, WarningOutline } from '@vicons/ionicons5'
import { useClipboard } from '@vueuse/core'
import { renderMarkdown } from '@/utils/markdown'
import { useSettingsStore } from '@/stores/settings'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import DocumentEvidenceGroup from '@/components/search/DocumentEvidenceGroup.vue'
import { persistedAnswerSources } from '@/utils/chatEvidence'
import { groupEvidenceByDocument, safeExternalSourceUrl } from '@/utils/evidenceDocuments'
import { isClarificationSubmittable, normalizeClarification } from '@/utils/chatClarification'
import { hasSearchSnapshot } from '@/utils/chatHistory'
import { normalizeTraceId } from '@/utils/chatRequest'
import { isNonAnswerEvidenceStatus, normalizeEvidenceStatus } from '@/utils/evidenceStatus'
import { emptyAssistantPresentation } from '@/utils/chatTurnState'

const props = defineProps({ message: Object })
const emit = defineEmits(['retry', 'preview', 'clarify', 'inspect'])

const msg = useMessage()
const { copy: copyText } = useClipboard()
const settingsStore = useSettingsStore()
const chatStore = useChatStore()
const authStore = useAuthStore()
const evidenceExpanded = ref(false)

const isUser = computed(() => props.message.role === 'user')
const isGeneralModelAnswer = computed(() => (
  !isUser.value
  && (
    props.message.answer_provenance
    || props.message.search_meta?.answer_provenance
    || props.message.search_snapshot?.answer_provenance
  ) === 'general_model'
))
const emptyPresentation = computed(() => emptyAssistantPresentation(props.message, {
  isStreaming: chatStore.isStreaming,
  activeRequestId: chatStore.activeRequestId,
}))
const clarification = computed(() => (
  isUser.value ? null : normalizeClarification(props.message.clarification)
))
const clarificationCanSubmit = computed(() => (
  !chatStore.isStreaming && isClarificationSubmittable(clarification.value)
))
const clarificationStatusId = computed(() => {
  const messageId = String(props.message.id || 'streaming').replace(/[^a-zA-Z0-9_-]/g, '-')
  return `message-clarification-status-${messageId}`
})
const clarificationEyebrow = computed(() => {
  if (clarification.value?.submitted) return '已确认'
  if (clarification.value?.retryable) return '可以重试'
  if (clarification.value?.invalidated) return '选择已失效'
  if (clarification.value?.status !== 'active') return '正在保存'
  return '需要你确认'
})
const clarificationHint = computed(() => (
  clarification.value?.adapter === 'evidence'
    ? '选择后只会在当前授权范围内使用对应资料继续回答。'
    : '选择或补充后，系统会重新理解原问题并按当前授权范围继续处理。'
))
const clarificationStatusText = computed(() => {
  const value = clarification.value
  if (!value || value.submitted) return ''
  if (value.retryable) {
    return '上次选择未完成，系统已确认原澄清状态仍有效。你可以再次选择，重复点击不会创建新的逻辑请求。'
  }
  if (value.invalidated) {
    if (value.invalid_reason === 'stream_aborted') {
      return '生成已停止，本次范围选择没有生效。请重新提问。'
    }
    return '本次澄清未能完成保存，以下选项已失效。请重新提问。'
  }
  if (value.status !== 'active') return '正在确认本次澄清是否已保存，确认前不能选择。'
  if (chatStore.isStreaming) return '范围选择已安全保存，当前处理结束后即可选择。'
  return ''
})
const clarificationDisabledReason = computed(() => (
  clarificationCanSubmit.value ? undefined : (clarificationStatusText.value || '该范围选择当前不可用')
))
const clarificationSubmittedText = computed(() => {
  if (!clarification.value?.submitted) return ''
  if (clarification.value.submitted_reply === '都对比') return '已提交“都对比”，查询结果会显示在下方。'
  const selected = clarification.value.choices.find(choice => (
    choice.reply === clarification.value.submitted_reply
  ))
  return selected
    ? `已提交第 ${selected.index} 项：${selected.label}`
    : '已提交补充范围，查询结果会显示在下方。'
})
const canInspectSearch = computed(() => (
  !isUser.value && hasSearchSnapshot(props.message)
))
const traceId = computed(() => normalizeTraceId(
  props.message.trace_id
    || props.message.traceId
    || props.message.search_meta?.trace_id
    || props.message.search_snapshot?.trace_id,
))
const showTraceDiagnostic = computed(() => Boolean(
  traceId.value
  && (
    (Array.isArray(props.message.stream_errors) && props.message.stream_errors.length)
    || ['failed', 'incomplete', 'stopped'].includes(props.message.delivery_status)
    || props.message.persistence_status === 'failed'
  ),
))
const retryActionLabel = computed(() => {
  if (props.message.failure_reason === 'turn_in_progress') return '获取结果'
  if (
    props.message.retry_with_new_request_id === true
    || props.message.failure_reason === 'persistence_unrecoverable'
    || (
      ['failed', 'cancelled'].includes(props.message.turn_status)
      && props.message.error_code
      && props.message.same_request_recoverable !== true
    )
  ) return '重新发送'
  if (
    props.message.failure_reason === 'persistence_failed'
    || (
      props.message.persistence_status === 'failed'
      && props.message.turn_status === 'persist_failed'
    )
  ) return '恢复回答'
  return '重新生成'
})
// 系统设置「显示参考来源」总开关：关闭则隐藏所有来源行（默认显示）
const showSources = computed(() => settingsStore.data.show_sources !== false)
// 普通问答角色可查看本轮命中片段，但不能因此获得整篇文档读取能力。
// 只有后端同样允许的 doc:read 角色才显示可点击的全文预览行为。
const canPreviewSources = computed(() => authStore.hasPerm('doc:read'))
const responseDurationText = computed(() => {
  const milliseconds = Number(props.message.duration_ms)
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return ''
  if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`
  return `${(milliseconds / 1000).toFixed(milliseconds < 10_000 ? 1 : 0)} 秒`
})

// 去掉正文里的内联引用标记（如 [1]、[2,3]），来源改到底部统一展示。
// 单次扫描：命中代码块/行内代码则原样保留（避免误删代码里的 [0]），命中引用标记则删除。
function stripCitations(text) {
  if (!text) return ''
  return text.replace(
    /(```[\s\S]*?```|`[^`\n]+`)|\[\s*\d+(?:\s*[,，]\s*\d+)*\s*\]/g,
    (match, code) => (code ? code : '')
  )
}

const rendered = computed(() => renderMarkdown(stripCitations(props.message.content)))

function finiteScore(value) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  return Math.min(1, Math.max(0, parsed))
}

function normalizedEvidenceRole(source) {
  const role = typeof source?.evidence_role === 'string'
    ? source.evidence_role.trim().toLowerCase()
    : ''
  if (['direct', 'related', 'unverified', 'irrelevant'].includes(role)) return role

  // 旧消息可能没有 evidence_role；只能把确定性约束冲突降级为相近资料，
  // 不能仅凭一个高相似度分数推断它是回答依据。
  const constraint = typeof source?.constraint_status === 'string'
    ? source.constraint_status.trim().toLowerCase()
    : ''
  if (constraint.includes('mismatch') || constraint.includes('conflict')) return 'related'
  return ''
}

function normalizedSource(source, index) {
  const effectiveScore = finiteScore(source?.effective_score) ?? finiteScore(source?.score)
  const answerSupport = finiteScore(source?.answer_support)
  const chunkKey = source?.id || source?.chunk_id
  return {
    ...source,
    id: chunkKey || source?.id,
    score: effectiveScore,
    effective_score: effectiveScore,
    answer_support: answerSupport,
    evidence_role: normalizedEvidenceRole(source),
    _evidenceKey: chunkKey
      || `${source?.doc_id || source?.filename || 'source'}-${source?.chunk_index ?? index}`,
  }
}

// 历史接口直接返回消息持久化的 sources。先过滤旧版本误存的 no_hit 候选，
// 再按具体片段去重；同一文档的不同回答依据仍需保留。
const sources = computed(() => {
  const list = persistedAnswerSources(props.message)
  const seen = new Set()
  const out = []
  for (const [index, rawSource] of list.entries()) {
    if (!rawSource || typeof rawSource !== 'object') continue
    const source = normalizedSource(rawSource, index)
    const key = source._evidenceKey
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(source)
  }
  return out
})
const sourceGroups = computed(() => groupEvidenceByDocument(sources.value))
const sourceDocumentCount = computed(() => sourceGroups.value.length)
const relatedCandidateGroups = computed(() => {
  if (isUser.value || !props.message.content || !showSources.value) return []
  const snapshot = props.message.search_snapshot
  const rawCandidates = Array.isArray(snapshot?.results)
    ? snapshot.results
    : (Array.isArray(snapshot?.candidates) ? snapshot.candidates : [])
  if (!rawCandidates.length) return []

  const status = normalizeEvidenceStatus(
    props.message.evidence_status
      || props.message.search_meta?.evidence_status
      || snapshot?.evidence_status,
  )
  if (['scope_mismatch', 'error', 'skipped'].includes(status)) return []
  const answerDocumentKeys = new Set(sourceGroups.value.map(group => group.key))
  return groupEvidenceByDocument(rawCandidates.filter(candidate => {
    const role = normalizedEvidenceRole(candidate)
    if (role === 'irrelevant' || role === 'unverified') return false
    return role === 'related' || isNonAnswerEvidenceStatus(status)
  })).filter(group => !answerDocumentKeys.has(group.key))
})
const visibleRelatedCandidateGroups = computed(() => (
  relatedCandidateGroups.value.slice(0, 3)
))
const hiddenRelatedCandidateCount = computed(() => Math.max(
  0,
  relatedCandidateGroups.value.length - visibleRelatedCandidateGroups.value.length,
))

const hasPersistedEvidence = computed(() => (
  !isUser.value
  && Boolean(props.message.content)
  && showSources.value
  && sources.value.length > 0
  && sources.value.some(source => source.retrieval_executed !== false)
  && props.message.retrieval_executed !== false
  && props.message.search_meta?.retrieval_executed !== false
))

function documentCountForSources(items) {
  return groupEvidenceByDocument(items).length
}

const evidenceSummary = computed(() => {
  const unverifiedSources = sources.value.filter(source => (
    source.source_verification === 'unverified'
    || source.rerank_status === 'unverified'
  ))
  const directSources = sources.value.filter(source => source.evidence_role === 'direct')
  const partialSources = sources.value.filter(source => (
    source.jointly_selected && source.coverage_status === 'partial'
  ))
  const relatedSources = sources.value.filter(source => source.evidence_role === 'related')
  if (unverifiedSources.length) {
    const documentCount = documentCountForSources(unverifiedSources)
    return {
      label: '待验证参考资料',
      level: `${documentCount} 篇文章`,
      percent: null,
      tone: 'warning',
      actionLabel: '查看参考资料',
      ariaLabel: `重排模型发生技术故障；本条回答使用了 ${documentCount} 篇经过权限和适用范围过滤、但尚未完成语义验证的知识库文章`,
    }
  }
  if (directSources.length) {
    const documentCount = documentCountForSources(directSources)
    return {
      label: '知识库依据',
      level: `${documentCount} 篇文章`,
      percent: null,
      tone: 'success',
      actionLabel: '查看依据',
      ariaLabel: `本条回答使用了 ${documentCount} 篇知识库文章`,
    }
  }
  if (partialSources.length) {
    const documentCount = documentCountForSources(partialSources)
    return {
      label: '部分知识库依据',
      level: `${documentCount} 篇文章`,
      percent: null,
      tone: 'warning',
      actionLabel: '查看依据',
      ariaLabel: `本条回答使用了 ${documentCount} 篇文章支撑部分内容，但知识库证据尚未完整覆盖问题`,
    }
  }
  if (relatedSources.length) {
    const documentCount = documentCountForSources(relatedSources)
    return {
      label: '相近资料',
      level: `${documentCount} 篇文章`,
      percent: null,
      tone: 'warning',
      actionLabel: '查看资料',
      ariaLabel: `本条回答仅使用了 ${documentCount} 篇相近文章，不能作为直接回答依据`,
    }
  }
  return {
    label: '检索候选',
    level: `${sourceDocumentCount.value} 篇文章`,
    percent: null,
    tone: 'neutral',
    actionLabel: '查看结果',
    ariaLabel: `本条回答保存了 ${sourceDocumentCount.value} 篇尚待验证的检索文章`,
  }
})

const evidencePanelId = computed(() => {
  const messageId = String(props.message.id || 'streaming').replace(/[^a-zA-Z0-9_-]/g, '-')
  return `message-evidence-${messageId}`
})

// 参考来源：只取带数据来源链接的来源，点击跳转外部 URL
const urlSources = computed(() => sourceGroups.value
  .map(group => {
    const source = group.items.find(item => safeExternalSourceUrl(item.source_url))
    if (!source) return null
    return { ...source, source_url: safeExternalSourceUrl(source.source_url) }
  })
  .filter(Boolean))

function openSource(src) {
  // 不跳转，交给父组件在当前页弹出只读预览
  emit('preview', src)
}

function selectClarification(reply) {
  if (!clarificationCanSubmit.value) return
  emit('clarify', { message: props.message, reply })
}

function copy() {
  copyText(props.message.content)
  msg.success('已复制')
}

function copyTraceId() {
  if (!traceId.value) return
  Promise.resolve(copyText(traceId.value))
    .then(() => msg.success('Trace ID 已复制'))
    .catch(() => msg.error('复制失败，请手动选择 Trace ID'))
}

// 浏览器从块级气泡复制文本时可能把元素边界转换成首尾换行。
// 只接管用户气泡内的原生复制，保留正文内部换行，同时清理并非消息内容的边界换行。
function handleUserCopy(event) {
  if (!event.clipboardData) return
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || !selection.rangeCount) return

  const range = selection.getRangeAt(0)
  if (!event.currentTarget.contains(range.commonAncestorContainer)) return

  const selectedText = selection.toString()
  if (!selectedText) return

  event.preventDefault()
  event.clipboardData.setData(
    'text/plain',
    selectedText.replace(/^[\r\n]+|[\r\n]+$/g, '')
  )
}

function handleMarkdownClick(e) {
  if (!(e.target instanceof Element)) return
  const btn = e.target.closest('.copy-btn')
  if (!btn) return
  const code = btn.closest('.code-block-wrapper')?.querySelector('code')
  if (!code) return
  navigator.clipboard.writeText(code.innerText).then(() => {
    btn.textContent = '已复制'
    setTimeout(() => { btn.textContent = '复制' }, 1500)
  })
}

function formatTime(t) {
  if (!t) return ''
  return new Date(t).toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  })
}
</script>

<style scoped>
.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 22px;
}

.chat-message--user { justify-content: flex-end; }
.chat-message--assistant { justify-content: flex-start; }

.chat-message__avatar {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  place-items: center;
  margin-top: 2px;
  border-radius: 11px;
  font-size: 15px;
}

.chat-message__avatar--assistant {
  background: linear-gradient(180deg, var(--ui-primary) 0%, var(--ui-primary-hover) 100%);
  box-shadow: 0 7px 18px color-mix(in srgb, var(--ui-primary) 18%, transparent);
  color: var(--ui-text-on-primary);
}

.chat-message__avatar--user {
  border: 1px solid var(--ui-border);
  background: var(--ui-surface);
  color: var(--ui-text-secondary);
}

.chat-message__content {
  min-width: 0;
  max-width: min(82%, 760px);
}

.chat-message--user .chat-message__content { max-width: min(76%, 680px); }

.chat-message__assistant-card {
  border: 1px solid var(--ui-border);
  border-radius: 4px var(--ui-radius-card) var(--ui-radius-card) var(--ui-radius-card);
  background: var(--ui-surface);
  box-shadow: 0 7px 24px color-mix(in srgb, var(--ui-text) 4%, transparent);
  padding: 15px 16px;
}

.chat-message__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
  margin-top: 13px;
  border-top: 1px solid var(--ui-divider);
  padding-top: 9px;
}

:deep(.chat-message__action.n-button) {
  --n-height: 30px !important;
  --n-border-radius: 9px !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-color-pressed: var(--ui-surface-pressed) !important;
  --n-text-color: var(--ui-text-tertiary) !important;
  --n-text-color-hover: var(--ui-primary) !important;
  padding: 0 8px;
  font-size: 11px;
}

.chat-message__meta {
  margin-top: 6px;
  padding: 0 3px;
  color: var(--ui-text-tertiary);
  font-size: 10px;
  line-height: 1.5;
}

.chat-message--user .chat-message__meta { text-align: right; }

.message-general-fallback {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 12px;
  border: 1px solid color-mix(in srgb, var(--ui-warning) 38%, var(--ui-border));
  border-radius: var(--ui-radius-control);
  background: color-mix(in srgb, var(--ui-warning) 8%, var(--ui-surface));
  padding: 9px 10px;
  color: var(--ui-warning);
  font-size: 12px;
  line-height: 1.5;
}

.message-general-fallback > div {
  display: grid;
  min-width: 0;
  gap: 1px;
}

.message-general-fallback strong {
  color: var(--ui-text);
  font-weight: 650;
}

.message-general-fallback span {
  color: var(--ui-text-secondary);
  overflow-wrap: anywhere;
}

.message-clarification {
  margin-top: 12px;
  border: 1px solid color-mix(in srgb, var(--ui-warning) 34%, var(--ui-border));
  border-radius: var(--ui-radius-popover);
  background: color-mix(in srgb, var(--ui-warning) 7%, var(--ui-surface));
  padding: 12px;
}

.message-clarification__heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 9px;
  color: var(--ui-text);
  font-size: 13px;
  line-height: 1.5;
}

.message-clarification__heading strong {
  min-width: 0;
  flex: 1 1 240px;
  font-weight: 650;
  overflow-wrap: anywhere;
}

.message-clarification__eyebrow {
  border-radius: var(--ui-radius-pill);
  background: color-mix(in srgb, var(--ui-warning) 15%, var(--ui-surface));
  padding: 2px 7px;
  color: var(--ui-warning);
  font-size: 11px;
  font-weight: 650;
}

.message-clarification__hint,
.message-clarification__refinement,
.message-clarification__submitted,
.message-clarification__status {
  margin-top: 7px;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.6;
}

.message-clarification__refinement {
  color: var(--ui-warning);
}

.message-clarification__submitted {
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface-muted);
  padding: 8px 10px;
  color: var(--ui-text-secondary);
}

.message-clarification__status {
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface-muted);
  padding: 8px 10px;
  color: var(--ui-warning);
}

.message-clarification__status--invalid {
  background: var(--ui-danger-subtle);
  color: var(--ui-danger);
}

.message-clarification__choices {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 8px;
  margin-top: 10px;
}

.message-clarification__choice {
  display: flex;
  min-height: var(--ui-control-height);
  width: 100%;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--ui-border-strong);
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface);
  padding: 8px 10px;
  color: var(--ui-text);
  font-size: 12px;
  line-height: 1.45;
  text-align: left;
  overflow-wrap: anywhere;
  transition: border-color 150ms ease, background-color 150ms ease, color 150ms ease;
}

.message-clarification__choice:hover:not(:disabled) {
  border-color: var(--ui-border-focus);
  background: var(--ui-surface-hover);
}

.message-clarification__choice:focus-visible {
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: 2px;
  box-shadow: var(--ui-focus-ring);
}

.message-clarification__choice:disabled {
  cursor: not-allowed;
  border-color: var(--ui-border);
  background: var(--ui-surface-disabled);
  color: var(--ui-text-disabled);
}

.message-clarification__choice--compare {
  justify-content: center;
  border-color: color-mix(in srgb, var(--ui-primary) 42%, var(--ui-border));
  background: var(--ui-primary-subtle);
  color: var(--ui-primary);
  font-weight: 650;
  text-align: center;
}

.message-clarification__index {
  display: inline-grid;
  flex: 0 0 22px;
  width: 22px;
  height: 22px;
  place-items: center;
  border-radius: var(--ui-radius-pill);
  background: var(--ui-surface-muted);
  color: var(--ui-primary);
  font-size: 11px;
  font-weight: 700;
}

.message-related-discovery {
  display: flex;
  min-width: 0;
  align-items: flex-start;
  gap: 9px;
  margin-top: 12px;
  border: 1px solid color-mix(in srgb, var(--ui-warning) 28%, var(--ui-border));
  border-radius: var(--ui-radius-popover);
  background: color-mix(in srgb, var(--ui-warning) 6%, var(--ui-surface));
  padding: 10px;
  color: var(--ui-warning);
}

.message-related-discovery__content {
  display: grid;
  min-width: 0;
  flex: 1 1 auto;
  gap: 2px;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.message-related-discovery__content strong {
  color: var(--ui-text);
  font-weight: 650;
}

.message-related-discovery__titles {
  display: grid;
  gap: 2px;
  margin: 5px 0 0;
  padding-left: 16px;
  color: var(--ui-text);
}

.message-related-discovery__titles li {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.message-related-discovery__more {
  color: var(--ui-text-tertiary);
}

.message-related-discovery__action {
  flex: 0 0 auto;
  color: var(--ui-warning);
}

.message-related-discovery__action:focus-visible {
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: 2px;
  box-shadow: var(--ui-focus-ring);
}

.message-evidence {
  margin-top: 12px;
}

.message-evidence__toggle {
  display: inline-flex;
  min-height: var(--ui-control-height-compact);
  max-width: 100%;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-pill);
  background: var(--ui-surface-muted);
  padding: 5px 9px;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.25;
  text-align: left;
  transition: border-color 150ms ease, background-color 150ms ease, color 150ms ease;
}

.message-evidence__toggle:hover {
  border-color: var(--ui-border-strong);
  background: var(--ui-surface-hover);
  color: var(--ui-text);
}

.message-evidence__toggle:focus-visible {
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: 2px;
  box-shadow: var(--ui-focus-ring);
}

.message-evidence__toggle--success {
  border-color: color-mix(in srgb, var(--ui-success) 28%, var(--ui-border));
  color: var(--ui-success);
}

.message-evidence__toggle--warning {
  border-color: color-mix(in srgb, var(--ui-warning) 32%, var(--ui-border));
  color: var(--ui-warning);
}

.message-evidence__summary {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}

.message-evidence__summary strong {
  font-weight: 650;
}

.message-evidence__score {
  border-left: 1px solid currentColor;
  margin-left: 1px;
  padding-left: 6px;
  font-variant-numeric: tabular-nums;
  opacity: 0.9;
}

.message-evidence__action {
  color: var(--ui-primary);
  font-weight: 600;
  white-space: nowrap;
}

.message-evidence__chevron {
  flex: 0 0 auto;
  color: var(--ui-icon);
  transition: transform 150ms ease;
}

.message-evidence__chevron--open {
  transform: rotate(180deg);
}

.message-evidence__panel {
  margin-top: 8px;
  overflow: hidden;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-popover);
  background: var(--ui-bg-subtle);
}

.message-evidence__panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--ui-divider);
  padding: 10px 12px;
}

.message-evidence__panel-title {
  color: var(--ui-text);
  font-size: 12px;
  font-weight: 650;
  line-height: 1.4;
}

.message-evidence__panel-hint {
  margin-top: 2px;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  line-height: 1.5;
}

.message-evidence__count {
  flex: 0 0 auto;
  border-radius: var(--ui-radius-pill);
  background: var(--ui-surface-muted);
  padding: 3px 7px;
  color: var(--ui-text-secondary);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.message-evidence__results {
  max-height: min(42vh, 360px);
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 4px;
}

.message-evidence__links {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 10px;
  border-top: 1px solid var(--ui-divider);
  padding: 8px 12px;
  color: var(--ui-text-tertiary);
  font-size: 11px;
}

.message-evidence__links a {
  display: inline-flex;
  max-width: 220px;
  align-items: center;
  gap: 4px;
  color: var(--ui-primary);
}

.message-evidence__links a span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.message-evidence__links a:hover {
  color: var(--ui-primary-hover);
  text-decoration: underline;
}

.message-evidence__links a:focus-visible {
  border-radius: 4px;
  outline: 2px solid var(--ui-focus-outline);
  outline-offset: 2px;
}

.message-trace {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 7px;
  margin-top: 10px;
  border: 1px solid color-mix(in srgb, var(--ui-danger) 28%, var(--ui-border));
  border-radius: var(--ui-radius-control);
  background: var(--ui-danger-subtle);
  padding: 7px 9px;
  color: var(--ui-text-secondary);
  font-size: 11px;
  line-height: 1.4;
}

.message-trace__label {
  flex: 0 0 auto;
  color: var(--ui-danger);
  font-weight: 650;
}

.message-trace code {
  min-width: 0;
  flex: 1 1 auto;
  overflow: hidden;
  color: var(--ui-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.message-trace__copy {
  flex: 0 0 auto;
}

@media (max-width: 639px) {
  .chat-message { gap: 8px; margin-bottom: 18px; }
  .chat-message__avatar { width: 30px; height: 30px; flex-basis: 30px; border-radius: 9px; }
  .chat-message__content,
  .chat-message--user .chat-message__content { max-width: calc(100% - 38px); }
  .chat-message__assistant-card { padding: 13px 14px; }

  .message-clarification__choice {
    min-height: 40px;
  }

  .message-evidence__toggle {
    width: 100%;
    min-height: 40px;
    border-radius: var(--ui-radius-control);
  }

  .message-evidence__summary {
    flex: 1 1 120px;
    flex-wrap: wrap;
    white-space: normal;
  }

  .message-evidence__action {
    margin-left: auto;
  }

  .message-evidence__panel-head {
    align-items: center;
  }

  .message-evidence__results {
    max-height: none;
  }

  .message-trace {
    min-height: 40px;
    flex-wrap: wrap;
  }

  .message-trace code {
    flex-basis: calc(100% - 92px);
  }
}

@media (prefers-reduced-motion: reduce) {
  .message-clarification__choice,
  .message-evidence__toggle,
  .message-evidence__chevron {
    transition: none;
  }
}
</style>

<style scoped>
.chat-message__user-bubble {
  color: var(--ui-text-on-primary);
  border-radius: var(--ui-radius-card) 4px var(--ui-radius-card) var(--ui-radius-card);
  background: linear-gradient(180deg, var(--ui-primary) 0%, var(--ui-primary-hover) 100%);
  box-shadow: 0 7px 20px color-mix(in srgb, var(--ui-primary) 16%, transparent);
  padding: 12px 16px;
  font-size: 14px;
  line-height: 1.7;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

/* 用户气泡本身已经是蓝色，使用更深的品牌色标记选区，并保持白字。 */
.chat-message__user-bubble::selection {
  color: var(--ui-text-on-primary);
  background: var(--ui-primary-pressed);
}
</style>
