<template>
  <div class="flex gap-3 mb-4" :class="isUser ? 'justify-end' : 'justify-start'">
    <!-- AI avatar -->
    <div v-if="!isUser" class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm shrink-0 mt-1">
      <n-icon><HardwareChipOutline /></n-icon>
    </div>

    <div class="max-w-[85%] sm:max-w-[75%] min-w-0">
      <!-- User bubble -->
      <div
        v-if="isUser"
        class="chat-message__user-bubble px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed break-words"
        @copy="handleUserCopy"
      >
        {{ message.content }}
      </div>

      <!-- AI card -->
      <div v-else class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
        <div
          v-if="message.content"
          class="markdown-body text-sm text-gray-800 dark:text-gray-200"
          v-html="rendered"
          @click="handleMarkdownClick"
        />
        <div v-else-if="message.stopped" class="flex items-center gap-2 text-gray-400 text-sm">
          <n-icon><StopCircleOutline /></n-icon><span>已停止生成</span>
        </div>
        <div v-else class="flex items-center gap-2 text-gray-400 text-sm">
          <n-spin size="small" /><span>思考中...</span>
        </div>

        <section
          v-if="clarification"
          class="message-clarification"
          aria-label="选择知识库适用范围"
        >
          <div class="message-clarification__heading">
            <span class="message-clarification__eyebrow">{{ clarificationEyebrow }}</span>
            <strong>{{ clarification.question || '请选择本次查询的适用范围' }}</strong>
          </div>

          <p
            v-if="clarification.acknowledged && !clarification.invalidated && !clarification.submitted"
            class="message-clarification__hint"
          >
            选择后会沿用当前问题重新检索；系统不会把未选择的资料混入答案。
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
                v-if="clarification.choices.length > 1"
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
            v-else-if="clarification.acknowledged && !clarification.invalidated && !clarification.submitted"
            class="message-clarification__refinement"
            aria-live="polite"
          >
            当前候选范围较多，无法安全列成有限选项。请在输入框补充具体产品、版本、项目或制度名称。
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
            :aria-label="`本条回答实际使用了 ${sourceDocumentCount} 篇知识库文档、${sources.length} 个片段`"
          >
            <div class="message-evidence__panel-head">
              <div>
                <p class="message-evidence__panel-title">本条回答使用的知识库片段</p>
                <p class="message-evidence__panel-hint">这里仅展示实际进入回答上下文的片段；右侧候选可能更多。</p>
              </div>
              <span class="message-evidence__count">{{ sourceDocumentCount }} 篇 · {{ sources.length }} 个片段</span>
            </div>

            <div class="message-evidence__results">
              <DocumentEvidenceGroup
                v-for="(group, index) in sourceGroups"
                :key="group.key"
                :group="group"
                :rank="index + 1"
                :preview-enabled="canPreviewSources"
                :default-expanded="index === 0"
                fragment-label="采用片段"
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

        <!-- Actions -->
        <div v-if="message.content" class="flex items-center gap-3 mt-3 pt-2 border-t border-gray-100 dark:border-gray-700">
          <n-button text size="tiny" @click="copy">
            <template #icon><n-icon><CopyOutline /></n-icon></template>
            复制
          </n-button>
          <n-button text size="tiny" :disabled="chatStore.isStreaming" @click="$emit('retry', message)">
            <template #icon><n-icon><RefreshOutline /></n-icon></template>
            重新生成
          </n-button>
        </div>
      </div>

      <div class="text-xs text-gray-400 mt-1 px-1">
        {{ formatTime(message.created_at) }}
        <span v-if="message.tokens"> · {{ message.tokens }} tokens</span>
      </div>
    </div>

    <!-- User avatar -->
    <div v-if="isUser" class="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 shrink-0 mt-1">
      <n-icon><PersonOutline /></n-icon>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { NButton, NIcon, NSpin, useMessage } from 'naive-ui'
import { ChevronDownOutline, CopyOutline, RefreshOutline, HardwareChipOutline, PersonOutline, DocumentTextOutline, StopCircleOutline, LinkOutline } from '@vicons/ionicons5'
import { useClipboard } from '@vueuse/core'
import { renderMarkdown } from '@/utils/markdown'
import { useSettingsStore } from '@/stores/settings'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import DocumentEvidenceGroup from '@/components/search/DocumentEvidenceGroup.vue'
import { persistedAnswerSources } from '@/utils/chatEvidence'
import { groupEvidenceByDocument, safeExternalSourceUrl } from '@/utils/evidenceDocuments'
import { isClarificationSubmittable, normalizeEvidenceClarification } from '@/utils/chatClarification'

const props = defineProps({ message: Object })
const emit = defineEmits(['retry', 'preview', 'clarify'])

const msg = useMessage()
const { copy: copyText } = useClipboard()
const settingsStore = useSettingsStore()
const chatStore = useChatStore()
const authStore = useAuthStore()
const evidenceExpanded = ref(false)

const isUser = computed(() => props.message.role === 'user')
const clarification = computed(() => (
  isUser.value ? null : normalizeEvidenceClarification(props.message.clarification)
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
  if (clarification.value?.invalidated) return '选择已失效'
  if (!clarification.value?.acknowledged) return '正在保存'
  return '需要你确认'
})
const clarificationStatusText = computed(() => {
  const value = clarification.value
  if (!value || value.submitted) return ''
  if (value.invalidated) {
    if (value.invalid_reason === 'stream_aborted') {
      return '生成已停止，本次范围选择没有生效。请重新提问。'
    }
    return '本次澄清未能完成保存，以下选项已失效。请重新提问。'
  }
  if (!value.acknowledged) return '正在确认本次澄清是否已保存，确认前不能选择。'
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
// 系统设置「显示参考来源」总开关：关闭则隐藏所有来源行（默认显示）
const showSources = computed(() => settingsStore.data.show_sources !== false)
// 普通问答角色可查看本轮命中片段，但不能因此获得整篇文档读取能力。
// 只有后端同样允许的 doc:read 角色才显示可点击的全文预览行为。
const canPreviewSources = computed(() => authStore.hasPerm('doc:read'))

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
  if (['direct', 'related', 'irrelevant'].includes(role)) return role

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

const hasPersistedEvidence = computed(() => (
  !isUser.value
  && Boolean(props.message.content)
  && showSources.value
  && sources.value.length > 0
  && sources.value.some(source => source.retrieval_executed !== false)
  && props.message.retrieval_executed !== false
  && props.message.search_meta?.retrieval_executed !== false
))

function sourceSupportScore(source) {
  // answer_support 是“能否直接支撑答案”的专用指标；旧数据缺失时才回退到
  // 经过约束惩罚的 effective_score。原始 retrieval_score 只代表召回排序，
  // 不能在回答卡片上展示为答案匹配率。
  return finiteScore(source.answer_support) ?? finiteScore(source.effective_score)
}

const evidenceSummary = computed(() => {
  const directSources = sources.value.filter(source => source.evidence_role === 'direct')
  const partialSources = sources.value.filter(source => (
    source.jointly_selected && source.coverage_status === 'partial'
  ))
  const relatedSources = sources.value.filter(source => source.evidence_role === 'related')
  if (directSources.length) {
    const scores = directSources.map(sourceSupportScore).filter(score => score !== null)
    const score = scores.length ? Math.max(...scores) : null
    const percent = score === null ? null : Math.round(score * 100)
    const level = score === null ? '已验证' : (score >= 0.8 ? '高匹配' : (score >= 0.6 ? '中匹配' : '低匹配'))
    const scoreText = percent === null ? '' : `，证据支持分 ${percent}%`
    return {
      label: '知识库依据',
      level,
      percent,
      tone: score !== null && score < 0.6 ? 'warning' : 'success',
      actionLabel: '查看依据',
      ariaLabel: `本条回答使用了 ${directSources.length} 个知识库片段${scoreText}。该分数不是答案正确率`,
    }
  }
  if (partialSources.length) {
    return {
      label: '部分知识库依据',
      level: `${partialSources.length} 个片段`,
      percent: null,
      tone: 'warning',
      actionLabel: '查看依据',
      ariaLabel: `本条回答使用了 ${partialSources.length} 个片段支撑部分内容，但知识库证据尚未完整覆盖问题`,
    }
  }
  if (relatedSources.length) {
    return {
      label: '相近资料',
      level: `${relatedSources.length} 个片段`,
      percent: null,
      tone: 'warning',
      actionLabel: '查看资料',
      ariaLabel: `本条回答仅使用了 ${relatedSources.length} 个相近片段，不能作为直接回答依据`,
    }
  }
  return {
    label: '检索候选',
    level: `${sources.value.length} 个片段`,
    percent: null,
    tone: 'neutral',
    actionLabel: '查看结果',
    ariaLabel: `本条回答保存了 ${sources.value.length} 个尚待验证的检索片段`,
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

@media (max-width: 639px) {
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
  background: var(--ui-primary);
}

/* 用户气泡本身已经是蓝色，使用更深的品牌色标记选区，并保持白字。 */
.chat-message__user-bubble::selection {
  color: var(--ui-text-on-primary);
  background: var(--ui-primary-pressed);
}
</style>
