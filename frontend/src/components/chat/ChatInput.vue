<template>
  <div class="chat-composer">
    <n-input
      ref="inputRef"
      v-model:value="text"
      type="textarea"
      :autosize="{ minRows: 2, maxRows: 6 }"
      placeholder="输入问题，开始一场有依据的对话…"
      :bordered="false"
      :input-props="{ 'aria-label': '输入问题' }"
      class="chat-composer__input"
      @keydown.enter.exact="handleEnter"
    />
    <div class="chat-composer__footer">
      <div class="chat-composer__config" aria-label="检索配置">
        <n-popover
          trigger="click"
          placement="top-start"
          :show-arrow="false"
        >
          <template #trigger>
            <button type="button" class="composer-setting composer-setting--kb" aria-label="配置知识库范围">
              <n-icon :size="16"><FolderOpenOutline /></n-icon>
              <span class="composer-setting__content">
                <span class="composer-setting__label">知识库</span>
                <span class="composer-setting__value">{{ knowledgeBaseSummary }}</span>
              </span>
              <n-icon :size="14" class="composer-setting__chevron"><ChevronDownOutline /></n-icon>
            </button>
          </template>
          <div class="composer-popover">
            <p>知识库范围</p>
            <span>仅显示当前账号可访问的知识库</span>
            <n-select
              v-model:value="chatStore.selectedKbIds"
              :options="kbOptions"
              multiple
              clearable
              placeholder="选择知识库"
              class="composer-popover__select"
              :input-props="{ 'aria-label': '选择知识库' }"
              :max-tag-count="1"
            />
          </div>
        </n-popover>
        <n-popover
          trigger="click"
          placement="top-start"
          :show-arrow="false"
        >
          <template #trigger>
            <button type="button" class="composer-setting composer-setting--method" aria-label="配置检索方式">
              <n-icon :size="16"><SearchOutline /></n-icon>
              <span class="composer-setting__content">
                <span class="composer-setting__label">检索方式</span>
                <span class="composer-setting__value">{{ methodSummary }}</span>
              </span>
              <n-icon :size="14" class="composer-setting__chevron"><ChevronDownOutline /></n-icon>
            </button>
          </template>
          <div class="composer-popover">
            <p>检索方式</p>
            <span>混合检索通常能获得更平衡的召回效果</span>
            <n-select
              v-model:value="chatStore.searchConfig.method"
              :options="methodOptions"
              placeholder="选择检索方式"
              class="composer-popover__select"
              :input-props="{ 'aria-label': '选择检索方式' }"
            />
          </div>
        </n-popover>
        <n-popover
          v-if="tagOptions.length"
          trigger="click"
          placement="top-start"
          :show-arrow="false"
        >
          <template #trigger>
            <button type="button" class="composer-setting composer-setting--tag" aria-label="按标签筛选">
              <n-icon :size="16"><PricetagOutline /></n-icon>
              <span class="composer-setting__content">
                <span class="composer-setting__label">标签筛选</span>
                <span class="composer-setting__value">{{ tagSummary }}</span>
              </span>
              <n-icon :size="14" class="composer-setting__chevron"><ChevronDownOutline /></n-icon>
            </button>
          </template>
          <div class="composer-popover">
            <p>标签筛选</p>
            <span>标签用于让匹配内容在检索结果中优先排序</span>
            <n-select
              v-model:value="chatStore.searchConfig.tags"
              :options="tagOptions"
              multiple
              clearable
              placeholder="不限标签"
              class="composer-popover__select"
              :input-props="{ 'aria-label': '按标签筛选' }"
              :max-tag-count="1"
            />
          </div>
        </n-popover>
        <n-tooltip trigger="hover" placement="top">
          <template #trigger>
            <button
              type="button"
              class="composer-setting composer-setting--rerank"
              :class="{ 'is-active': chatStore.searchConfig.rerank }"
              :aria-pressed="chatStore.searchConfig.rerank"
              @click="chatStore.searchConfig.rerank = !chatStore.searchConfig.rerank"
            >
              <n-icon :size="16"><SparklesOutline /></n-icon>
              <span class="composer-setting__content">
                <span class="composer-setting__label">智能重排</span>
                <span class="composer-setting__value">{{ chatStore.searchConfig.rerank ? '已开启' : '已关闭' }}</span>
              </span>
            </button>
          </template>
          <div class="max-w-xs text-xs leading-relaxed">
            重排（Rerank）：先快速召回候选片段，再用模型评估相关度并重新排序。开启后更精准，但会略增耗时和成本。
          </div>
        </n-tooltip>
      </div>
      <div class="chat-composer__actions">
        <span class="chat-composer__shortcut">Enter 发送 · Shift + Enter 换行</span>
        <n-button v-if="chatStore.isStreaming" class="chat-composer__stop" size="small" @click="chatStore.stopStreaming()">
          <template #icon><n-icon><StopOutline /></n-icon></template>
          停止
        </n-button>
        <n-button
          class="chat-composer__send" type="primary" circle size="medium" :disabled="!text.trim() || chatStore.isStreaming"
          aria-label="发送问题"
          @click="handleSend"
        >
          <template #icon><n-icon><SendOutline /></n-icon></template>
        </n-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { NInput, NSelect, NButton, NIcon, NTooltip, NPopover } from 'naive-ui'
import { SendOutline, StopOutline, FolderOpenOutline, SearchOutline, PricetagOutline, SparklesOutline, ChevronDownOutline } from '@vicons/ionicons5'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeStore } from '@/stores/knowledge'
import { getDocumentTags } from '@/api/knowledge'

const chatStore = useChatStore()
const kbStore = useKnowledgeStore()
const text = ref('')
const inputRef = ref(null)
const kbOptions = computed(() =>
  kbStore.list.map(kb => ({ label: kb.name, value: kb.id }))
)

const knowledgeBaseSummary = computed(() => {
  const selected = chatStore.selectedKbIds || []
  if (!selected.length) return '按需选择'
  const selectedNames = kbStore.list.filter(kb => selected.includes(kb.id)).map(kb => kb.name)
  if (selectedNames.length === 1) return selectedNames[0]
  return `${selected.length} 个已选`
})

const availableTags = ref([])
const tagOptions = computed(() => availableTags.value.map(t => ({ label: t, value: t })))

// 所选知识库变化时刷新可用标签，并把已选但不再可用的标签剔除，避免发送无效过滤
async function refreshTags() {
  if (!chatStore.selectedKbIds.length) {
    availableTags.value = []
    chatStore.searchConfig.tags = []
    return
  }
  try {
    availableTags.value = await getDocumentTags(chatStore.selectedKbIds)
  } catch {
    availableTags.value = []
  }
  const allowed = new Set(availableTags.value)
  chatStore.searchConfig.tags = (chatStore.searchConfig.tags || []).filter(t => allowed.has(t))
}

onMounted(async () => {
  await kbStore.fetchList()
  // selectedKbIds 存在 localStorage 且跨用户共用：仅保留当前用户可访问的知识库，
  // 剔除他人残留/无权访问的选择 —— 新用户登录后默认即为空。
  const allowed = new Set(kbStore.list.map(kb => kb.id))
  chatStore.selectedKbIds = chatStore.selectedKbIds.filter(id => allowed.has(id))
  await refreshTags()
})

watch(() => chatStore.selectedKbIds, refreshTags, { deep: true })

const methodOptions = [
  { label: '混合检索（向量 + 关键词）', value: 'hybrid' },
  { label: '向量检索', value: 'vector' },
  { label: '关键词检索', value: 'keyword' },
]

const methodSummary = computed(() => {
  const current = methodOptions.find(option => option.value === chatStore.searchConfig.method)
  return current?.label?.replace(/（.*）/, '') || '混合检索'
})

const tagSummary = computed(() => {
  const selected = chatStore.searchConfig.tags || []
  if (!selected.length) return '不限'
  return selected.length === 1 ? selected[0] : `${selected.length} 个已选`
})

function handleSend() {
  if (!text.value.trim() || chatStore.isStreaming) return
  // 是否需要知识库由后端智能路由决定：知识库问答仍会在后端强制校验权限与选择范围。
  chatStore.sendMessage(text.value.trim())
  text.value = ''
}

// 中文输入法选词时也会产生 Enter；组合输入尚未结束时不能发送消息。
function handleEnter(event) {
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  handleSend()
}

// 供欢迎态示例与后续外部调用安全复用：不发送消息、不改变知识库或检索设置。
function setText(value) {
  text.value = typeof value === 'string' ? value : ''
  focus()
}

function focus() {
  nextTick(() => {
    if (inputRef.value?.focus) {
      inputRef.value.focus()
      return
    }
    inputRef.value?.$el?.querySelector?.('textarea, input')?.focus()
  })
}

defineExpose({ setText, focus })
</script>

<style scoped>
.chat-composer {
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
  box-shadow: var(--ui-shadow-card);
  padding: 14px 14px 11px;
  transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}

.chat-composer:focus-within {
  border-color: var(--ui-border-focus);
  box-shadow: var(--ui-focus-ring), var(--ui-shadow-card);
}

.chat-composer__input :deep(.n-input__textarea-el) {
  min-height: 58px !important;
  padding: 1px 2px 6px;
  color: var(--ui-text);
  font-size: 14px;
  line-height: 1.75;
}

.chat-composer__input :deep(.n-input__placeholder) {
  color: var(--ui-placeholder);
}

.chat-composer__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 9px;
  padding-top: 10px;
  border-top: 1px solid var(--ui-divider);
}

.chat-composer__config {
  display: flex;
  flex: 1 1 auto;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px;
  min-width: 0;
}

.composer-setting {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  height: 38px;
  padding: 0 9px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control);
  color: var(--ui-text-secondary);
  background: var(--ui-surface-muted);
  cursor: pointer;
  font: inherit;
  text-align: left;
  transition: border-color .18s ease, background .18s ease, box-shadow .18s ease, transform .18s ease;
}

.composer-setting:hover {
  border-color: var(--ui-border-strong);
  background: var(--ui-surface-hover);
}

.composer-setting:focus-visible {
  outline: 0;
  border-color: var(--ui-border-focus);
  box-shadow: var(--ui-focus-ring);
}

.composer-setting > :deep(.n-icon) { flex: 0 0 auto; color: var(--ui-primary); }
.composer-setting--kb { flex: 0 1 182px; max-width: 220px; }
.composer-setting--method { flex: 0 1 164px; max-width: 195px; }
.composer-setting--tag { flex: 0 1 150px; max-width: 185px; }
.composer-setting--rerank { flex: 0 0 auto; }

.composer-setting__content {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
}

.composer-setting__label {
  color: var(--ui-text-tertiary);
  font-size: 10px;
  font-weight: 680;
  letter-spacing: .035em;
  line-height: 1;
  white-space: nowrap;
}

.composer-setting__value {
  overflow: hidden;
  color: var(--ui-text-secondary);
  font-size: 11px;
  font-weight: 600;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.composer-setting__chevron {
  flex: 0 0 auto;
  margin-left: auto;
  color: var(--ui-icon) !important;
}

.composer-setting--rerank.is-active {
  border-color: var(--ui-border-focus);
  background: var(--ui-primary-subtle);
}

.composer-setting--rerank.is-active > :deep(.n-icon) { color: var(--ui-primary); }
.composer-setting--rerank.is-active .composer-setting__value { color: var(--ui-primary); }

.composer-popover { width: min(280px, calc(100vw - 32px)); padding: 2px; }
.composer-popover p { margin: 0; color: var(--ui-text); font-size: 13px; font-weight: 700; }
.composer-popover > span { display: block; margin-top: 5px; color: var(--ui-text-secondary); font-size: 11px; line-height: 1.55; }
.composer-popover__select { width: 100%; margin-top: 11px; }

.chat-composer__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 8px;
}

.chat-composer__shortcut {
  color: var(--ui-text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

:deep(.chat-composer__stop.n-button) {
  --n-border: 1px solid var(--ui-border) !important;
  --n-color-hover: var(--ui-surface-hover) !important;
  --n-border-hover: 1px solid var(--ui-border-strong) !important;
  border-radius: 9px;
}

.chat-composer__send {
  box-shadow: var(--ui-shadow-card);
  transition: transform .18s ease, box-shadow .18s ease;
}

.chat-composer__send:not(.n-button--disabled):hover {
  transform: translateY(-1px);
  box-shadow: var(--ui-shadow-float);
}

@media (max-width: 860px) {
  .chat-composer__footer { align-items: flex-end; }
  .chat-composer__shortcut { display: none; }
}

@media (max-width: 639px) {
  .chat-composer { padding: 12px 12px 10px; }
  .chat-composer__input :deep(.n-input__textarea-el) { min-height: 52px !important; }
  .chat-composer__footer { align-items: stretch; flex-direction: column; }
  .chat-composer__config { flex-wrap: nowrap; gap: 6px; overflow-x: auto; padding-bottom: 2px; scrollbar-width: none; }
  .chat-composer__config::-webkit-scrollbar { display: none; }
  .composer-setting,
  .composer-setting--kb,
  .composer-setting--method,
  .composer-setting--tag,
  .composer-setting--rerank { flex: 0 0 auto; max-width: 180px; }
  .chat-composer__actions { justify-content: flex-end; }
}
</style>
