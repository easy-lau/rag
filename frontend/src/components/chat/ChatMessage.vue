<template>
  <div class="flex gap-3 mb-4" :class="isUser ? 'justify-end' : 'justify-start'">
    <!-- AI avatar -->
    <div v-if="!isUser" class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm shrink-0 mt-1">
      <n-icon><HardwareChipOutline /></n-icon>
    </div>

    <div class="max-w-[85%] sm:max-w-[75%] min-w-0">
      <!-- User bubble -->
      <div v-if="isUser" class="bg-blue-500 text-white px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed break-words">
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

        <!-- 知识库来源：点击标题在当前页只读预览（所有命中文档都列出） -->
        <div v-if="message.content && sources.length && showSources" class="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span class="text-xs text-gray-400">知识库来源：</span>
          <a
            v-for="(src, i) in sources"
            :key="i"
            class="inline-flex items-center gap-1 max-w-[200px] text-xs text-blue-500 hover:text-blue-600 hover:underline cursor-pointer"
            :title="src.filename"
            @click="openSource(src)"
          >
            <n-icon :size="13"><DocumentTextOutline /></n-icon>
            <span class="truncate">{{ src.filename }}</span>
          </a>
        </div>

        <!-- 参考来源：仅列出带数据来源链接的文档，点击新标签页打开外部 URL；没有则整行不显示 -->
        <div v-if="message.content && urlSources.length && showSources" class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span class="text-xs text-gray-400">参考来源：</span>
          <a
            v-for="(src, i) in urlSources"
            :key="i"
            class="inline-flex items-center gap-1 max-w-[220px] text-xs text-blue-500 hover:text-blue-600 hover:underline cursor-pointer"
            :href="src.source_url"
            target="_blank"
            rel="noopener noreferrer"
            :title="src.source_url"
          >
            <n-icon :size="13"><LinkOutline /></n-icon>
            <span class="truncate">{{ src.filename }}</span>
          </a>
        </div>

        <!-- Actions -->
        <div v-if="message.content" class="flex items-center gap-3 mt-3 pt-2 border-t border-gray-100 dark:border-gray-700">
          <n-button text size="tiny" @click="copy">
            <template #icon><n-icon><CopyOutline /></n-icon></template>
            复制
          </n-button>
          <n-button text size="tiny" @click="$emit('retry')">
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
import { computed } from 'vue'
import { NButton, NIcon, NSpin, useMessage } from 'naive-ui'
import { CopyOutline, RefreshOutline, HardwareChipOutline, PersonOutline, DocumentTextOutline, StopCircleOutline, LinkOutline } from '@vicons/ionicons5'
import { useClipboard } from '@vueuse/core'
import { renderMarkdown } from '@/utils/markdown'
import { useSettingsStore } from '@/stores/settings'

const props = defineProps({ message: Object })
const emit = defineEmits(['retry', 'preview'])

const msg = useMessage()
const { copy: copyText } = useClipboard()
const settingsStore = useSettingsStore()

const isUser = computed(() => props.message.role === 'user')
// 系统设置「显示参考来源」总开关：关闭则隐藏所有来源行（默认显示）
const showSources = computed(() => settingsStore.data.show_sources !== false)

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

// 去重后的来源文档列表（按 doc_id / 文件名去重）
const sources = computed(() => {
  const list = props.message.sources || []
  const seen = new Set()
  const out = []
  for (const s of list) {
    const key = s.doc_id || s.filename
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(s)
  }
  return out
})

// 参考来源：只取带数据来源链接的来源，点击跳转外部 URL
const urlSources = computed(() => sources.value.filter(s => s.source_url))

function openSource(src) {
  // 不跳转，交给父组件在当前页弹出只读预览
  emit('preview', src)
}

function copy() {
  copyText(props.message.content)
  msg.success('已复制')
}

function handleMarkdownClick(e) {
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
