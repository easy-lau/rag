<template>
  <section class="w-full max-w-3xl mx-auto" aria-labelledby="chat-welcome-title">
    <div class="relative mb-4 overflow-hidden rounded-2xl border border-blue-100 bg-gradient-to-br from-blue-50 via-white to-indigo-50 p-5 shadow-sm dark:border-blue-900/60 dark:from-blue-950/30 dark:via-gray-800 dark:to-indigo-950/20 sm:p-6">
      <span class="pointer-events-none absolute -right-8 -top-10 h-32 w-32 rounded-full bg-blue-300/20 blur-2xl dark:bg-blue-500/10" aria-hidden="true" />
      <span class="pointer-events-none absolute right-20 top-9 h-12 w-12 rounded-full border border-indigo-200/60 dark:border-indigo-700/30" aria-hidden="true" />
      <div class="relative flex items-start gap-3">
        <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-500 text-white shadow-sm shadow-blue-500/30">
          <n-icon :size="20"><ChatbubbleEllipsesOutline /></n-icon>
        </div>
        <div class="min-w-0">
          <p class="text-xs font-semibold tracking-[0.16em] text-blue-600 uppercase dark:text-blue-300">智能助手</p>
          <h2 id="chat-welcome-title" class="mt-1 text-xl font-semibold tracking-tight text-gray-900 dark:text-white sm:text-2xl">
            {{ greeting }}，今天想解决什么问题？
          </h2>
          <p class="mt-1 text-xs leading-5 text-gray-500 dark:text-gray-400">
            在 {{ siteName }} 中选择知识库检索资料，或直接进行通用问答、写作和系统使用咨询。
          </p>
        </div>
      </div>

      <div class="relative mt-5" role="group" aria-label="示例问题">
        <p class="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">试试这样问</p>
        <div class="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <button
            v-for="example in examples"
            :key="example"
            type="button"
            class="rounded-xl border border-white/90 bg-white/80 px-3 py-2.5 text-left text-xs leading-5 text-gray-600 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:border-gray-700 dark:bg-gray-800/80 dark:text-gray-300 dark:hover:border-blue-800 dark:hover:bg-blue-950/40 dark:hover:text-blue-300"
            :aria-label="`使用示例问题：${example}`"
            @click="$emit('select-example', example)"
          >
            {{ example }}
          </button>
        </div>
      </div>
    </div>

    <!-- 保持输入框由 ChatInput 管理，欢迎态仅包裹并提供示例触发。 -->
    <slot name="input" />
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { NIcon } from 'naive-ui'
import { ChatbubbleEllipsesOutline } from '@vicons/ionicons5'

const props = defineProps({
  siteName: { type: String, default: '知识工作台' },
  userName: { type: String, default: '' },
  examples: {
    type: Array,
    default: () => [
      '公司的报销流程是什么？',
      '如何创建知识库并上传文档？',
      '帮我把这段通知润色得更专业',
    ],
  },
})

defineEmits(['select-example'])

const greeting = computed(() => props.userName ? `你好，${props.userName}` : '你好')
</script>
