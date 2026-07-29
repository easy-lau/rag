<template>
  <button
    type="button"
    class="group flex w-full items-start gap-2 rounded-xl border border-transparent p-3 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/60"
    :class="canPreview
      ? 'cursor-pointer hover:border-blue-100 hover:bg-blue-50/70 dark:hover:border-blue-900/70 dark:hover:bg-blue-950/25'
      : 'cursor-default'"
    :disabled="!canPreview"
    :aria-label="canPreview ? `预览检索文档：${item.filename || '未命名文档'}` : undefined"
    :title="canPreview ? '预览文档' : undefined"
    @click="openPreview"
  >
    <span class="text-xs font-bold text-gray-400 w-4 shrink-0 mt-0.5">{{ rank }}</span>
    <FileTypeIcon :type="item.file_type" class="shrink-0" />
    <div class="flex-1 min-w-0">
      <p class="text-sm font-medium text-gray-800 dark:text-gray-200 truncate group-hover:text-blue-700 dark:group-hover:text-blue-200">{{ item.filename }}</p>
      <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2 leading-relaxed">{{ item.content }}</p>
    </div>
    <ScoreTag :score="item.score" class="shrink-0 mt-0.5" />
  </button>
</template>

<script setup>
import { computed } from 'vue'
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import ScoreTag from '@/components/common/ScoreTag.vue'

const props = defineProps({ item: Object, rank: Number })
const emit = defineEmits(['preview'])

const canPreview = computed(() => Boolean(props.item?.kb_id && props.item?.doc_id))

function openPreview() {
  if (!canPreview.value) return
  emit('preview', props.item)
}
</script>
