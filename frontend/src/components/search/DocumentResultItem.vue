<template>
  <div class="flex items-start gap-2 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors">
    <span class="text-xs font-bold text-gray-400 w-4 shrink-0 mt-0.5">{{ rank }}</span>
    <FileTypeIcon :type="item.file_type" class="shrink-0" />
    <div class="flex-1 min-w-0">
      <p class="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">{{ item.filename }}</p>
      <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2 leading-relaxed">{{ item.content }}</p>
    </div>
    <ScoreTag :score="item.score" class="shrink-0 mt-0.5" />
  </div>
</template>

<script setup>
import FileTypeIcon from '@/components/common/FileTypeIcon.vue'
import ScoreTag from '@/components/common/ScoreTag.vue'
defineProps({ item: Object, rank: Number })
</script>
