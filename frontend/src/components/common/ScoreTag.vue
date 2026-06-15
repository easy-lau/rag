<template>
  <span class="text-xs font-semibold px-1.5 py-0.5 rounded" :class="cls">{{ display }}</span>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ score: Number })
const display = computed(() => props.score?.toFixed(2) ?? '--')
const cls = computed(() => {
  const s = props.score ?? 0
  if (s >= 0.8) return 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
  if (s >= 0.6) return 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300'
  return 'bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400'
})
</script>
