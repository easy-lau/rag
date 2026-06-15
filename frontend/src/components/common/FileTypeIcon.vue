<template>
  <div class="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold shrink-0" :style="{ background: color }">
    {{ label }}
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({ type: String })
const MAP = {
  pdf:  { label: 'PDF',  color: '#EF4444' },
  docx: { label: 'DOC',  color: '#3B82F6' },
  doc:  { label: 'DOC',  color: '#3B82F6' },
  pptx: { label: 'PPT',  color: '#F97316' },
  ppt:  { label: 'PPT',  color: '#F97316' },
  txt:  { label: 'TXT',  color: '#6B7280' },
  md:   { label: 'MD',   color: '#8B5CF6' },
}
const info = computed(() => MAP[props.type?.toLowerCase()] || { label: 'DOC', color: '#6B7280' })
const label = computed(() => info.value.label)
const color = computed(() => info.value.color)
</script>
