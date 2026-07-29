<template>
  <div class="min-w-0" aria-live="polite">
    <!-- Step nodes -->
    <div class="flex min-w-0 items-center">
      <template v-for="(step, i) in steps" :key="step.key">
        <div class="flex w-11 shrink-0 flex-col items-center sm:w-12">
          <div
            class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300"
            :class="stepClass(step.status)"
          >
            <n-icon v-if="step.status === 'done'" :size="14"><CheckmarkOutline /></n-icon>
            <span v-else-if="step.status === 'active'">
              <n-spin :size="14" />
            </span>
            <span v-else class="text-gray-400">{{ i + 1 }}</span>
          </div>
          <span class="mt-1 w-full truncate text-center text-[10px] leading-4" :class="labelClass(step.status)">{{ step.label }}</span>
        </div>
        <div
          v-if="i < steps.length - 1"
          class="min-w-0 flex-1 h-0.5 mx-1 mb-5 transition-colors duration-300"
          :class="step.status === 'done' ? 'bg-green-400' : 'bg-gray-200 dark:bg-gray-600'"
        />
      </template>
    </div>
  </div>
</template>

<script setup>
import { NSpin, NIcon } from 'naive-ui'
import { CheckmarkOutline } from '@vicons/ionicons5'
import { useSearchStore } from '@/stores/search'
import { storeToRefs } from 'pinia'

const { steps } = storeToRefs(useSearchStore())

function stepClass(status) {
  if (status === 'done')   return 'bg-green-500 text-white'
  if (status === 'active') return 'bg-blue-500 text-white ring-2 ring-blue-300 ring-offset-1 dark:ring-blue-400 dark:ring-offset-gray-800'
  return 'bg-gray-200 dark:bg-gray-600 text-gray-400'
}

function labelClass(status) {
  if (status === 'done')   return 'text-green-600 dark:text-green-400 font-medium'
  if (status === 'active') return 'text-blue-500 font-medium'
  return 'text-gray-400'
}
</script>
