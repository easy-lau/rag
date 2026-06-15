<template>
  <aside class="w-80 flex flex-col border-l border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto shrink-0">
    <!-- Results header -->
    <div class="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <span class="font-medium text-sm text-gray-800 dark:text-gray-200">检索结果</span>
        <span v-if="searchStore.totalCount" class="text-xs text-gray-500">
          共检索到 {{ searchStore.totalCount }} 条结果
        </span>
      </div>
    </div>

    <!-- Result list -->
    <div class="flex-1 overflow-y-auto px-2 py-2">
      <template v-if="searchStore.results.length">
        <DocumentResultItem
          v-for="(item, i) in searchStore.results"
          :key="item.id"
          :item="item"
          :rank="i + 1"
        />
        <div class="text-center mt-2">
          <n-button text size="tiny" class="text-blue-500">查看更多</n-button>
        </div>
      </template>
      <div v-else class="flex flex-col items-center justify-center h-32 text-gray-400 text-sm">
        <n-icon :size="32" class="mb-2"><SearchOutline /></n-icon>
        <span>发送问题后显示检索结果</span>
      </div>
    </div>

    <!-- Search process -->
    <div class="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
      <div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">检索过程</div>
      <SearchProcess />

      <!-- Meta info -->
      <div v-if="searchStore.searchMeta.method" class="mt-3 space-y-1">
        <div class="flex justify-between text-xs">
          <span class="text-gray-500">检索方式</span>
          <span class="text-gray-700 dark:text-gray-300">{{ methodLabel }}</span>
        </div>
        <div class="flex justify-between text-xs">
          <span class="text-gray-500">Top K</span>
          <span class="text-gray-700 dark:text-gray-300">{{ searchStore.searchMeta.top_k }}</span>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { NButton, NIcon } from 'naive-ui'
import { SearchOutline } from '@vicons/ionicons5'
import { useSearchStore } from '@/stores/search'
import DocumentResultItem from './DocumentResultItem.vue'
import SearchProcess from './SearchProcess.vue'

const searchStore = useSearchStore()
const methodLabel = computed(() => {
  const m = { hybrid: '混合检索（向量+关键词）', vector: '向量检索', keyword: '关键词检索' }
  return m[searchStore.searchMeta.method] || '--'
})
</script>
