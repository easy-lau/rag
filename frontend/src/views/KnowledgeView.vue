<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="flex items-center justify-end mb-6">
      <n-button v-if="authStore.hasPerm('kb:write')" type="primary" @click="showCreate = true">
        <template #icon><n-icon><AddOutline /></n-icon></template>
        创建知识库
      </n-button>
    </div>

    <n-spin :show="kbStore.loading">
      <div v-if="!kbStore.list.length && !kbStore.loading"
        class="flex flex-col items-center justify-center py-20 text-gray-400">
        <n-icon :size="40" class="mb-2"><LibraryOutline /></n-icon>
        <p class="text-sm">暂无知识库，点击右上角创建</p>
      </div>
      <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        <div
          v-for="kb in kbStore.list" :key="kb.id"
          class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow"
        >
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2 min-w-0">
              <div class="w-3 h-3 rounded-full shrink-0" :style="{ background: kb.icon_color }" />
              <h3 class="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">{{ kb.name }}</h3>
            </div>
            <div v-if="authStore.hasPerm('kb:write')" class="flex items-center gap-2 shrink-0 ml-2">
              <n-button text size="tiny" @click="openEdit(kb)">
                <template #icon><n-icon><PencilOutline /></n-icon></template>
              </n-button>
              <n-popconfirm @positive-click="handleDelete(kb)">
                <template #trigger>
                  <n-button text size="tiny" type="error">
                    <template #icon><n-icon><TrashOutline /></n-icon></template>
                  </n-button>
                </template>
                确定删除该知识库？（需先清空其下文档）
              </n-popconfirm>
            </div>
          </div>
          <p class="text-xs text-gray-500 dark:text-gray-400 mb-2 line-clamp-1">{{ kb.description || '暂无描述' }}</p>
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-gray-400">{{ kb.doc_count }} 个文档</span>
            <n-button size="tiny" @click="$router.push({ name: 'documents', query: { kb: kb.id } })">
              管理文档
            </n-button>
          </div>
          <div class="flex items-center justify-between text-[11px] text-gray-400 dark:text-gray-500 pt-2 border-t border-gray-100 dark:border-gray-700">
            <span class="truncate">创建人：{{ kb.created_by_name || '—' }}</span>
            <span class="shrink-0 ml-2">{{ new Date(kb.created_at).toLocaleString('zh-CN') }}</span>
          </div>
        </div>
      </div>
    </n-spin>

    <n-modal v-model:show="showCreate" to="#app" preset="card" title="创建知识库" style="width: 90vw; max-width: 384px">
      <n-form :model="form" label-placement="top">
        <n-form-item label="名称" required>
          <n-input v-model:value="form.name" placeholder="请输入知识库名称" />
        </n-form-item>
        <n-form-item label="描述">
          <n-input v-model:value="form.description" type="textarea" :rows="3" placeholder="可选描述" />
        </n-form-item>
        <n-form-item label="颜色标识">
          <div class="flex gap-2">
            <div
              v-for="c in colors" :key="c"
              class="w-6 h-6 rounded-full cursor-pointer transition-transform"
              :style="{
                background: c,
                boxShadow: form.icon_color === c ? `0 0 0 2px #fff, 0 0 0 4px ${c}` : 'none',
                transform: form.icon_color === c ? 'scale(1.2)' : 'scale(1)'
              }"
              @click="form.icon_color = c"
            />
          </div>
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button @click="showCreate = false">取消</n-button>
          <n-button type="primary" :loading="creating" @click="handleCreate">创建</n-button>
        </div>
      </template>
    </n-modal>

    <n-modal v-model:show="showEdit" to="#app" preset="card" title="编辑知识库" style="width: 90vw; max-width: 360px">
      <n-form :model="editForm" label-placement="top">
        <n-form-item label="名称" required>
          <n-input v-model:value="editForm.name" placeholder="请输入知识库名称" />
        </n-form-item>
        <n-form-item label="描述">
          <n-input v-model:value="editForm.description" type="textarea" :rows="3" placeholder="可选描述" />
        </n-form-item>
        <n-form-item label="颜色标识">
          <div class="flex gap-2">
            <div
              v-for="c in colors" :key="c"
              class="w-6 h-6 rounded-full cursor-pointer transition-transform"
              :style="{
                background: c,
                boxShadow: editForm.icon_color === c ? `0 0 0 2px #fff, 0 0 0 4px ${c}` : 'none',
                transform: editForm.icon_color === c ? 'scale(1.2)' : 'scale(1)'
              }"
              @click="editForm.icon_color = c"
            />
          </div>
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button @click="showEdit = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleEdit">保存</n-button>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { NButton, NIcon, NSpin, NModal, NForm, NFormItem, NInput, NPopconfirm, useMessage } from 'naive-ui'
import { AddOutline, LibraryOutline, TrashOutline, PencilOutline } from '@vicons/ionicons5'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useAuthStore } from '@/stores/auth'

const kbStore = useKnowledgeStore()
const authStore = useAuthStore()
const message = useMessage()

async function handleDelete(kb) {
  try {
    await kbStore.remove(kb.id)
    message.success('知识库已删除')
  } catch (e) {
    // 后端校验：知识库下仍有文档时会拒绝删除，提示用户先清空文档
    message.warning(e?.response?.data?.detail || '删除失败，请重试')
  }
}
const showCreate = ref(false)
const creating = ref(false)
const form = ref({ name: '', description: '', icon_color: '#3B82F6' })
const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6B7280']

const showEdit = ref(false)
const saving = ref(false)
const editingId = ref(null)
const editForm = ref({ name: '', description: '', icon_color: '#3B82F6' })

onMounted(() => kbStore.fetchList())

async function handleCreate() {
  if (!form.value.name.trim()) return
  creating.value = true
  try {
    await kbStore.create({ ...form.value })
    showCreate.value = false
    form.value = { name: '', description: '', icon_color: '#3B82F6' }
  } finally {
    creating.value = false
  }
}

function openEdit(kb) {
  editingId.value = kb.id
  editForm.value = { name: kb.name, description: kb.description || '', icon_color: kb.icon_color }
  showEdit.value = true
}

async function handleEdit() {
  if (!editForm.value.name.trim()) return
  saving.value = true
  try {
    await kbStore.update(editingId.value, { ...editForm.value })
    showEdit.value = false
  } finally {
    saving.value = false
  }
}
</script>
