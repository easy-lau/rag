<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-5">
      <PageHeader title="知识库管理" description="创建并维护可检索的知识库，文档会在对应知识库内统一管理。">
        <template #actions>
          <n-button v-if="canCreateKb" type="primary" @click="showCreate = true">
            <template #icon><n-icon><AddOutline /></n-icon></template>
            创建知识库
          </n-button>
        </template>
      </PageHeader>

      <n-spin :show="kbStore.loading">
        <SurfaceCard
          v-if="!kbStore.list.length && !kbStore.loading"
          class="flex flex-col items-center justify-center py-20 text-center"
        >
          <n-icon :size="40" class="mb-2 text-slate-400"><LibraryOutline /></n-icon>
          <p class="text-sm text-slate-600 dark:text-slate-300">暂无知识库</p>
          <p class="mt-1 text-xs text-slate-400">
            {{ canCreateKb ? '可从右上角创建第一个知识库' : '当前角色尚未被授权可访问的知识库' }}
          </p>
        </SurfaceCard>
        <div v-else class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          <SurfaceCard
            v-for="kb in kbStore.list" :key="kb.id"
            padding="sm"
            class="kb-card"
          >
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2 min-w-0">
              <div class="w-3 h-3 rounded-full shrink-0" :style="{ background: kb.icon_color }" />
              <h3 class="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">{{ kb.name }}</h3>
            </div>
            <div v-if="canUpdateKb || canDeleteKb" class="flex items-center gap-2 shrink-0 ml-2">
              <n-button v-if="canUpdateKb" text size="tiny" aria-label="编辑知识库" @click="openEdit(kb)">
                <template #icon><n-icon><PencilOutline /></n-icon></template>
              </n-button>
              <n-button v-if="canDeleteKb" text size="tiny" type="error" aria-label="删除知识库" @click="openDelete(kb)">
                <template #icon><n-icon><TrashOutline /></n-icon></template>
              </n-button>
            </div>
          </div>
          <p class="text-xs text-gray-500 dark:text-gray-400 mb-2 line-clamp-1">{{ kb.description || '暂无描述' }}</p>
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-gray-400">{{ kb.doc_count }} 个文档</span>
            <n-button
              v-if="canOpenDocuments"
              size="tiny"
              @click="$router.push({ name: 'documents', query: { kb: kb.id } })"
            >
              {{ canManageDocuments ? '管理文档' : '查看文档' }}
            </n-button>
          </div>
          <div class="flex items-center justify-between text-[11px] text-gray-400 dark:text-gray-500 pt-2 border-t border-gray-100 dark:border-gray-700">
            <span class="truncate">创建人：{{ kb.created_by_name || '—' }}</span>
            <span class="shrink-0 ml-2">{{ new Date(kb.created_at).toLocaleString('zh-CN') }}</span>
          </div>
          </SurfaceCard>
        </div>
      </n-spin>

    <AppModal v-model:show="showCreate" title="创建知识库" width="min(92vw, 384px)" :loading="creating">
      <n-form :model="form" label-placement="top">
        <n-form-item label="名称" required>
          <n-input v-model:value="form.name" placeholder="请输入知识库名称" />
        </n-form-item>
        <n-form-item label="描述">
          <n-input v-model:value="form.description" type="textarea" :rows="3" placeholder="可选描述" />
        </n-form-item>
        <n-form-item label="颜色标识">
          <div class="flex gap-2">
            <button
              v-for="c in colors" :key="c"
              type="button"
              class="w-6 h-6 rounded-full cursor-pointer transition-transform border-0 p-0"
              :aria-label="`选择颜色 ${c}`"
              :aria-pressed="form.icon_color === c"
              :style="{
                background: c,
                boxShadow: form.icon_color === c ? `0 0 0 2px #fff, 0 0 0 4px ${c}` : 'none',
                transform: form.icon_color === c ? 'scale(1.2)' : 'scale(1)'
              }"
              @click="form.icon_color = c"
            ></button>
          </div>
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="creating" @click="showCreate = false">取消</n-button>
          <n-button type="primary" :loading="creating" @click="handleCreate">创建</n-button>
        </div>
      </template>
    </AppModal>

    <AppModal v-model:show="showEdit" title="编辑知识库" width="min(92vw, 360px)" :loading="saving">
      <n-form :model="editForm" label-placement="top">
        <n-form-item label="名称" required>
          <n-input v-model:value="editForm.name" placeholder="请输入知识库名称" />
        </n-form-item>
        <n-form-item label="描述">
          <n-input v-model:value="editForm.description" type="textarea" :rows="3" placeholder="可选描述" />
        </n-form-item>
        <n-form-item label="颜色标识">
          <div class="flex gap-2">
            <button
              v-for="c in colors" :key="c"
              type="button"
              class="w-6 h-6 rounded-full cursor-pointer transition-transform border-0 p-0"
              :aria-label="`选择颜色 ${c}`"
              :aria-pressed="editForm.icon_color === c"
              :style="{
                background: c,
                boxShadow: editForm.icon_color === c ? `0 0 0 2px #fff, 0 0 0 4px ${c}` : 'none',
                transform: editForm.icon_color === c ? 'scale(1.2)' : 'scale(1)'
              }"
              @click="editForm.icon_color = c"
            ></button>
          </div>
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="saving" @click="showEdit = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleEdit">保存</n-button>
        </div>
      </template>
    </AppModal>

    <DangerConfirm
      v-model:show="showDeleteConfirm"
      title="删除知识库？"
      :subject="pendingDelete?.name || ''"
      description="删除前需先清空该知识库下的文档。删除成功后，知识库配置无法恢复。"
      :loading="deleting"
      @confirm="confirmDelete"
      @cancel="pendingDelete = null"
    />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { NButton, NIcon, NSpin, NForm, NFormItem, NInput, useMessage } from 'naive-ui'
import { AddOutline, LibraryOutline, TrashOutline, PencilOutline } from '@vicons/ionicons5'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useAuthStore } from '@/stores/auth'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'

const kbStore = useKnowledgeStore()
const authStore = useAuthStore()
// 创建知识库没有现成对象可套用“指定知识库”范围，因此只允许全范围管理员创建。
// 已授权知识库的修改和删除分别由独立能力 + 后端对象范围校验控制。
const canCreateKb = computed(() => authStore.hasPerm('kb:create') && authStore.hasAllKbScope)
const canUpdateKb = computed(() => authStore.hasPerm('kb:update'))
const canDeleteKb = computed(() => authStore.hasPerm('kb:delete'))
const canOpenDocuments = computed(() => (
  authStore.hasPerm('menu:documents') && authStore.hasPerm('doc:read')
))
const canManageDocuments = computed(() => (
  authStore.hasPerm('doc:create')
  || authStore.hasPerm('doc:update')
  || authStore.hasPerm('doc:delete')
))
const message = useMessage()
const showDeleteConfirm = ref(false)
const pendingDelete = ref(null)
const deleting = ref(false)

function openDelete(kb) {
  if (!canDeleteKb.value) {
    message.warning('当前角色没有删除知识库的权限')
    return
  }
  pendingDelete.value = kb
  showDeleteConfirm.value = true
}

async function confirmDelete() {
  const kb = pendingDelete.value
  if (!kb) return
  if (!canDeleteKb.value) {
    message.warning('当前角色没有删除知识库的权限')
    return
  }
  deleting.value = true
  try {
    await kbStore.remove(kb.id)
    message.success('知识库已删除')
    showDeleteConfirm.value = false
    pendingDelete.value = null
  } catch (e) {
    // 后端校验：知识库下仍有文档时会拒绝删除，提示用户先清空文档
    message.warning(e?.response?.data?.detail || '删除失败，请重试')
  } finally {
    deleting.value = false
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
  if (!canCreateKb.value) {
    message.warning('创建知识库需要新增权限和全部知识库范围')
    return
  }
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
  if (!canUpdateKb.value) {
    message.warning('当前角色没有修改知识库的权限')
    return
  }
  editingId.value = kb.id
  editForm.value = { name: kb.name, description: kb.description || '', icon_color: kb.icon_color }
  showEdit.value = true
}

async function handleEdit() {
  if (!editForm.value.name.trim()) return
  if (!canUpdateKb.value) {
    message.warning('当前角色没有修改知识库的权限')
    return
  }
  saving.value = true
  try {
    await kbStore.update(editingId.value, { ...editForm.value })
    showEdit.value = false
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.kb-card {
  min-height: 164px;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.kb-card:hover {
  border-color: var(--ui-border-strong, #cbd5e1);
  box-shadow: var(--ui-shadow-float, 0 10px 24px rgb(15 23 42 / 0.1));
  transform: translateY(-2px);
}
</style>
