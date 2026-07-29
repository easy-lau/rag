<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-5">
      <PageHeader title="用户管理" description="维护账号、角色归属与启用状态；高风险删除需二次确认。">
        <template #actions>
          <n-button type="primary" @click="openCreate">
            <template #icon><n-icon><AddOutline /></n-icon></template>
            新建用户
          </n-button>
        </template>
      </PageHeader>

      <SurfaceCard padding="none" class="overflow-hidden">
        <n-data-table
          :columns="columns" :data="users" :loading="loading"
          :pagination="pagination" :scroll-x="ui.isCompact ? 760 : undefined"
          class="admin-data-table"
        />
      </SurfaceCard>
    </div>

    <!-- Create / Edit modal -->
    <AppModal v-model:show="showModal" :title="editingId ? '编辑用户' : '新建用户'" width="min(92vw, 384px)" :loading="saving">
      <n-form :model="form" label-placement="top">
        <n-form-item label="用户名" required>
          <n-input v-model:value="form.username" :disabled="!!editingId" placeholder="登录用户名" />
        </n-form-item>
        <n-form-item :label="editingId ? '重置密码（留空则不修改）' : '初始密码'" :required="!editingId">
          <n-input v-model:value="form.password" type="password" show-password-on="click"
            :placeholder="editingId ? '留空保持原密码' : '请输入初始密码'" />
        </n-form-item>
        <n-form-item label="显示名">
          <n-input v-model:value="form.display_name" placeholder="可选" />
        </n-form-item>
        <n-form-item label="角色">
          <div class="w-full">
            <n-select
              v-model:value="form.role_id"
              :options="roleOptions"
              :disabled="editingIsSuperadmin"
              placeholder="选择角色"
              clearable
            />
            <p v-if="editingIsSuperadmin" class="mt-1.5 text-xs text-gray-500 dark:text-gray-400">
              超级管理员的系统角色不可在此调整。
            </p>
          </div>
        </n-form-item>
        <n-form-item label="状态">
          <n-switch v-model:value="form.is_active">
            <template #checked>启用</template>
            <template #unchecked>禁用</template>
          </n-switch>
        </n-form-item>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="saving" @click="showModal = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleSave">保存</n-button>
        </div>
      </template>
    </AppModal>

    <DangerConfirm
      v-model:show="showDeleteConfirm"
      title="删除用户？"
      :subject="pendingDelete?.username || ''"
      description="删除后，该账号无法恢复，也无法继续登录系统。"
      :loading="deleting"
      @confirm="confirmDelete"
      @cancel="pendingDelete = null"
    />
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch, h } from 'vue'
import { NButton, NIcon, NDataTable, NForm, NFormItem, NInput, NSelect, NSwitch, NTag, useMessage } from 'naive-ui'
import { AddOutline } from '@vicons/ionicons5'
import { getUsers, createUser, updateUser, deleteUser } from '@/api/users'
import { getAssignableRoles } from '@/api/roles'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import RowActions from '@/components/ui/RowActions.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'

const ui = useUiStore()
const authStore = useAuthStore()
const msg = useMessage()
const users = ref([])
const roles = ref([])
const loading = ref(false)
const saving = ref(false)

const showModal = ref(false)
const editingId = ref(null)
const editingIsSuperadmin = ref(false)
const form = ref({ username: '', password: '', display_name: '', role_id: null, is_active: true })
const showDeleteConfirm = ref(false)
const pendingDelete = ref(null)
const deleting = ref(false)

const pagination = reactive({
  page: 1,
  pageSize: 10,                      // 默认每页 10 条
  showSizePicker: true,
  pageSizes: [10, 20, 30, 50],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (p) => { pagination.page = p },
  onUpdatePageSize: (ps) => { pagination.pageSize = ps; pagination.page = 1 },
})
// 删除后若当前页超出范围则回退到最后一页
watch(() => users.value.length, () => {
  const max = Math.max(1, Math.ceil(users.value.length / pagination.pageSize))
  if (pagination.page > max) pagination.page = max
})

const roleOptions = computed(() => roles.value.map(r => ({
  label: r.name,
  value: r.id,
  disabled: r.is_assignable === false,
})))
const roleName = (id) => roles.value.find(r => r.id === id)?.name || '—'

const columns = [
  { title: '用户名', key: 'username', minWidth: 140, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  { title: '显示名', key: 'display_name', minWidth: 130, align: 'left', titleAlign: 'left', render: r => r.display_name || '—' },
  { title: '角色', key: 'role', minWidth: 130, align: 'left', titleAlign: 'left', render: r => r.role_name || roleName(r.role_id) },
  {
    title: '状态', key: 'is_active', width: 96, align: 'center', titleAlign: 'center',
    render: r => h(NTag, { type: r.is_active ? 'success' : 'default', size: 'small' }, () => r.is_active ? '启用' : '禁用')
  },
  {
    title: '操作', key: 'actions', width: 208, align: 'center', titleAlign: 'center',
    render: row => h(RowActions, { label: `用户 ${row.username} 操作` }, {
      default: () => [
        h(NButton, {
          text: true,
          type: 'primary',
          size: 'small',
          disabled: row.is_superadmin && !authStore.user?.is_superadmin,
          onClick: () => openEdit(row),
        }, () => '编辑'),
        h(NButton, { text: true, size: 'small', disabled: row.is_superadmin, onClick: () => toggleActive(row) },
          () => row.is_active ? '禁用' : '启用'),
        h(NButton, { text: true, type: 'error', size: 'small', disabled: row.is_superadmin, onClick: () => openDelete(row) }, () => '删除'),
      ],
    })
  }
]

onMounted(async () => {
  await Promise.all([loadUsers(), loadRoles()])
})

async function loadUsers() {
  loading.value = true
  try { users.value = await getUsers() }
  finally { loading.value = false }
}

async function loadRoles() {
  roles.value = await getAssignableRoles()
}

function openCreate() {
  editingId.value = null
  editingIsSuperadmin.value = false
  form.value = { username: '', password: '', display_name: '', role_id: null, is_active: true }
  showModal.value = true
}

function openEdit(row) {
  editingId.value = row.id
  editingIsSuperadmin.value = Boolean(row.is_superadmin)
  // 不可分配的超级管理员角色不会出现在 /roles/assignable，编辑账号时补一个禁用展示项，
  // 避免 Select 回退显示难以理解的 UUID。
  if (row.is_superadmin && row.role_id && !roles.value.some(role => role.id === row.role_id)) {
    roles.value = [...roles.value, {
      id: row.role_id,
      name: row.role_name || '超级管理员',
      is_assignable: false,
    }]
  }
  form.value = {
    username: row.username,
    password: '',
    display_name: row.display_name || '',
    role_id: row.role_id || null,
    is_active: row.is_active,
  }
  showModal.value = true
}

async function handleSave() {
  if (!form.value.username.trim()) { msg.warning('请输入用户名'); return }
  if (!editingId.value && !form.value.password.trim()) { msg.warning('请输入初始密码'); return }
  saving.value = true
  try {
    if (editingId.value) {
      const payload = {
        display_name: form.value.display_name || null,
        role_id: form.value.role_id || null,
        is_active: form.value.is_active,
      }
      if (form.value.password.trim()) payload.password = form.value.password.trim()
      await updateUser(editingId.value, payload)
      msg.success('用户已更新')
    } else {
      await createUser({
        username: form.value.username.trim(),
        password: form.value.password.trim(),
        display_name: form.value.display_name || null,
        role_id: form.value.role_id || null,
        is_active: form.value.is_active,
      })
      msg.success('用户已创建')
    }
    showModal.value = false
    await loadUsers()
  } catch (e) {
    msg.error(e?.response?.data?.detail || '保存失败，请重试')
  } finally {
    saving.value = false
  }
}

async function toggleActive(row) {
  try {
    await updateUser(row.id, { is_active: !row.is_active })
    await loadUsers()
    msg.success(!row.is_active ? '用户已启用' : '用户已禁用')
  } catch (e) {
    msg.error(e?.response?.data?.detail || '操作失败，请重试')
  }
}

function openDelete(row) {
  pendingDelete.value = row
  showDeleteConfirm.value = true
}

async function confirmDelete() {
  const row = pendingDelete.value
  if (!row) return
  deleting.value = true
  try {
    await deleteUser(row.id)
    users.value = users.value.filter(u => u.id !== row.id)
    msg.success('用户已删除')
    showDeleteConfirm.value = false
    pendingDelete.value = null
  } catch (e) {
    msg.error(e?.response?.data?.detail || '删除失败，请重试')
  } finally {
    deleting.value = false
  }
}
</script>
