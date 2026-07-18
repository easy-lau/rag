<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="flex items-center justify-end mb-6">
      <n-button type="primary" @click="openCreate">
        <template #icon><n-icon><AddOutline /></n-icon></template>
        新建用户
      </n-button>
    </div>

    <n-data-table
      :columns="columns" :data="users" :loading="loading"
      :pagination="pagination" :scroll-x="ui.isMobile ? 900 : undefined"
      class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden"
    />

    <!-- Create / Edit modal -->
    <n-modal v-model:show="showModal" to="#app" preset="card" :title="editingId ? '编辑用户' : '新建用户'" style="width: 90vw; max-width: 384px">
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
          <n-select v-model:value="form.role_id" :options="roleOptions" placeholder="选择角色" clearable />
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
          <n-button @click="showModal = false">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleSave">保存</n-button>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch, h } from 'vue'
import { NButton, NIcon, NDataTable, NModal, NForm, NFormItem, NInput, NSelect, NSwitch, NTag, NPopconfirm, useMessage } from 'naive-ui'
import { AddOutline } from '@vicons/ionicons5'
import { getUsers, createUser, updateUser, deleteUser } from '@/api/users'
import { getRoles } from '@/api/roles'
import { useUiStore } from '@/stores/ui'

const ui = useUiStore()
const msg = useMessage()
const users = ref([])
const roles = ref([])
const loading = ref(false)
const saving = ref(false)

const showModal = ref(false)
const editingId = ref(null)
const form = ref({ username: '', password: '', display_name: '', role_id: null, is_active: true })

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

const roleOptions = computed(() => roles.value.map(r => ({ label: r.name, value: r.id })))
const roleName = (id) => roles.value.find(r => r.id === id)?.name || '—'

const columns = [
  { title: '用户名', key: 'username', ellipsis: { tooltip: true } },
  { title: '显示名', key: 'display_name', render: r => r.display_name || '—' },
  { title: '角色', key: 'role', render: r => r.role_name || roleName(r.role_id) },
  {
    title: '状态', key: 'is_active', width: 100,
    render: r => h(NTag, { type: r.is_active ? 'success' : 'default', size: 'small' }, () => r.is_active ? '启用' : '禁用')
  },
  {
    title: '操作', key: 'actions', width: 220,
    render: row => h('div', { style: 'display:flex;gap:8px;align-items:center;flex-wrap:wrap' }, [
      h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openEdit(row) }, () => '编辑'),
      h(NButton, { text: true, size: 'small', disabled: row.is_superadmin, onClick: () => toggleActive(row) },
        () => row.is_active ? '禁用' : '启用'),
      h(NPopconfirm, { onPositiveClick: () => handleDelete(row) }, {
        trigger: () => h(NButton, { text: true, type: 'error', size: 'small', disabled: row.is_superadmin }, () => '删除'),
        default: () => '确定删除该用户？',
      }),
    ])
  }
]
columns.forEach(c => { c.titleAlign = 'center'; c.align = 'center' })  // 表头 + 内容统一居中

onMounted(async () => {
  await Promise.all([loadUsers(), loadRoles()])
})

async function loadUsers() {
  loading.value = true
  try { users.value = await getUsers() }
  finally { loading.value = false }
}

async function loadRoles() {
  roles.value = await getRoles()
}

function openCreate() {
  editingId.value = null
  form.value = { username: '', password: '', display_name: '', role_id: null, is_active: true }
  showModal.value = true
}

function openEdit(row) {
  editingId.value = row.id
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

async function handleDelete(row) {
  try {
    await deleteUser(row.id)
    users.value = users.value.filter(u => u.id !== row.id)
    msg.success('用户已删除')
  } catch (e) {
    msg.error(e?.response?.data?.detail || '删除失败，请重试')
  }
}
</script>
