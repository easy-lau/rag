<template>
  <div class="p-6 h-full overflow-y-auto">
    <div class="flex items-center justify-end mb-6">
      <n-button type="primary" @click="openCreate">
        <template #icon><n-icon><AddOutline /></n-icon></template>
        新建角色
      </n-button>
    </div>

    <n-data-table
      :columns="columns" :data="roles" :loading="loading"
      :pagination="pagination"
      class="bg-white dark:bg-gray-800 rounded-xl"
    />

    <!-- Create / Edit modal -->
    <n-modal v-model:show="showModal" to="#app" preset="card"
      :title="editingId ? '编辑角色' : '新建角色'" style="width: 720px; max-width: 92vw">
      <n-form :model="form" label-placement="top">
        <div class="grid grid-cols-2 gap-4">
          <n-form-item label="角色名称" required>
            <n-input v-model:value="form.name" :disabled="editingIsSystem" placeholder="角色名称" />
          </n-form-item>
          <n-form-item label="描述">
            <n-input v-model:value="form.description" placeholder="可选描述" />
          </n-form-item>
        </div>

        <n-form-item label="权限">
          <div class="w-full space-y-4">
            <div v-for="group in permissionGroups" :key="group.title">
              <div class="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">{{ group.title }}</div>
              <n-checkbox-group v-model:value="form.permissions">
                <div class="grid grid-cols-2 sm:grid-cols-3 gap-y-1.5">
                  <n-checkbox v-for="key in group.keys" :key="key" :value="key" :label="permLabel(key)" />
                </div>
              </n-checkbox-group>
            </div>
          </div>
        </n-form-item>

        <n-form-item label="可访问知识库">
          <n-select
            v-model:value="form.kb_ids" :options="kbOptions" multiple
            placeholder="选择可访问的知识库（含 kb:access_all 权限可访问全部）" clearable
          />
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
import { NButton, NIcon, NDataTable, NModal, NForm, NFormItem, NInput, NSelect, NCheckbox, NCheckboxGroup, NTag, NPopconfirm, useMessage } from 'naive-ui'
import { AddOutline } from '@vicons/ionicons5'
import { getRoles, createRole, updateRole, deleteRole, getPermissionCatalog } from '@/api/roles'
import { getKnowledgeBases } from '@/api/knowledge'

const msg = useMessage()
const roles = ref([])
const allPermissions = ref([])
const menuMeta = ref([]) // [{key,route,title,permission}]
const kbs = ref([])
const loading = ref(false)
const saving = ref(false)

const showModal = ref(false)
const editingId = ref(null)
const editingIsSystem = ref(false)
const form = ref({ name: '', description: '', permissions: [], kb_ids: [] })

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
watch(() => roles.value.length, () => {
  const max = Math.max(1, Math.ceil(roles.value.length / pagination.pageSize))
  if (pagination.page > max) pagination.page = max
})

const kbOptions = computed(() => kbs.value.map(kb => ({ label: kb.name, value: kb.id })))

// 已知权限的中文标签；其余回退到 key 本身
const PERM_LABELS = {
  'menu:chat': '菜单·问答对话', 'menu:knowledge': '菜单·知识库', 'menu:documents': '菜单·文档',
  'menu:search_test': '菜单·检索测试', 'menu:settings': '菜单·系统设置',
  'menu:users': '菜单·用户管理', 'menu:roles': '菜单·角色管理',
  'chat:use': '使用问答', 'search:use': '使用检索',
  'kb:read': '知识库·读', 'kb:write': '知识库·写', 'kb:access_all': '访问全部知识库',
  'doc:read': '文档·读', 'doc:write': '文档·写',
  'settings:read': '设置·读', 'settings:write': '设置·写',
  'user:manage': '用户管理', 'role:manage': '角色管理',
  'log:read': '查看登录日志',
}
function permLabel(key) {
  if (PERM_LABELS[key]) return PERM_LABELS[key]
  // 用菜单元数据补充标题
  const m = menuMeta.value.find(x => x.permission === key)
  return m ? `菜单·${m.title}` : key
}

// 按 menu: / 操作 分组
const permissionGroups = computed(() => {
  const all = allPermissions.value
  const menuKeys = all.filter(k => k.startsWith('menu:'))
  const opKeys = all.filter(k => !k.startsWith('menu:'))
  return [
    { title: '菜单权限', keys: menuKeys },
    { title: '操作权限', keys: opKeys },
  ]
})

const columns = [
  { title: '名称', key: 'name', ellipsis: { tooltip: true } },
  { title: '描述', key: 'description', render: r => r.description || '—' },
  { title: '权限数', key: 'perm_count', width: 90, render: r => (r.permissions?.length || 0) },
  { title: '知识库数', key: 'kb_count', width: 90, render: r => (r.kb_ids?.length || 0) },
  {
    title: '类型', key: 'is_system', width: 90,
    render: r => r.is_system ? h(NTag, { type: 'info', size: 'small' }, () => '内建') : h(NTag, { size: 'small' }, () => '自定义')
  },
  {
    title: '操作', key: 'actions', width: 160,
    render: row => h('div', { style: 'display:flex;gap:8px;align-items:center' }, [
      h(NButton, { text: true, type: 'primary', size: 'small', onClick: () => openEdit(row) }, () => '编辑'),
      h(NPopconfirm, { onPositiveClick: () => handleDelete(row) }, {
        trigger: () => h(NButton, { text: true, type: 'error', size: 'small', disabled: row.is_system }, () => '删除'),
        default: () => '确定删除该角色？',
      }),
    ])
  }
]

onMounted(async () => {
  await Promise.all([loadRoles(), loadCatalog(), loadKbs()])
})

async function loadRoles() {
  loading.value = true
  try { roles.value = await getRoles() }
  finally { loading.value = false }
}

async function loadCatalog() {
  const catalog = await getPermissionCatalog()
  allPermissions.value = catalog.permissions || []
  menuMeta.value = catalog.menus || []
}

async function loadKbs() {
  kbs.value = await getKnowledgeBases()
}

function openCreate() {
  editingId.value = null
  editingIsSystem.value = false
  form.value = { name: '', description: '', permissions: [], kb_ids: [] }
  showModal.value = true
}

function openEdit(row) {
  editingId.value = row.id
  editingIsSystem.value = row.is_system
  form.value = {
    name: row.name,
    description: row.description || '',
    permissions: [...(row.permissions || [])],
    kb_ids: [...(row.kb_ids || [])],
  }
  showModal.value = true
}

async function handleSave() {
  if (!form.value.name.trim()) { msg.warning('请输入角色名称'); return }
  saving.value = true
  try {
    const payload = {
      name: form.value.name.trim(),
      description: form.value.description || null,
      permissions: form.value.permissions,
      kb_ids: form.value.kb_ids,
    }
    if (editingId.value) {
      await updateRole(editingId.value, payload)
      msg.success('角色已更新')
    } else {
      await createRole(payload)
      msg.success('角色已创建')
    }
    showModal.value = false
    await loadRoles()
  } catch (e) {
    msg.error(e?.response?.data?.detail || '保存失败，请重试')
  } finally {
    saving.value = false
  }
}

async function handleDelete(row) {
  try {
    await deleteRole(row.id)
    roles.value = roles.value.filter(r => r.id !== row.id)
    msg.success('角色已删除')
  } catch (e) {
    msg.error(e?.response?.data?.detail || '删除失败，请重试')
  }
}
</script>
