<template>
  <div class="terminology-page">
    <div class="terminology-page__inner">
      <PageHeader
        title="受控术语"
        description="维护已审核的概念、同义术语和适用范围。术语变更会影响检索与证据语义，因此所有记录都保留在所选知识库的版本化注册表中。"
      >
        <template #meta>
          <n-tag :type="canManage ? 'warning' : 'default'" :bordered="false" round>
            {{ canManage ? '可维护' : '仅查看' }}
          </n-tag>
        </template>
        <template #actions>
          <n-button
            v-if="canRead"
            :loading="registryLoading"
            :disabled="!selectedKbId"
            @click="refreshSelectedRegistry"
          >
            <template #icon><n-icon><RefreshOutline /></n-icon></template>
            刷新
          </n-button>
          <n-button
            v-if="canManage"
            type="primary"
            :disabled="!selectedKbId || registryLoading"
            @click="openCreateConcept"
          >
            <template #icon><n-icon><AddOutline /></n-icon></template>
            新增概念
          </n-button>
        </template>
      </PageHeader>

      <SurfaceCard v-if="!canRead" class="terminology-page__empty">
        <n-icon :size="30" class="text-gray-400"><LockClosedOutline /></n-icon>
        <div class="mt-3 text-sm font-medium text-[var(--ui-text)]">暂无查看受控术语的权限</div>
        <p class="mt-1 text-xs text-[var(--ui-text-tertiary)]">
          请联系管理员为当前角色授予 <code>terminology:read</code> 以及相应的知识库范围。
        </p>
      </SurfaceCard>

      <template v-else>
        <SurfaceCard class="terminology-page__selector-card">
          <div class="terminology-page__selector-copy">
            <label for="terminology-kb-select" class="terminology-page__selector-label">知识库</label>
            <p>选择已授权知识库后，读取该知识库当前版本的术语注册表。</p>
          </div>
          <div class="terminology-page__selector-control">
            <n-select
              id="terminology-kb-select"
              v-model:value="selectedKbId"
              :options="knowledgeBaseOptions"
              :loading="knowledgeBasesLoading"
              :disabled="knowledgeBasesLoading || !knowledgeBaseOptions.length"
              filterable
              placeholder="选择可访问知识库"
              aria-label="选择受控术语知识库"
            />
            <div v-if="registry" class="terminology-page__revision" aria-live="polite">
              注册表版本 r{{ registry.registry_revision }}
            </div>
          </div>
        </SurfaceCard>

        <n-alert
          v-if="!canManage"
          type="info"
          :show-icon="false"
          class="terminology-page__permission-alert"
        >
          当前仅查看。创建、编辑和停用术语的入口已禁用；如需维护，请申请 <code>terminology:manage</code> 权限。
        </n-alert>

        <n-alert
          v-else-if="!canReadDocuments"
          type="warning"
          :show-icon="false"
          class="terminology-page__permission-alert"
        >
          当前角色没有 <code>doc:read</code> 权限。为避免误把术语绑定到不可见文档，页面仅允许维护知识库及业务范围，不提供文档范围选择；后端仍会进行对象级校验。
        </n-alert>

        <n-spin :show="registryLoading">
          <SurfaceCard
            v-if="!knowledgeBasesLoading && !knowledgeBaseOptions.length"
            class="terminology-page__empty"
          >
            <div class="text-sm font-medium text-[var(--ui-text)]">当前没有可访问的知识库</div>
            <p class="mt-1 text-xs text-[var(--ui-text-tertiary)]">术语注册表必须归属于一个已授权知识库。</p>
          </SurfaceCard>

          <SurfaceCard v-else-if="selectedKbId && registryLoadError" class="terminology-page__empty">
            <div class="text-sm font-medium text-[var(--ui-text)]">术语注册表暂时无法读取</div>
            <p class="mt-1 text-xs text-[var(--ui-text-tertiary)]">{{ registryLoadError }}</p>
            <n-button class="mt-4" :loading="registryLoading" @click="refreshSelectedRegistry">重新读取</n-button>
          </SurfaceCard>

          <template v-else-if="selectedKbId && registry">
            <SurfaceCard padding="none" class="terminology-page__table-card">
              <div class="terminology-page__section-head">
                <div>
                  <h2>概念与术语</h2>
                  <p>
                    概念编码是稳定标识；同义术语按“严格等价”或“仅用于召回”分别管理。
                    {{ canManage ? '点击术语可编辑。' : '' }} 停用是可审计的状态变更，不提供删除操作。
                  </p>
                </div>
                <n-button
                  v-if="canManage"
                  secondary
                  :disabled="registryLoading"
                  @click="openCreateConcept"
                >
                  <template #icon><n-icon><AddOutline /></n-icon></template>
                  新增概念
                </n-button>
              </div>
              <n-data-table
                :columns="conceptColumns"
                :data="concepts"
                :loading="registryLoading"
                :scroll-x="ui.isCompact ? 980 : undefined"
                :bordered="false"
                class="terminology-page__table"
              >
                <template #empty>
                  <n-empty description="当前知识库尚未建立受控术语概念">
                    <template v-if="canManage" #extra>
                      <n-button type="primary" @click="openCreateConcept">新增第一个概念</n-button>
                    </template>
                  </n-empty>
                </template>
              </n-data-table>
            </SurfaceCard>

            <SurfaceCard padding="none" class="terminology-page__table-card">
              <div class="terminology-page__section-head">
                <div>
                  <h2>作用域绑定</h2>
                  <p>一个概念可按知识库、文档、产品、版本或项目范围生效。留空范围表示整个知识库，不要求输入任何对象 ID。</p>
                </div>
                <n-button
                  v-if="canManage"
                  secondary
                  :disabled="registryLoading || !concepts.length"
                  @click="openCreateBinding"
                >
                  <template #icon><n-icon><AddOutline /></n-icon></template>
                  新增绑定
                </n-button>
              </div>
              <n-data-table
                :columns="bindingColumns"
                :data="bindings"
                :loading="registryLoading"
                :scroll-x="ui.isCompact ? 1040 : undefined"
                :bordered="false"
                class="terminology-page__table"
              >
                <template #empty>
                  <n-empty description="尚未建立作用域绑定" />
                </template>
              </n-data-table>
            </SurfaceCard>
          </template>
        </n-spin>
      </template>
    </div>
  </div>

  <AppModal
    v-model:show="showConceptModal"
    :title="conceptDialogTitle"
    width="min(92vw, 620px)"
    :loading="savingConcept"
    @close="resetConceptDialog"
  >
    <form @submit.prevent="saveConcept">
      <n-form ref="conceptFormRef" :model="conceptForm" :rules="conceptRules" label-placement="top">
        <div class="terminology-page__form-grid">
          <n-form-item label="概念编码" path="code" required>
            <n-input
              v-model:value="conceptForm.code"
              :disabled="conceptMode === 'edit' || savingConcept"
              maxlength="64"
              show-count
              placeholder="例如：travel_meal_allowance"
              aria-label="概念编码"
            />
            <template #feedback>
              {{ conceptMode === 'edit' ? '概念编码是稳定注册表标识，建立后不可修改。' : '建议使用稳定、可读的英文或拼音编码；创建后不可修改。' }}
            </template>
          </n-form-item>
          <n-form-item label="规范术语" path="canonical_term" required>
            <n-input
              v-model:value="conceptForm.canonical_term"
              :disabled="savingConcept"
              maxlength="120"
              show-count
              placeholder="例如：餐饮补贴"
              aria-label="规范术语"
            />
          </n-form-item>
        </div>
        <n-form-item label="说明" path="description">
          <n-input
            v-model:value="conceptForm.description"
            type="textarea"
            :rows="3"
            :disabled="savingConcept"
            maxlength="2000"
            show-count
            placeholder="说明该概念的业务含义、使用边界或审核依据（可选）"
            aria-label="概念说明"
          />
        </n-form-item>

        <n-form-item v-if="conceptMode === 'edit'" label="状态">
          <n-switch v-model:value="conceptForm.is_active" :disabled="savingConcept">
            <template #checked>启用</template>
            <template #unchecked>停用</template>
          </n-switch>
          <template #feedback>停用会保留审计记录和历史注册表版本，不会删除数据。</template>
        </n-form-item>

        <template v-else>
          <div class="terminology-page__form-section">
            <div>
              <h3>初始适用范围</h3>
              <p>概念创建时会同时建立一条绑定。所有范围留空表示该概念适用于整个知识库。</p>
            </div>
            <n-form-item label="限定到文档（可选）">
              <n-select
                v-model:value="conceptForm.initial_document_id"
                :options="documentScopeOptions"
                :loading="documentsLoading"
                :disabled="savingConcept || !canReadDocuments"
                filterable
                aria-label="选择概念初始适用文档"
              />
              <template #feedback>
                {{ canReadDocuments
                  ? (documentsLoadError || '不选择文档时，初始范围不限定到单个文档。')
                  : '当前没有文档查看权限，不能创建文档范围绑定。' }}
              </template>
            </n-form-item>
            <div class="terminology-page__form-grid">
              <n-form-item label="产品范围（可选）">
                <n-input v-model:value="conceptForm.initial_scope_product_key" :disabled="savingConcept" maxlength="160" placeholder="例如：企业差旅" />
              </n-form-item>
              <n-form-item label="版本范围（可选）">
                <n-input v-model:value="conceptForm.initial_scope_version_key" :disabled="savingConcept" maxlength="160" placeholder="例如：2026" />
              </n-form-item>
              <n-form-item label="项目范围（可选）">
                <n-input v-model:value="conceptForm.initial_scope_project_key" :disabled="savingConcept" maxlength="160" placeholder="例如：总部制度" />
              </n-form-item>
            </div>
          </div>
        </template>
      </n-form>
    </form>
    <template #footer>
      <n-button :disabled="savingConcept" @click="showConceptModal = false">取消</n-button>
      <n-button type="primary" :loading="savingConcept" @click="saveConcept">
        {{ conceptMode === 'create' ? '创建概念' : '保存变更' }}
      </n-button>
    </template>
  </AppModal>

  <AppModal
    v-model:show="showTermModal"
    :title="termDialogTitle"
    width="min(92vw, 520px)"
    :loading="savingTerm"
    @close="resetTermDialog"
  >
    <form @submit.prevent="saveTerm">
      <n-form ref="termFormRef" :model="termForm" :rules="termRules" label-placement="top">
        <n-alert type="info" :show-icon="false" class="mb-4">
          所属概念：<strong>{{ termConcept?.canonical_term || '—' }}</strong>
        </n-alert>
        <n-form-item label="术语形式" path="term" required>
          <n-input
            v-model:value="termForm.term"
            :disabled="savingTerm"
            maxlength="120"
            show-count
            placeholder="例如：餐补"
            aria-label="术语形式"
          />
        </n-form-item>
        <n-form-item label="匹配方式" path="match_mode" required>
          <n-select
            v-model:value="termForm.match_mode"
            :options="termMatchModeOptions"
            :disabled="savingTerm"
            aria-label="术语匹配方式"
          />
          <template #feedback>“严格等价”可参与证据语义；“仅用于召回”不能被提升为严格同义证据。</template>
        </n-form-item>
        <n-form-item label="状态">
          <n-switch v-model:value="termForm.is_active" :disabled="savingTerm">
            <template #checked>启用</template>
            <template #unchecked>停用</template>
          </n-switch>
          <template #feedback>停用不会删除历史记录；规范术语的完整性由后端统一校验。</template>
        </n-form-item>
      </n-form>
    </form>
    <template #footer>
      <n-button :disabled="savingTerm" @click="showTermModal = false">取消</n-button>
      <n-button type="primary" :loading="savingTerm" @click="saveTerm">
        {{ termMode === 'create' ? '新增术语' : '保存变更' }}
      </n-button>
    </template>
  </AppModal>

  <AppModal
    v-model:show="showBindingModal"
    :title="bindingDialogTitle"
    width="min(92vw, 660px)"
    :loading="savingBinding"
    @close="resetBindingDialog"
  >
    <form @submit.prevent="saveBinding">
      <n-form ref="bindingFormRef" :model="bindingForm" :rules="bindingRules" label-placement="top">
        <n-form-item label="所属概念" path="concept_id" required>
          <n-select
            v-model:value="bindingForm.concept_id"
            :options="conceptOptions"
            :disabled="bindingMode === 'edit' || savingBinding"
            filterable
            placeholder="选择已有概念"
            aria-label="选择作用域绑定概念"
          />
          <template #feedback>
            {{ bindingMode === 'edit' ? '绑定建立后不能改到其他概念；如需新关系，请新增绑定。' : '仅可从当前知识库已建立的概念中选择。' }}
          </template>
        </n-form-item>

        <n-form-item label="限定到文档（可选）">
          <n-select
            v-model:value="bindingForm.document_id"
            :options="documentScopeOptions"
            :loading="documentsLoading"
            :disabled="savingBinding || !canReadDocuments"
            filterable
            aria-label="选择作用域绑定文档"
          />
          <template #feedback>
            {{ canReadDocuments
              ? (documentsLoadError || '不选择文档时，不按单个文档限定。')
              : '当前没有文档查看权限，不能创建或编辑文档范围绑定。' }}
          </template>
        </n-form-item>

        <div class="terminology-page__form-grid">
          <n-form-item label="产品范围（可选）">
            <n-input v-model:value="bindingForm.scope_product_key" :disabled="savingBinding" maxlength="160" placeholder="例如：企业差旅" />
          </n-form-item>
          <n-form-item label="版本范围（可选）">
            <n-input v-model:value="bindingForm.scope_version_key" :disabled="savingBinding" maxlength="160" placeholder="例如：2026" />
          </n-form-item>
          <n-form-item label="项目范围（可选）">
            <n-input v-model:value="bindingForm.scope_project_key" :disabled="savingBinding" maxlength="160" placeholder="例如：总部制度" />
          </n-form-item>
        </div>
        <n-form-item label="状态">
          <n-switch v-model:value="bindingForm.is_active" :disabled="savingBinding">
            <template #checked>启用</template>
            <template #unchecked>停用</template>
          </n-switch>
          <template #feedback>停用会让该适用范围失效，但不会删除可审计记录。</template>
        </n-form-item>
      </n-form>
    </form>
    <template #footer>
      <n-button :disabled="savingBinding" @click="showBindingModal = false">取消</n-button>
      <n-button type="primary" :loading="savingBinding" @click="saveBinding">
        {{ bindingMode === 'create' ? '创建绑定' : '保存变更' }}
      </n-button>
    </template>
  </AppModal>
</template>

<script setup>
import { computed, h, onMounted, ref, watch } from 'vue'
import {
  NAlert,
  NButton,
  NDataTable,
  NEmpty,
  NForm,
  NFormItem,
  NIcon,
  NInput,
  NSelect,
  NSpin,
  NSwitch,
  NTag,
  useMessage,
} from 'naive-ui'
import { AddOutline, LockClosedOutline, PencilOutline, RefreshOutline } from '@vicons/ionicons5'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import AppModal from '@/components/ui/AppModal.vue'
import { getAllDocuments } from '@/api/document'
import { getKnowledgeBases } from '@/api/knowledge'
import {
  createTerminologyBinding,
  createTerminologyConcept,
  createTerminologyTerm,
  getTerminologyRegistry,
  updateTerminologyBinding,
  updateTerminologyConcept,
  updateTerminologyTerm,
} from '@/api/terminology'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'

const KB_WIDE_SCOPE = '__knowledge_base_wide__'

const message = useMessage()
const authStore = useAuthStore()
const ui = useUiStore()

const canRead = computed(() => authStore.hasPerm('terminology:read'))
const canManage = computed(() => authStore.hasPerm('terminology:manage'))
const canReadDocuments = computed(() => authStore.hasPerm('doc:read'))

const knowledgeBases = ref([])
const knowledgeBasesLoading = ref(false)
const selectedKbId = ref(null)
const registry = ref(null)
const registryLoading = ref(false)
const registryLoadError = ref('')
const documents = ref([])
const documentsLoading = ref(false)
const documentsLoadError = ref('')
let selectionRevision = 0

const showConceptModal = ref(false)
const conceptMode = ref('create')
const conceptFormRef = ref(null)
const savingConcept = ref(false)
const conceptForm = ref(emptyConceptForm())
const editingConcept = ref(null)

const showTermModal = ref(false)
const termMode = ref('create')
const termFormRef = ref(null)
const savingTerm = ref(false)
const termForm = ref(emptyTermForm())
const termConcept = ref(null)
const editingTerm = ref(null)

const showBindingModal = ref(false)
const bindingMode = ref('create')
const bindingFormRef = ref(null)
const savingBinding = ref(false)
const bindingForm = ref(emptyBindingForm())
const editingBinding = ref(null)

const knowledgeBaseOptions = computed(() => knowledgeBases.value.map(kb => ({
  label: kb.name,
  value: kb.id,
})))
const concepts = computed(() => registry.value?.concepts || [])
const bindings = computed(() => registry.value?.bindings || [])
const conceptOptions = computed(() => concepts.value.map(concept => ({
  label: `${concept.canonical_term}${concept.is_active ? '' : '（已停用）'} · ${concept.code}`,
  value: concept.id,
})))
const documentScopeOptions = computed(() => [
  { label: '不限定到单个文档', value: KB_WIDE_SCOPE },
  ...documents.value.map(document => ({ label: document.filename, value: document.id })),
])
const documentNameById = computed(() => new Map(
  documents.value.map(document => [document.id, document.filename]),
))
const conceptById = computed(() => new Map(concepts.value.map(concept => [concept.id, concept])))

const conceptDialogTitle = computed(() => conceptMode.value === 'create' ? '新增受控术语概念' : '编辑受控术语概念')
const termDialogTitle = computed(() => termMode.value === 'create' ? '新增同义术语' : '编辑同义术语')
const bindingDialogTitle = computed(() => bindingMode.value === 'create' ? '新增作用域绑定' : '编辑作用域绑定')

const termMatchModeOptions = [
  { label: '严格等价（可作为证据同义）', value: 'strict_equivalent' },
  { label: '仅用于召回（不可提升为严格证据）', value: 'retrieval_only' },
]

const conceptRules = {
  code: { required: true, message: '请输入概念编码', trigger: ['blur', 'input'] },
  canonical_term: { required: true, message: '请输入规范术语', trigger: ['blur', 'input'] },
}
const termRules = {
  term: { required: true, message: '请输入术语形式', trigger: ['blur', 'input'] },
  match_mode: { required: true, message: '请选择匹配方式', trigger: ['blur', 'change'] },
}
const bindingRules = {
  concept_id: { required: true, message: '请选择所属概念', trigger: ['blur', 'change'] },
}

const conceptColumns = computed(() => {
  const columns = [
    { title: '概念编码', key: 'code', width: 180, ellipsis: { tooltip: true } },
    { title: '规范术语', key: 'canonical_term', minWidth: 140, ellipsis: { tooltip: true } },
    {
      title: '同义术语',
      key: 'terms',
      minWidth: 280,
      render: row => renderTerms(row),
    },
    {
      title: '状态',
      key: 'is_active',
      width: 100,
      align: 'center',
      render: row => renderStatusTag(row.is_active),
    },
  ]
  if (canManage.value) {
    columns.push({
      title: '操作',
      key: 'actions',
      width: 172,
      fixed: ui.isCompact ? undefined : 'right',
      align: 'center',
      render: row => h('div', { class: 'terminology-page__row-actions' }, [
        actionButton('编辑概念', () => openEditConcept(row)),
        actionButton('术语', () => openCreateTerm(row)),
      ]),
    })
  }
  return columns
})

const bindingColumns = computed(() => {
  const columns = [
    {
      title: '概念',
      key: 'concept',
      minWidth: 180,
      render: row => {
        const concept = row.concept || conceptById.value.get(row.concept_id)
        return h('div', { class: 'terminology-page__concept-cell' }, [
          h('strong', concept?.canonical_term || '未知概念'),
          h('span', concept?.code || '编码未记录'),
        ])
      },
    },
    {
      title: '适用范围',
      key: 'scope',
      minWidth: 360,
      render: row => h('span', { class: 'terminology-page__scope-cell' }, formatScope(row)),
    },
    {
      title: '状态',
      key: 'is_active',
      width: 100,
      align: 'center',
      render: row => renderStatusTag(row.is_active),
    },
  ]
  if (canManage.value) {
    columns.push({
      title: '操作',
      key: 'actions',
      width: 108,
      fixed: ui.isCompact ? undefined : 'right',
      align: 'center',
      render: row => actionButton('编辑', () => openEditBinding(row)),
    })
  }
  return columns
})

watch(selectedKbId, kbId => {
  selectionRevision += 1
  registry.value = null
  registryLoadError.value = ''
  documents.value = []
  documentsLoadError.value = ''
  if (!kbId) return
  const requestRevision = selectionRevision
  void loadRegistry(kbId, requestRevision)
  if (canReadDocuments.value) void loadDocuments(kbId, requestRevision)
})

onMounted(() => {
  if (canRead.value) void loadKnowledgeBases()
})

function emptyConceptForm() {
  return {
    code: '',
    canonical_term: '',
    description: '',
    is_active: true,
    initial_document_id: KB_WIDE_SCOPE,
    initial_scope_product_key: '',
    initial_scope_version_key: '',
    initial_scope_project_key: '',
  }
}

function emptyTermForm() {
  return { term: '', match_mode: 'strict_equivalent', is_active: true }
}

function emptyBindingForm() {
  return {
    concept_id: null,
    document_id: KB_WIDE_SCOPE,
    scope_product_key: '',
    scope_version_key: '',
    scope_project_key: '',
    is_active: true,
  }
}

function normalizeOptionalText(value) {
  const normalized = String(value || '').trim()
  return normalized || null
}

function errorDetail(error, fallback) {
  return error?.response?.data?.detail || fallback
}

function requireManage() {
  if (canManage.value) return true
  message.warning('当前仅查看，维护受控术语需要 terminology:manage 权限')
  return false
}

async function loadKnowledgeBases() {
  knowledgeBasesLoading.value = true
  try {
    const result = await getKnowledgeBases()
    knowledgeBases.value = Array.isArray(result) ? result : []
    if (!selectedKbId.value || !knowledgeBases.value.some(item => item.id === selectedKbId.value)) {
      selectedKbId.value = knowledgeBases.value[0]?.id || null
    }
  } catch (error) {
    knowledgeBases.value = []
    message.error(errorDetail(error, '加载可访问知识库失败'))
  } finally {
    knowledgeBasesLoading.value = false
  }
}

async function loadRegistry(kbId, requestRevision = selectionRevision) {
  if (!kbId || !canRead.value) return
  registryLoading.value = true
  registryLoadError.value = ''
  try {
    const result = await getTerminologyRegistry(kbId, { includeInactive: true })
    if (requestRevision !== selectionRevision || kbId !== selectedKbId.value) return
    registry.value = result
  } catch (error) {
    if (requestRevision !== selectionRevision || kbId !== selectedKbId.value) return
    registry.value = null
    registryLoadError.value = errorDetail(error, '请稍后重试，或确认数据库迁移已完成。')
  } finally {
    if (requestRevision === selectionRevision && kbId === selectedKbId.value) {
      registryLoading.value = false
    }
  }
}

async function loadDocuments(kbId, requestRevision = selectionRevision) {
  if (!kbId || !canReadDocuments.value) return
  documentsLoading.value = true
  documentsLoadError.value = ''
  try {
    const result = await getAllDocuments(kbId)
    if (requestRevision !== selectionRevision || kbId !== selectedKbId.value) return
    documents.value = Array.isArray(result) ? result : []
  } catch (error) {
    if (requestRevision !== selectionRevision || kbId !== selectedKbId.value) return
    documents.value = []
    documentsLoadError.value = errorDetail(error, '文档范围选项暂时无法加载；仍可创建不限定文档的绑定。')
  } finally {
    if (requestRevision === selectionRevision && kbId === selectedKbId.value) {
      documentsLoading.value = false
    }
  }
}

function refreshSelectedRegistry() {
  if (!selectedKbId.value) return
  const requestRevision = selectionRevision
  void loadRegistry(selectedKbId.value, requestRevision)
  if (canReadDocuments.value) void loadDocuments(selectedKbId.value, requestRevision)
}

function renderStatusTag(isActive) {
  return h(NTag, {
    type: isActive ? 'success' : 'default',
    size: 'small',
    round: true,
    bordered: false,
  }, { default: () => isActive ? '启用' : '已停用' })
}

function renderTerms(concept) {
  // The canonical spelling is already displayed in its own column. Keeping it
  // out of this cell makes the remaining list truthful: it is the alias list,
  // not a second rendering of the concept title.
  const terms = (Array.isArray(concept.terms) ? concept.terms : [])
    .filter(term => term.term !== concept.canonical_term)
  if (!terms.length) return h('span', { class: 'terminology-page__muted' }, '暂无附加同义术语')
  return h('div', { class: 'terminology-page__term-list' }, terms.map(term => {
    const label = `${term.term}${term.match_mode === 'retrieval_only' ? '（召回）' : ''}${term.is_active ? '' : '（停用）'}`
    const title = `${term.match_mode === 'strict_equivalent' ? '严格等价' : '仅用于召回'}${term.is_active ? '' : '，已停用'}`
    if (!canManage.value) {
      return h(NTag, {
        key: term.id,
        type: term.is_active ? (term.match_mode === 'strict_equivalent' ? 'info' : 'default') : 'default',
        size: 'small',
        round: true,
        bordered: false,
        title,
      }, { default: () => label })
    }
    return h(NButton, {
      key: term.id,
      text: true,
      type: term.is_active ? 'primary' : 'default',
      size: 'small',
      title: `编辑${label}：${title}`,
      'aria-label': `编辑术语 ${term.term}`,
      onClick: () => openEditTerm(concept, term),
    }, { default: () => label })
  }))
}

function actionButton(label, onClick) {
  return h(NButton, {
    text: true,
    type: 'primary',
    size: 'small',
    onClick,
  }, { default: () => label })
}

function formatScope(binding) {
  const values = []
  if (binding.document_id) values.push(`文档：${documentNameById.value.get(binding.document_id) || '指定文档'}`)
  if (binding.scope_product_key) values.push(`产品：${binding.scope_product_key}`)
  if (binding.scope_version_key) values.push(`版本：${binding.scope_version_key}`)
  if (binding.scope_project_key) values.push(`项目：${binding.scope_project_key}`)
  return values.length ? values.join('；') : '整个知识库'
}

function openCreateConcept() {
  if (!requireManage() || !selectedKbId.value) return
  conceptMode.value = 'create'
  editingConcept.value = null
  conceptForm.value = emptyConceptForm()
  showConceptModal.value = true
}

function openEditConcept(concept) {
  if (!requireManage()) return
  conceptMode.value = 'edit'
  editingConcept.value = concept
  conceptForm.value = {
    ...emptyConceptForm(),
    code: concept.code,
    canonical_term: concept.canonical_term,
    description: concept.description || '',
    is_active: concept.is_active,
  }
  showConceptModal.value = true
}

function resetConceptDialog() {
  if (savingConcept.value) return
  conceptFormRef.value?.restoreValidation?.()
  editingConcept.value = null
  conceptForm.value = emptyConceptForm()
}

async function saveConcept() {
  if (!requireManage() || !selectedKbId.value || savingConcept.value) return
  try {
    await conceptFormRef.value?.validate()
  } catch {
    return
  }
  savingConcept.value = true
  try {
    let result
    if (conceptMode.value === 'create') {
      result = await createTerminologyConcept(selectedKbId.value, {
        code: conceptForm.value.code.trim(),
        canonical_term: conceptForm.value.canonical_term.trim(),
        description: normalizeOptionalText(conceptForm.value.description),
        initial_binding: {
          document_id: documentIdFromScopeValue(conceptForm.value.initial_document_id),
          scope_product_key: normalizeOptionalText(conceptForm.value.initial_scope_product_key),
          scope_version_key: normalizeOptionalText(conceptForm.value.initial_scope_version_key),
          scope_project_key: normalizeOptionalText(conceptForm.value.initial_scope_project_key),
          is_active: true,
        },
      })
      message.success('受控术语概念已创建')
    } else {
      result = await updateTerminologyConcept(selectedKbId.value, editingConcept.value.id, {
        canonical_term: conceptForm.value.canonical_term.trim(),
        description: normalizeOptionalText(conceptForm.value.description),
        is_active: conceptForm.value.is_active,
      })
      message.success('概念变更已保存')
    }
    applyMutation(result)
    showConceptModal.value = false
  } catch (error) {
    message.error(errorDetail(error, conceptMode.value === 'create' ? '创建概念失败' : '保存概念失败'))
  } finally {
    savingConcept.value = false
  }
}

function openCreateTerm(concept) {
  if (!requireManage()) return
  termMode.value = 'create'
  termConcept.value = concept
  editingTerm.value = null
  termForm.value = emptyTermForm()
  showTermModal.value = true
}

function openEditTerm(concept, term) {
  if (!requireManage()) return
  termMode.value = 'edit'
  termConcept.value = concept
  editingTerm.value = term
  termForm.value = {
    term: term.term,
    match_mode: term.match_mode,
    is_active: term.is_active,
  }
  showTermModal.value = true
}

function resetTermDialog() {
  if (savingTerm.value) return
  termFormRef.value?.restoreValidation?.()
  termConcept.value = null
  editingTerm.value = null
  termForm.value = emptyTermForm()
}

async function saveTerm() {
  if (!requireManage() || !selectedKbId.value || !termConcept.value || savingTerm.value) return
  try {
    await termFormRef.value?.validate()
  } catch {
    return
  }
  savingTerm.value = true
  try {
    const payload = {
      term: termForm.value.term.trim(),
      match_mode: termForm.value.match_mode,
      is_active: termForm.value.is_active,
    }
    const result = termMode.value === 'create'
      ? await createTerminologyTerm(selectedKbId.value, termConcept.value.id, payload)
      : await updateTerminologyTerm(selectedKbId.value, termConcept.value.id, editingTerm.value.id, payload)
    applyMutation(result)
    message.success(termMode.value === 'create' ? '同义术语已创建' : '同义术语变更已保存')
    showTermModal.value = false
  } catch (error) {
    message.error(errorDetail(error, termMode.value === 'create' ? '创建同义术语失败' : '保存同义术语失败'))
  } finally {
    savingTerm.value = false
  }
}

function openCreateBinding() {
  if (!requireManage() || !selectedKbId.value || !concepts.value.length) return
  bindingMode.value = 'create'
  editingBinding.value = null
  bindingForm.value = { ...emptyBindingForm(), concept_id: concepts.value[0].id }
  showBindingModal.value = true
}

function openEditBinding(binding) {
  if (!requireManage()) return
  bindingMode.value = 'edit'
  editingBinding.value = binding
  bindingForm.value = {
    concept_id: binding.concept_id,
    document_id: binding.document_id || KB_WIDE_SCOPE,
    scope_product_key: binding.scope_product_key || '',
    scope_version_key: binding.scope_version_key || '',
    scope_project_key: binding.scope_project_key || '',
    is_active: binding.is_active,
  }
  showBindingModal.value = true
}

function resetBindingDialog() {
  if (savingBinding.value) return
  bindingFormRef.value?.restoreValidation?.()
  editingBinding.value = null
  bindingForm.value = emptyBindingForm()
}

function documentIdFromScopeValue(value) {
  if (!canReadDocuments.value || value === KB_WIDE_SCOPE || !value) return null
  return value
}

async function saveBinding() {
  if (!requireManage() || !selectedKbId.value || savingBinding.value) return
  try {
    await bindingFormRef.value?.validate()
  } catch {
    return
  }
  // This client-side boundary prevents an administrator without doc:read
  // from accidentally submitting a remembered document ID. The API remains
  // authoritative and independently validates KB/document access.
  if (!canReadDocuments.value && bindingForm.value.document_id !== KB_WIDE_SCOPE) {
    message.warning('当前没有文档查看权限，不能创建或编辑文档范围绑定')
    return
  }
  savingBinding.value = true
  try {
    const payload = {
      ...(bindingMode.value === 'create' ? { concept_id: bindingForm.value.concept_id } : {}),
      document_id: documentIdFromScopeValue(bindingForm.value.document_id),
      scope_product_key: normalizeOptionalText(bindingForm.value.scope_product_key),
      scope_version_key: normalizeOptionalText(bindingForm.value.scope_version_key),
      scope_project_key: normalizeOptionalText(bindingForm.value.scope_project_key),
      is_active: bindingForm.value.is_active,
    }
    const result = bindingMode.value === 'create'
      ? await createTerminologyBinding(selectedKbId.value, payload)
      : await updateTerminologyBinding(selectedKbId.value, editingBinding.value.id, payload)
    applyMutation(result)
    message.success(bindingMode.value === 'create' ? '作用域绑定已创建' : '作用域绑定变更已保存')
    showBindingModal.value = false
  } catch (error) {
    message.error(errorDetail(error, bindingMode.value === 'create' ? '创建作用域绑定失败' : '保存作用域绑定失败'))
  } finally {
    savingBinding.value = false
  }
}

function applyMutation(result) {
  if (result?.registry && result.registry.kb_id === selectedKbId.value) {
    registry.value = result.registry
    registryLoadError.value = ''
    return
  }
  refreshSelectedRegistry()
}

</script>

<style scoped>
.terminology-page {
  height: 100%;
  overflow-y: auto;
  padding: 16px;
}

.terminology-page__inner {
  max-width: 1180px;
  margin: 0 auto;
  display: grid;
  gap: 20px;
}

.terminology-page__empty {
  min-height: 220px;
  display: grid;
  place-content: center;
  text-align: center;
}

.terminology-page__selector-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 420px);
  align-items: end;
  gap: 18px;
}

.terminology-page__selector-label,
.terminology-page__section-head h2,
.terminology-page__form-section h3 {
  color: var(--ui-text);
  font-size: 14px;
  font-weight: 650;
}

.terminology-page__selector-copy p,
.terminology-page__section-head p,
.terminology-page__form-section p {
  margin: 5px 0 0;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.6;
}

.terminology-page__selector-control {
  display: flex;
  align-items: center;
  gap: 10px;
}

.terminology-page__selector-control :deep(.n-base-selection) {
  min-width: 0;
  flex: 1;
}

.terminology-page__revision {
  flex: 0 0 auto;
  color: var(--ui-text-secondary);
  font-size: 12px;
  white-space: nowrap;
}

.terminology-page__permission-alert { margin: 0; }

.terminology-page__table-card { overflow: hidden; }

.terminology-page__section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px;
  border-bottom: 1px solid var(--ui-divider);
}

.terminology-page__section-head h2,
.terminology-page__form-section h3 { margin: 0; }

.terminology-page__table :deep(.n-data-table-th) {
  color: var(--ui-text-secondary);
  font-size: 12px;
  font-weight: 600;
}

.terminology-page__row-actions,
.terminology-page__term-list {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}

.terminology-page__concept-cell {
  display: grid;
  gap: 2px;
}

.terminology-page__concept-cell strong { color: var(--ui-text); font-weight: 600; }
.terminology-page__concept-cell span,
.terminology-page__muted { color: var(--ui-text-tertiary); font-size: 12px; }
.terminology-page__scope-cell { color: var(--ui-text-secondary); line-height: 1.55; }

.terminology-page__form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 16px;
}

.terminology-page__form-section {
  margin: 4px 0 18px;
  padding: 16px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control, 10px);
  background: var(--ui-surface-muted);
}

.terminology-page__form-section > div:first-child { margin-bottom: 12px; }

@media (min-width: 640px) {
  .terminology-page { padding: 24px; }
}

@media (max-width: 639px) {
  .terminology-page__selector-card,
  .terminology-page__form-grid { grid-template-columns: minmax(0, 1fr); }

  .terminology-page__selector-control,
  .terminology-page__section-head {
    align-items: stretch;
    flex-direction: column;
  }

  .terminology-page__selector-control :deep(.n-base-selection) { width: 100%; }
  .terminology-page__section-head :deep(.n-button) { min-height: 40px; }
}
</style>
