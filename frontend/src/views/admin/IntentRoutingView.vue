<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-6">
      <PageHeader title="智能路由" description="分别记录用户意图、回答模式和检索策略，由后端策略层决定最终是否查询知识库。">
        <template #meta>
          <n-tag :type="routingActive ? 'success' : 'default'" :bordered="false" round>
            {{ canRead ? (routingActive ? '路由已启用' : '路由未启用') : '无访问权限' }}
          </n-tag>
        </template>
      </PageHeader>

      <SurfaceCard
        v-if="!canRead"
        class="px-6 py-12 text-center"
      >
        <n-icon :size="30" class="text-gray-400"><LockClosedOutline /></n-icon>
        <div class="mt-3 text-sm font-medium text-gray-700 dark:text-gray-200">暂无查看智能路由的权限</div>
        <p class="mt-1 text-xs text-gray-400">请联系管理员为当前角色授予 <code>intent:read</code> 权限。</p>
      </SurfaceCard>

      <n-spin v-else :show="initialLoading">
        <!-- NSpin 不会为多个 slot 子项自动添加间距；统一由此容器管理页面模块的纵向节奏。 -->
        <div class="space-y-6">
          <!-- 路由策略 -->
          <SurfaceCard>
          <div class="flex flex-wrap items-start justify-between gap-3 mb-5">
            <div>
              <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                路由策略
              </h3>
              <p class="mt-1 text-xs text-gray-400">模型只负责识别意图；回答模式和是否检索由服务端策略层独立决策，并继续受权限校验约束。</p>
            </div>
            <div class="flex items-center gap-3">
              <span v-if="!canManage" class="text-xs text-gray-400">当前仅可查看</span>
              <n-button v-if="canManage" type="primary" :loading="savingConfig" @click="saveConfig">保存策略</n-button>
            </div>
          </div>

          <n-form :model="config" label-placement="top">
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-5">
              <n-form-item label="开启智能路由">
                <n-switch v-model:value="config.enabled" :disabled="!canManage || savingConfig">
                  <template #checked>开启</template>
                  <template #unchecked>关闭</template>
                </n-switch>
              </n-form-item>
              <n-form-item label="判定模式">
                <n-select
                  v-model:value="config.mode" :options="modeOptions"
                  :disabled="!canManage || savingConfig"
                />
              </n-form-item>
              <n-form-item label="低置信度阈值">
                <n-input-number
                  v-model:value="config.confidence_threshold" :min="0" :max="1" :step="0.05"
                  :disabled="!canManage || savingConfig" class="w-full"
                />
              </n-form-item>
              <n-form-item label="无法判定时的兜底意图">
                <n-select
                  v-model:value="config.fallback_intent_code" :options="fallbackOptions"
                  :disabled="!canManage || savingConfig" filterable
                />
              </n-form-item>
              <n-form-item label="允许非检索回答">
                <n-switch v-model:value="config.allow_general_chat" :disabled="!canManage || savingConfig">
                  <template #checked>允许</template>
                  <template #unchecked>仅知识库</template>
                </n-switch>
              </n-form-item>
            </div>
          </n-form>
          </SurfaceCard>

          <!-- 意图分类 -->
          <SurfaceCard padding="none" class="overflow-hidden">
          <div class="px-5 sm:px-6 py-5 flex flex-wrap items-start justify-between gap-3 border-b border-gray-100 dark:border-gray-700">
            <div>
              <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-purple-500 inline-block"></span>
                意图分类
              </h3>
              <p class="mt-1 text-xs text-gray-400">描述和示例会提供给分类器；每行一个示例问题，未匹配时按策略进入兜底意图。</p>
            </div>
            <n-button v-if="canManage" type="primary" @click="openCreateCategory">
              <template #icon><n-icon><AddOutline /></n-icon></template>
              新增意图
            </n-button>
          </div>

          <n-data-table
            :columns="categoryColumns" :data="categories" :loading="categoriesLoading"
            :scroll-x="ui.isCompact ? 880 : undefined"
            class="intent-routing-table"
          />
          </SurfaceCard>

          <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <!-- 在线测试 -->
          <SurfaceCard>
            <div class="flex items-start justify-between gap-3 mb-4">
              <div>
                <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
                  <span class="w-2 h-2 rounded-full bg-orange-500 inline-block"></span>
                  在线测试
                </h3>
                <p class="mt-1 text-xs text-gray-400">只模拟无知识库场景下的最终路由决策，不会创建对话或执行真实检索。</p>
              </div>
              <n-button type="primary" :loading="testing" :disabled="!testQuery.trim()" @click="runTest">测试路由</n-button>
            </div>

            <n-input
              v-model:value="testQuery" type="textarea" :rows="4"
              placeholder="例如：请说明公司的报销审批流程"
              @keyup.ctrl.enter="runTest"
              @keyup.meta.enter="runTest"
            />
            <p class="mt-1.5 text-xs text-gray-400">按 Ctrl / Cmd + Enter 可快速测试。</p>

            <div v-if="testResult" class="mt-4 rounded-lg bg-gray-50 dark:bg-gray-700/40 border border-gray-100 dark:border-gray-700 p-4">
              <div class="flex flex-wrap items-center gap-2 mb-3">
                <span class="text-sm font-medium text-gray-700 dark:text-gray-200">路由结果</span>
                <n-tag type="info" size="small" :bordered="false">{{ testResult.intent_code || testResult.intent || 'unknown' }}</n-tag>
                <n-tag size="small" :type="actionTagType(testAction)" :bordered="false">{{ actionLabel(testAction) }}</n-tag>
                <n-tag size="small" :type="retrievalPolicyTagType(retrievalPolicyFor(testResult))" :bordered="false">
                  {{ retrievalPolicyLabel(retrievalPolicyFor(testResult)) }}
                </n-tag>
              </div>
              <div class="grid grid-cols-1 gap-x-4 gap-y-3 text-xs sm:grid-cols-2">
                <div>
                  <div class="text-gray-400">判定来源</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ sourceLabel(testResult.decision_source || testResult.source) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">置信度</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ formatConfidence(testResult.confidence) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">最终是否检索</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ needsRetrieval(testResult) ? '是' : '否' }}</div>
                </div>
                <div>
                  <div class="text-gray-400">回答模式</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ responseModeLabel(responseModeFor(testResult)) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">检索策略</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ retrievalPolicyLabel(retrievalPolicyFor(testResult)) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">执行状态</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ retrievalExecutionLabel(testResult, true) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">证据状态</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ evidenceStatusLabel(testResult.evidence_status, true) }}</div>
                </div>
                <div v-if="testResult.latency_ms !== undefined && testResult.latency_ms !== null">
                  <div class="text-gray-400">耗时</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ testResult.latency_ms }} ms</div>
                </div>
              </div>
              <div v-if="testReason" class="mt-3 border-t border-gray-200 pt-3 text-xs leading-relaxed text-gray-500 dark:border-gray-600 dark:text-gray-400">
                <span class="text-gray-400">策略原因：</span>
                <span :title="testReason">{{ decisionReasonLabel(testReason) }}</span>
              </div>
            </div>
          </SurfaceCard>

          <!-- 运行说明 -->
          <section class="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-800 rounded-xl border border-blue-100 dark:border-gray-700 p-5 sm:p-6">
            <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200">安全的路由边界</h3>
            <div class="mt-4 space-y-3 text-sm text-gray-600 dark:text-gray-300">
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">1</span><span>规则和模型只产生意图分类，不能直接控制接口、知识库或越过权限边界。</span></div>
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">2</span><span>后端策略层结合意图与知识库选择状态，独立确定回答模式和检索策略。</span></div>
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">3</span><span>日志同时记录原始分类、最终策略和证据状态，便于区分“跳过检索”与“检索无命中”。</span></div>
            </div>
          </section>
          </div>

          <!-- 路由日志 -->
          <SurfaceCard padding="none" class="overflow-hidden">
          <div class="px-5 sm:px-6 py-5 flex flex-wrap items-start justify-between gap-3 border-b border-gray-100 dark:border-gray-700">
            <div>
              <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-green-500 inline-block"></span>
                路由日志
              </h3>
              <p class="mt-1 text-xs text-gray-400">默认显示决策元数据，便于观察命中情况；不依赖完整问题正文进行调优。</p>
            </div>
            <n-button secondary size="small" :loading="logsLoading" @click="loadLogs">刷新日志</n-button>
          </div>

          <n-data-table
            remote :columns="logColumns" :data="logs" :loading="logsLoading"
            :pagination="logPagination" :scroll-x="1580"
            class="intent-routing-table"
          />
          </SurfaceCard>
        </div>
      </n-spin>
    </div>

    <AppModal
      v-model:show="categoryModalVisible"
      :title="editingCategoryId ? '编辑意图分类' : '新增意图分类'"
      width="min(92vw, 680px)"
      :loading="savingCategory"
    >
      <n-form :model="categoryForm" label-placement="top">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-4">
          <n-form-item label="名称" required>
            <n-input v-model:value="categoryForm.name" placeholder="例如：知识库问答" />
          </n-form-item>
          <n-form-item label="意图编码" required>
            <n-input v-model:value="categoryForm.code" :disabled="!!editingCategoryId" placeholder="例如：knowledge_qa" />
          </n-form-item>
        </div>
        <p class="-mt-3 mb-3 text-xs text-gray-400">编码仅使用小写字母、数字和下划线，创建后不可修改；它会成为服务端允许返回的白名单值。</p>
        <n-form-item label="说明">
          <n-input v-model:value="categoryForm.description" type="textarea" :rows="2" placeholder="说明该类问题的范围和判断边界" />
        </n-form-item>
        <n-form-item label="示例问题">
          <n-input
            v-model:value="categoryForm.examplesText" type="textarea" :rows="5"
            placeholder="每行一个示例，例如：\n公司差旅报销需要哪些材料？\n请查一下采购审批流程"
          />
        </n-form-item>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-x-4">
          <n-form-item label="路由动作">
            <n-select v-model:value="categoryForm.action" :options="actionOptions" />
          </n-form-item>
          <n-form-item label="优先级">
            <n-input-number v-model:value="categoryForm.priority" :min="-10000" :max="10000" class="w-full" />
          </n-form-item>
          <n-form-item label="状态">
            <n-switch v-model:value="categoryForm.enabled">
              <template #checked>启用</template>
              <template #unchecked>停用</template>
            </n-switch>
          </n-form-item>
        </div>
      </n-form>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="savingCategory" @click="categoryModalVisible = false">取消</n-button>
          <n-button type="primary" :loading="savingCategory" @click="saveCategory">保存</n-button>
        </div>
      </template>
    </AppModal>

    <DangerConfirm
      v-model:show="showCategoryDeleteConfirm"
      title="删除意图分类？"
      :subject="pendingCategoryDelete?.name || ''"
      description="删除后，分类说明、示例问题和路由动作配置都无法恢复。"
      :loading="deletingCategory"
      @confirm="confirmDeleteCategory"
      @cancel="pendingCategoryDelete = null"
    />
  </div>
</template>

<script setup>
import { computed, h, onMounted, reactive, ref } from 'vue'
import {
  NButton, NDataTable, NForm, NFormItem, NIcon, NInput, NInputNumber,
  NSelect, NSpin, NSwitch, NTag, useMessage,
} from 'naive-ui'
import { AddOutline, LockClosedOutline } from '@vicons/ionicons5'
import {
  createIntentCategory, deleteIntentCategory, getIntentCategories,
  getIntentRouteLogs, getIntentRoutingConfig, submitIntentRouteFeedback,
  testIntentRouting, updateIntentCategory, updateIntentRoutingConfig,
} from '@/api/intentRouting'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import RowActions from '@/components/ui/RowActions.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'

const msg = useMessage()
const authStore = useAuthStore()
const ui = useUiStore()

const DEFAULT_CONFIG = {
  enabled: true,
  mode: 'rules_then_llm',
  confidence_threshold: 0.65,
  fallback_intent_code: 'other',
  allow_general_chat: true,
}

const config = ref({ ...DEFAULT_CONFIG })
const categories = ref([])
const logs = ref([])
const initialLoading = ref(false)
const categoriesLoading = ref(false)
const logsLoading = ref(false)
const savingConfig = ref(false)
const savingCategory = ref(false)
const testing = ref(false)
const testQuery = ref('')
const testResult = ref(null)

const categoryModalVisible = ref(false)
const editingCategoryId = ref(null)
const categoryForm = ref(newCategoryForm())
const showCategoryDeleteConfirm = ref(false)
const pendingCategoryDelete = ref(null)
const deletingCategory = ref(false)

const canRead = computed(() => authStore.hasPerm('intent:read'))
const canManage = computed(() => authStore.hasPerm('intent:manage'))
const routingActive = computed(() => config.value.enabled && config.value.mode !== 'off')

const modeOptions = [
  { label: '规则优先 + 模型兜底', value: 'rules_then_llm' },
  { label: '仅模型分类', value: 'llm_only' },
  { label: '关闭分类，仅安全兜底', value: 'off' },
]
const actionOptions = [
  { label: '知识库检索问答', value: 'retrieve' },
  { label: '通用回答', value: 'chat' },
  { label: '写作 / 润色', value: 'writing' },
  { label: '系统使用帮助', value: 'system_help' },
]
const fallbackOptions = computed(() => {
  const items = categories.value
    .filter(item => item.enabled && item.action === 'retrieve')
    .map(item => ({ label: `${item.name}（${item.code}）`, value: item.code }))
  if (!items.some(item => item.value === 'other')) items.push({ label: '其他 / 未识别（other）', value: 'other' })
  if (config.value.fallback_intent_code && !items.some(item => item.value === config.value.fallback_intent_code)) {
    items.unshift({ label: `当前值（${config.value.fallback_intent_code}）`, value: config.value.fallback_intent_code })
  }
  return items
})

const logPagination = reactive({
  page: 1,
  pageSize: 10,
  itemCount: 0,
  showSizePicker: true,
  pageSizes: [10, 20, 30, 50],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (page) => { logPagination.page = page; loadLogs() },
  onUpdatePageSize: (pageSize) => {
    logPagination.pageSize = pageSize
    logPagination.page = 1
    loadLogs()
  },
})

const categoryColumns = [
  { title: '名称', key: 'name', width: 150, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  {
    title: '编码', key: 'code', width: 150, align: 'left', titleAlign: 'left',
    render: row => h('code', { class: 'text-xs text-blue-600 dark:text-blue-400' }, row.code),
  },
  {
    title: '说明 / 示例', key: 'description', minWidth: 220, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true },
    render: row => row.description || (row.examples?.length ? `示例：${row.examples[0]}` : '—'),
  },
  {
    title: '动作', key: 'action', width: 125, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', type: actionTagType(row.action), bordered: false }, () => actionLabel(row.action)),
  },
  {
    title: '示例数', key: 'examples', width: 90, align: 'center', titleAlign: 'center',
    render: row => row.examples?.length || 0,
  },
  {
    title: '状态', key: 'enabled', width: 90, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', type: row.enabled ? 'success' : 'default', bordered: false }, () => row.enabled ? '启用' : '停用'),
  },
  { title: '优先级', key: 'priority', width: 85, align: 'center', titleAlign: 'center', render: row => row.priority ?? 0 },
  {
    title: '操作', key: 'actions', width: 140, align: 'center', titleAlign: 'center',
    render: row => h(RowActions, { label: `意图 ${row.name} 操作` }, {
      default: () => [
        h(NButton, { text: true, type: 'primary', size: 'small', disabled: !canManage.value, onClick: () => openEditCategory(row) }, () => '编辑'),
        h(NButton, { text: true, type: 'error', size: 'small', disabled: !canManage.value, onClick: () => openDeleteCategory(row) }, () => '删除'),
      ],
    }),
  },
]

const logColumns = [
  {
    title: '时间', key: 'created_at', width: 160, align: 'center', titleAlign: 'center',
    render: row => h('span', { class: 'whitespace-nowrap' }, formatTime(row.created_at)),
  },
  {
    title: '识别意图', key: 'intent_code', width: 190, align: 'left', titleAlign: 'left',
    render: row => h('div', { class: 'flex min-w-0 flex-col gap-0.5 py-1' }, [
      h('span', { class: 'text-sm truncate' }, row.intent_name || '—'),
      h('code', {
        class: 'truncate text-xs text-blue-600 dark:text-blue-400',
        title: row.intent_code || row.intent || '',
      }, row.intent_code || row.intent || '—'),
    ]),
  },
  {
    title: '分类动作', key: 'action', width: 115, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', type: actionTagType(row.action), bordered: false }, () => actionLabel(row.action)),
  },
  {
    title: '判定', key: 'decision_source', width: 115, align: 'center', titleAlign: 'center',
    render: row => h('div', { class: 'flex flex-col items-center gap-0.5 whitespace-nowrap' }, [
      h('span', { class: 'text-sm' }, sourceLabel(row.decision_source || row.source)),
      h('span', { class: 'text-xs text-gray-400' }, formatConfidence(row.confidence)),
    ]),
  },
  {
    title: '最终策略', key: 'retrieval_policy', width: 230, align: 'left', titleAlign: 'left',
    render: row => h('div', { class: 'flex min-w-0 flex-col gap-1.5 py-1' }, [
      h('div', { class: 'flex items-center gap-1.5 whitespace-nowrap' }, [
        h(NTag, { size: 'small', type: responseModeTagType(responseModeFor(row)), bordered: false }, () => responseModeLabel(responseModeFor(row))),
        h(NTag, { size: 'small', type: retrievalPolicyTagType(retrievalPolicyFor(row)), bordered: false }, () => retrievalPolicyLabel(retrievalPolicyFor(row))),
      ]),
      h('span', { class: 'text-xs text-gray-500 dark:text-gray-400' }, `最终检索：${needsRetrieval(row) ? '是' : '否'}`),
    ]),
  },
  {
    title: '策略原因', key: 'decision_reason', width: 220, align: 'left', titleAlign: 'left',
    render: row => {
      const reason = decisionReasonFor(row)
      return h('div', { class: 'flex min-w-0 flex-col gap-0.5 py-1' }, [
        h('span', { class: 'truncate text-sm', title: decisionReasonLabel(reason) }, decisionReasonLabel(reason)),
        h('code', { class: 'truncate text-xs text-gray-400', title: reason }, reason || '—'),
      ])
    },
  },
  {
    title: '执行 / 证据', key: 'evidence_status', width: 180, align: 'left', titleAlign: 'left',
    render: row => {
      const status = evidenceStatusFor(row)
      return h('div', { class: 'flex min-w-0 flex-col items-start gap-1 py-1' }, [
        h(NTag, { size: 'small', type: evidenceStatusTagType(status), bordered: false }, () => evidenceStatusLabel(status)),
        h('span', { class: 'truncate text-xs text-gray-500 dark:text-gray-400' }, retrievalExecutionLabel(row)),
        status === 'hit'
          ? h('span', { class: 'text-xs text-gray-400' }, `有效命中：${Number(row.hit_count ?? 0)} 条`)
          : null,
      ])
    },
  },
  {
    title: '上下文', key: 'selected_kb_count', width: 125, align: 'center', titleAlign: 'center',
    render: row => h('div', { class: 'flex flex-col items-center gap-0.5 whitespace-nowrap' }, [
      h('span', { class: 'text-sm' }, `${row.selected_kb_count ?? 0} 个知识库`),
      h('span', { class: 'text-xs text-gray-400' }, row.latency_ms === undefined || row.latency_ms === null ? '耗时 —' : `${row.latency_ms} ms`),
    ]),
  },
  {
    title: '反馈', key: 'feedback', width: 100, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', type: feedbackTagType(row.feedback), bordered: false }, () => feedbackLabel(row.feedback)),
  },
  {
    title: '操作', key: 'actions', width: 145, align: 'center', titleAlign: 'center',
    render: row => h('div', { style: 'display:flex;justify-content:center;gap:6px;align-items:center' }, [
      h(NButton, {
        text: true, type: 'success', size: 'small', disabled: !canManage.value || !!row.feedback,
        onClick: () => saveFeedback(row, 'correct'),
      }, () => '正确'),
      h(NButton, {
        text: true, type: 'error', size: 'small', disabled: !canManage.value || !!row.feedback,
        onClick: () => saveFeedback(row, 'incorrect'),
      }, () => '错误'),
    ]),
  },
]

const testAction = computed(() => testResult.value?.action || testResult.value?.route_action || '')
const testReason = computed(() => decisionReasonFor(testResult.value))

onMounted(loadPage)

function newCategoryForm() {
  return {
    name: '',
    code: '',
    description: '',
    examplesText: '',
    action: 'retrieve',
    enabled: true,
    priority: 0,
  }
}

function normalizeItems(data, key) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.[key])) return data[key]
  if (Array.isArray(data?.items)) return data.items
  return []
}

function normalizeCategory(item) {
  const examples = Array.isArray(item.examples)
    ? item.examples
    : typeof item.examples === 'string' ? item.examples.split(/\r?\n/) : []
  return {
    ...item,
    examples: examples.map(value => String(value).trim()).filter(Boolean),
    enabled: item.enabled !== false,
    priority: Number.isFinite(Number(item.priority)) ? Number(item.priority) : 0,
  }
}

async function loadPage() {
  if (!canRead.value) return
  initialLoading.value = true
  try {
    await Promise.all([loadConfig(), loadCategories(), loadLogs()])
  } finally {
    initialLoading.value = false
  }
}

async function loadConfig() {
  try {
    const data = await getIntentRoutingConfig()
    config.value = { ...DEFAULT_CONFIG, ...(data?.config || data || {}) }
  } catch (error) {
    showError(error, '加载路由策略失败')
  }
}

async function loadCategories() {
  categoriesLoading.value = true
  try {
    const data = await getIntentCategories()
    categories.value = normalizeItems(data, 'categories').map(normalizeCategory)
  } catch (error) {
    showError(error, '加载意图分类失败')
  } finally {
    categoriesLoading.value = false
  }
}

async function loadLogs() {
  if (!canRead.value) return
  logsLoading.value = true
  try {
    const data = await getIntentRouteLogs({ page: logPagination.page, page_size: logPagination.pageSize })
    logs.value = normalizeItems(data, 'logs')
    logPagination.itemCount = Number(data?.total ?? logs.value.length)
  } catch (error) {
    showError(error, '加载路由日志失败')
  } finally {
    logsLoading.value = false
  }
}

async function saveConfig() {
  if (!canManage.value) return
  savingConfig.value = true
  try {
    const payload = {
      enabled: !!config.value.enabled,
      mode: config.value.mode,
      confidence_threshold: Number(config.value.confidence_threshold),
      fallback_intent_code: config.value.fallback_intent_code,
      allow_general_chat: !!config.value.allow_general_chat,
    }
    const data = await updateIntentRoutingConfig(payload)
    config.value = { ...DEFAULT_CONFIG, ...(data?.config || data || payload) }
    msg.success('路由策略已保存')
  } catch (error) {
    showError(error, '保存路由策略失败')
  } finally {
    savingConfig.value = false
  }
}

function openCreateCategory() {
  if (!canManage.value) return
  editingCategoryId.value = null
  categoryForm.value = newCategoryForm()
  categoryModalVisible.value = true
}

function openEditCategory(category) {
  if (!canManage.value) return
  editingCategoryId.value = category.id
  categoryForm.value = {
    name: category.name || '',
    code: category.code || '',
    description: category.description || '',
    examplesText: (category.examples || []).join('\n'),
    action: category.action || 'retrieve',
    enabled: category.enabled !== false,
    priority: category.priority ?? 0,
  }
  categoryModalVisible.value = true
}

async function saveCategory() {
  if (!canManage.value) return
  const name = categoryForm.value.name.trim()
  const code = categoryForm.value.code.trim()
  if (!name) { msg.warning('请输入意图名称'); return }
  if (!/^[a-z][a-z0-9_]*$/.test(code)) {
    msg.warning('意图编码需以小写字母开头，只能包含小写字母、数字和下划线')
    return
  }

  savingCategory.value = true
  try {
    const payload = {
      name,
      code,
      description: categoryForm.value.description.trim(),
      examples: categoryForm.value.examplesText.split(/\r?\n/).map(item => item.trim()).filter(Boolean),
      action: categoryForm.value.action,
      enabled: !!categoryForm.value.enabled,
      priority: Number(categoryForm.value.priority) || 0,
    }
    if (editingCategoryId.value) {
      const { code: _code, ...updatePayload } = payload
      await updateIntentCategory(editingCategoryId.value, updatePayload)
      msg.success('意图分类已更新')
    } else {
      await createIntentCategory(payload)
      msg.success('意图分类已创建')
    }
    categoryModalVisible.value = false
    await loadCategories()
  } catch (error) {
    showError(error, '保存意图分类失败')
  } finally {
    savingCategory.value = false
  }
}

function openDeleteCategory(category) {
  if (!canManage.value) return
  pendingCategoryDelete.value = category
  showCategoryDeleteConfirm.value = true
}

async function confirmDeleteCategory() {
  const category = pendingCategoryDelete.value
  if (!category || !canManage.value) return
  deletingCategory.value = true
  try {
    await deleteIntentCategory(category.id)
    msg.success('意图分类已删除')
    await loadCategories()
    showCategoryDeleteConfirm.value = false
    pendingCategoryDelete.value = null
  } catch (error) {
    showError(error, '删除意图分类失败')
  } finally {
    deletingCategory.value = false
  }
}

async function runTest() {
  const query = testQuery.value.trim()
  if (!query) return
  testing.value = true
  testResult.value = null
  try {
    const data = await testIntentRouting({ question: query })
    testResult.value = data?.decision
      ? {
          ...data.decision,
          latency_ms: data.latency_ms,
          retrieval_executed: data.retrieval_executed ?? data.decision.retrieval_executed,
          evidence_status: data.evidence_status ?? data.decision.evidence_status,
          hit_count: data.hit_count ?? data.decision.hit_count,
        }
      : data
  } catch (error) {
    showError(error, '测试路由失败')
  } finally {
    testing.value = false
  }
}

async function saveFeedback(log, feedback) {
  if (!canManage.value || log.feedback) return
  try {
    const data = await submitIntentRouteFeedback(log.id, feedback)
    const updated = data?.log || data
    logs.value = logs.value.map(item => item.id === log.id ? { ...item, ...updated, feedback } : item)
    msg.success(feedback === 'correct' ? '已标记为正确' : '已标记为错误')
  } catch (error) {
    showError(error, '保存反馈失败')
  }
}

function actionLabel(action) {
  return {
    retrieve: '知识库检索',
    chat: '通用回答',
    writing: '写作 / 润色',
    system_help: '系统帮助',
  }[action] || action || '—'
}

function actionTagType(action) {
  return {
    retrieve: 'success',
    chat: 'info',
    writing: 'warning',
    system_help: 'default',
  }[action] || 'default'
}

function responseModeFor(result) {
  if (!result) return ''
  if (result.response_mode) return result.response_mode
  return ({
    retrieve: 'grounded_qa',
    chat: 'general_chat',
    writing: 'writing',
    system_help: 'platform_help',
  })[result.action || result.route_action] || ''
}

function responseModeLabel(mode) {
  return {
    grounded_qa: '知识库问答',
    general_chat: '通用回答',
    writing: '写作模式',
    platform_help: '平台帮助',
  }[mode] || mode || '未记录'
}

function responseModeTagType(mode) {
  return {
    grounded_qa: 'success',
    general_chat: 'info',
    writing: 'warning',
    platform_help: 'default',
  }[mode] || 'default'
}

function retrievalPolicyFor(result) {
  if (!result) return ''
  if (result.retrieval_policy) return result.retrieval_policy
  return needsRetrieval(result) ? 'required' : 'skip'
}

function retrievalPolicyLabel(policy) {
  return {
    required: '必须检索',
    optional: '按证据检索',
    skip: '跳过检索',
  }[policy] || policy || '未记录'
}

function retrievalPolicyTagType(policy) {
  return {
    required: 'success',
    optional: 'warning',
    skip: 'default',
  }[policy] || 'default'
}

function decisionReasonFor(result) {
  return result?.decision_reason || result?.reason || result?.message || ''
}

function decisionReasonLabel(reason) {
  return {
    safe_fallback: '分类异常或置信度不足，采用安全检索兜底',
    classification_pending_policy: '分类已完成，等待策略层决策',
    general_chat_disabled: '系统已关闭非检索回答',
    classified_retrieval: '意图分类明确要求知识库检索',
    exact_greeting: '明确的问候或礼貌用语',
    explicit_platform_help: '明确询问当前 RAG 平台功能',
    platform_help_scope_guard: '并非当前平台帮助，策略保护已强制检索',
    inline_writing_content: '用户已提供待处理文本，无需查询知识库',
    knowledge_dependent_writing: '写作任务依赖知识库资料，必须先检索',
    selected_knowledge_context: '已选择知识库，允许使用知识证据',
    no_selected_knowledge: '未选择知识库，按非检索模式回答',
    invalid_action_fallback: '分类动作无效，采用安全检索兜底',
    legacy_action_mapping: '历史日志按原分类动作补全执行策略',
    legacy_probe: '旧接口通过轻量判断生成检索计划',
    explicit_need_retrieval: '调用方明确指定是否检索',
    retrieval_required: '检索策略要求执行知识库检索',
    retrieval_skipped: '检索策略明确跳过知识库检索',
    optional_auto_detection: '可选检索由轻量判断决定',
  }[reason] || reason || '未记录'
}

function evidenceStatusFor(result) {
  const status = result?.evidence_status
  if (['skipped', 'hit', 'no_hit', 'unverified', 'error'].includes(status)) return status
  if (result?.retrieval_executed === false) return 'skipped'
  if (result?.retrieval_executed === true && Number(result?.hit_count) > 0) return 'hit'
  if (result?.retrieval_executed === true && result?.hit_count !== undefined && result?.hit_count !== null) return 'no_hit'
  if (Number(result?.hit_count) > 0) return 'hit'
  return 'unverified'
}

function evidenceStatusLabel(status, simulation = false) {
  if (!status && simulation) return '未执行（仅测试路由策略）'
  return {
    skipped: '已跳过检索',
    hit: '已命中证据',
    no_hit: '已检索但无命中',
    unverified: '状态未验证',
    error: '检索失败',
  }[status] || status || '未记录'
}

function evidenceStatusTagType(status) {
  return {
    skipped: 'default',
    hit: 'success',
    no_hit: 'warning',
    unverified: 'default',
    error: 'error',
  }[status] || 'default'
}

function retrievalExecutionLabel(result, simulation = false) {
  if (result?.retrieval_executed === true) return '已执行知识库检索'
  if (result?.retrieval_executed === false) return '已按策略跳过检索'
  if (simulation) return '未执行（仅测试路由策略）'
  if (result?.evidence_status === 'error') return '检索执行失败'
  return '未记录或请求提前中止'
}

function sourceLabel(source) {
  return { rule: '规则', llm: '模型', fallback: '兜底', policy_fallback: '策略兜底' }[source] || source || '—'
}

function feedbackLabel(feedback) {
  return { correct: '正确', incorrect: '错误' }[feedback] || '未标注'
}

function feedbackTagType(feedback) {
  return { correct: 'success', incorrect: 'error' }[feedback] || 'default'
}

function formatConfidence(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : '—'
}

function needsRetrieval(result) {
  if (!result) return false
  if (result.need_retrieval !== undefined) return !!result.need_retrieval
  if (result.needs_retrieval !== undefined) return !!result.needs_retrieval
  return (result.action || result.route_action) === 'retrieve'
}

function formatTime(value) {
  if (!value) return '—'
  const time = new Date(value)
  return Number.isNaN(time.getTime()) ? String(value) : time.toLocaleString('zh-CN')
}

function showError(error, fallback) {
  msg.error(error?.response?.data?.detail || fallback)
}
</script>
