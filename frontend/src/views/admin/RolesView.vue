<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-5">
      <PageHeader title="角色管理" description="按岗位配置可用能力和知识库范围；页面入口会随对应能力自动开放。">
        <template #actions>
          <n-button type="primary" @click="openCreate">
            <template #icon><n-icon><AddOutline /></n-icon></template>
            新建角色
          </n-button>
        </template>
      </PageHeader>

      <SurfaceCard padding="none" class="overflow-hidden">
        <n-data-table
          :columns="columns" :data="roles" :loading="loading"
          :pagination="pagination" :scroll-x="940"
          class="admin-data-table"
        />
      </SurfaceCard>
    </div>

    <AppModal
      v-model:show="showModal"
      :title="editingId ? '编辑角色' : '新建角色'"
      width="min(94vw, 960px)"
      :loading="saving"
      @close="closeEditor"
    >
      <div class="role-form-scroll">
        <n-form :model="form" :disabled="saving" label-placement="top">
          <div class="role-editor__overview">
            <div class="role-editor__overview-copy">
              <div class="role-editor__eyebrow">ROLE ACCESS</div>
              <p class="role-editor__hint">
                菜单入口不单独授权：勾选能力后，系统会自动开放相应页面入口，并补齐它依赖的基础能力。
              </p>
            </div>
            <n-tag v-if="editingIsSystem" type="info" :bordered="false" round>系统角色</n-tag>
            <n-tag v-else type="default" :bordered="false" round>自定义角色</n-tag>
          </div>

          <div v-if="!editingId" class="role-template-picker">
            <div class="role-template-picker__copy">
              <div class="role-template-picker__title">从模板开始</div>
              <p>模板会填充能力与默认范围，之后仍可逐项调整。</p>
            </div>
            <n-select
              v-model:value="selectedTemplateKey"
              :options="templateOptions"
              clearable
              placeholder="不使用模板，手动配置"
              class="role-template-picker__select"
              :disabled="saving"
              @update:value="applyTemplate"
            />
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-4">
            <n-form-item label="角色名称" required>
              <n-input v-model:value="form.name" :disabled="editingIsSystem || saving" placeholder="例如：知识库运营专员" />
            </n-form-item>
            <n-form-item v-if="isSuperadmin" label="角色编码">
              <n-input
                v-model:value="form.code"
                :disabled="!!editingId || saving"
                placeholder="例如：knowledge_operator"
              />
            </n-form-item>
            <n-form-item label="角色说明" class="sm:col-span-2">
              <n-input v-model:value="form.description" type="textarea" :rows="2" placeholder="说明该角色适用的岗位和职责边界" />
            </n-form-item>
            <n-form-item v-if="isSuperadmin" label="允许分配给用户">
              <div class="role-assignable-control">
                <n-switch v-model:value="form.is_assignable" :disabled="editingIsReservedSuperadmin || saving">
                  <template #checked>可分配</template>
                  <template #unchecked>仅保留</template>
                </n-switch>
                <span>
                  {{ editingIsReservedSuperadmin
                    ? '超级管理员角色固定为不可分配，普通账号不能获得该系统身份。'
                    : '关闭后不能在用户管理中为账号选择此角色。' }}
                </span>
              </div>
            </n-form-item>
          </div>

          <n-tabs v-model:value="activeTab" type="line" animated class="role-editor-tabs">
            <n-tab-pane name="capabilities" tab="功能权限">
              <section class="role-tab-panel" aria-labelledby="role-capability-title">
                <div class="permission-tree-toolbar">
                  <div class="min-w-0">
                    <h3 id="role-capability-title" class="permission-tree-toolbar__title">
                      已选择 {{ selectedCapabilityCount }} 项能力
                    </h3>
                    <p>管理能力会自动补齐读取能力；取消读取能力会同步移除依赖它的管理能力。</p>
                  </div>
                  <div class="flex shrink-0 flex-wrap items-center justify-end gap-1">
                    <n-button text size="small" @click="expandAllModules">展开全部</n-button>
                    <n-button text size="small" @click="collapseAllModules">收起</n-button>
                    <n-button text size="small" @click="clearCapabilities">清空能力</n-button>
                  </div>
                </div>

                <p class="permission-tree-intro">
                  以下按业务域组织。每个模块显示的都是实际可执行能力，入口随能力自动开放，不会出现“看得到但不能用”的菜单。
                </p>

                <div class="role-permission-tree">
                  <section v-for="group in capabilityGroups" :key="group.key" class="permission-tree-group">
                    <header class="permission-tree-group__header">
                      <span class="permission-tree-group__dot" aria-hidden="true"></span>
                      <div class="min-w-0">
                        <h4>{{ group.label }}</h4>
                        <p>{{ group.description || '按业务职责配置可执行能力。' }}</p>
                      </div>
                      <n-tag size="small" :bordered="false" round>{{ groupSelectedCount(group) }} 项已选</n-tag>
                    </header>

                    <div class="permission-tree__modules">
                      <article v-for="module in group.modules" :key="module.key" class="permission-module">
                        <button
                          type="button"
                          class="permission-module__head"
                          :aria-expanded="isModuleExpanded(module.key)"
                          @click="toggleModule(module.key)"
                        >
                          <span class="permission-module__dot" aria-hidden="true"></span>
                          <span class="permission-module__copy">
                            <span class="permission-module__title-row">
                              <strong>{{ module.label }}</strong>
                              <span class="permission-module__count">{{ moduleSelectedCount(module) }} / {{ module.permissions.length }}</span>
                            </span>
                            <span>{{ module.description || '为该模块选择可用能力。' }}</span>
                            <span v-if="module.menuPermission" class="permission-module__entry-hint">
                              选择任一能力后自动开放「{{ module.label }}」入口
                            </span>
                          </span>
                          <n-icon class="permission-module__chevron" :class="{ 'is-open': isModuleExpanded(module.key) }" :size="16">
                            <ChevronDownOutline />
                          </n-icon>
                        </button>

                        <div v-show="isModuleExpanded(module.key)" class="permission-module__abilities">
                          <div v-if="!module.permissions.length" class="permission-module__empty">
                            当前模块暂未定义可配置能力。
                          </div>
                          <div
                            v-for="ability in module.permissions"
                            :key="ability.key"
                            class="capability-option"
                            :class="{ 'capability-option--selected': hasCapability(ability.key), 'capability-option--risk': isHighRisk(ability) }"
                          >
                            <n-checkbox
                              :checked="hasCapability(ability.key)"
                              :disabled="saving || isDelegationRestricted(ability.key)"
                              @update:checked="checked => setCapability(ability.key, checked)"
                            >
                              <span class="capability-option__content">
                                <span class="capability-option__title-row">
                                  <span class="capability-option__title">{{ ability.label }}</span>
                                  <n-tag v-if="isHighRisk(ability)" type="error" size="small" :bordered="false">高风险</n-tag>
                                </span>
                                <span class="capability-option__description">{{ ability.description || '允许执行该项业务能力。' }}</span>
                                <span v-if="ability.requires?.length" class="capability-option__requires">
                                  依赖：{{ dependencyLabels(ability.requires).join('、') }}
                                </span>
                              </span>
                            </n-checkbox>
                          </div>
                        </div>
                      </article>
                    </div>
                  </section>

                  <section v-if="legacyCapabilities.length" class="permission-tree-group permission-tree-group--legacy">
                    <header class="permission-tree-group__header">
                      <span class="permission-tree-group__dot" aria-hidden="true"></span>
                      <div class="min-w-0">
                        <h4>历史能力</h4>
                        <p>这些权限未出现在当前权限目录中，保存时不会继续提交；请联系管理员确认其迁移方式。</p>
                      </div>
                    </header>
                    <div class="legacy-capability-list">
                      <n-tag v-for="key in legacyCapabilities" :key="key" size="small" type="warning" :bordered="false">{{ key }}</n-tag>
                    </div>
                  </section>
                </div>
              </section>
            </n-tab-pane>

            <n-tab-pane name="scope" tab="数据范围">
              <section class="role-tab-panel" aria-labelledby="role-scope-title">
                <div class="scope-heading">
                  <div>
                    <h3 id="role-scope-title">知识库数据范围</h3>
                    <p>范围约束知识库检索和文档访问；新增知识库没有既有对象可校验，因此必须配合“全部知识库”范围。</p>
                  </div>
                  <n-tag size="small" :type="scopeTagType" :bordered="false" round>{{ scopeLabel }}</n-tag>
                </div>

                <n-radio-group v-model:value="form.scope_mode" class="scope-options" :disabled="saving" @update:value="changeScopeMode">
                  <n-radio value="none" class="scope-option" :class="{ 'scope-option--active': form.scope_mode === 'none' }">
                    <span class="scope-option__copy">
                      <strong>不授予知识库访问</strong>
                      <span>该角色不能检索、浏览或维护任何知识库内容。</span>
                    </span>
                  </n-radio>
                  <n-radio value="selected" class="scope-option" :class="{ 'scope-option--active': form.scope_mode === 'selected' }">
                    <span class="scope-option__copy">
                      <strong>指定知识库</strong>
                      <span>只可访问下方选择的知识库，适合部门或项目角色。</span>
                    </span>
                  </n-radio>
                  <n-radio
                    value="all"
                    class="scope-option"
                    :class="{ 'scope-option--active': form.scope_mode === 'all' }"
                    :disabled="!isSuperadmin && allScopeSuperadminOnly"
                  >
                    <span class="scope-option__copy">
                      <strong>全部知识库</strong>
                      <span>可访问当前及后续新建的全部知识库，适合平台级管理员。</span>
                    </span>
                  </n-radio>
                </n-radio-group>

                <div v-if="form.scope_mode === 'selected'" class="scope-selected-panel">
                  <div class="scope-selected-panel__head">
                    <div>
                      <h4>选择可访问知识库</h4>
                      <p>支持搜索和多选；未选择任何知识库时，该范围不会授予实际数据访问权。</p>
                    </div>
                    <span>{{ form.kb_ids.length }} 个已选</span>
                  </div>
                  <n-select
                    v-model:value="form.kb_ids"
                    :options="kbOptions"
                    multiple
                    filterable
                    clearable
                    :max-tag-count="3"
                    placeholder="搜索并选择知识库"
                    :disabled="saving"
                  />
                </div>

                <n-alert v-else-if="form.scope_mode === 'all'" type="warning" :show-icon="true" class="scope-alert">
                  <template #header>全部范围会自动覆盖未来新建的知识库</template>
                  已清空指定知识库名单，并将以平台级范围保存。请只授予确有跨业务访问职责的角色。
                </n-alert>

                <n-alert v-else :type="hasScopeRequiredCapability ? 'warning' : 'info'" :show-icon="true" class="scope-alert">
                  <template #header>{{ hasScopeRequiredCapability ? '请选择知识库范围' : '当前不授予知识库范围' }}</template>
                  {{ hasScopeRequiredCapability
                    ? '当前能力包含知识库、文档或检索访问，保存前必须改为“指定知识库”或“全部知识库”。'
                    : '角色仍可使用不依赖知识库数据的问答、智能路由或系统能力。' }}
                </n-alert>

                <n-alert
                  v-if="hasKnowledgeCreateCapability && form.scope_mode !== 'all'"
                  type="warning"
                  :show-icon="true"
                  class="scope-alert"
                >
                  <template #header>新增知识库需要全部知识库范围</template>
                  请将数据范围改为“全部知识库”；指定范围只能校验已有知识库，无法覆盖尚未创建的新对象。
                </n-alert>

                <p v-if="form.scope_mode !== 'none' && !hasKnowledgeDataCapability" class="scope-consistency-hint">
                  当前尚未选择会使用知识库数据的能力；范围会保存，但暂时不会带来可执行的数据访问。
                </p>
              </section>
            </n-tab-pane>

            <n-tab-pane name="summary" tab="授权摘要">
              <section class="role-tab-panel role-summary" aria-live="polite">
                <div class="role-summary__hero">
                  <span class="role-summary__hero-icon" aria-hidden="true"><n-icon :size="20"><ShieldCheckmarkOutline /></n-icon></span>
                  <div>
                    <h3>{{ form.name.trim() || '未命名角色' }} 的授权概览</h3>
                    <p>{{ naturalLanguageSummary }}</p>
                  </div>
                </div>

                <div class="role-summary__grid">
                  <section class="summary-surface">
                    <h4>自动开放的功能入口</h4>
                    <ul v-if="activeModules.length" class="summary-list">
                      <li v-for="module in activeModules" :key="module.key">
                        <strong>{{ module.label }}</strong>
                        <span>{{ moduleSelectedAbilities(module).map(item => item.label).join('、') }}</span>
                      </li>
                    </ul>
                    <p v-else class="summary-empty">尚未选择任何可执行能力，因此不会开放应用入口。</p>
                  </section>

                  <section class="summary-surface">
                    <h4>知识库范围</h4>
                    <p class="summary-scope-text">{{ scopeNarrative }}</p>
                    <div v-if="form.scope_mode === 'selected' && selectedKnowledgeBases.length" class="summary-kb-list">
                      <n-tag v-for="kb in selectedKnowledgeBases" :key="kb.id" size="small" type="info" :bordered="false">{{ kb.name }}</n-tag>
                    </div>
                  </section>
                </div>

                <n-alert v-if="highRiskAbilities.length || form.scope_mode === 'all'" type="warning" :show-icon="true" class="role-summary__risk">
                  <template #header>请确认高影响授权</template>
                  <p v-if="highRiskAbilities.length">高风险能力：{{ highRiskAbilities.map(item => item.label).join('、') }}。</p>
                  <p v-if="form.scope_mode === 'all'">该角色拥有全部知识库范围，未来新建知识库也会自动可访问。</p>
                </n-alert>

                <div v-if="legacyCapabilities.length" class="role-summary__legacy">
                  <n-icon :size="16"><InformationCircleOutline /></n-icon>
                  {{ legacyCapabilities.length }} 项未识别历史能力不会写回角色，请先完成迁移确认。
                </div>
              </section>
            </n-tab-pane>
          </n-tabs>
        </n-form>
      </div>

      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="saving" @click="closeEditor">取消</n-button>
          <n-button type="primary" :loading="saving" @click="handleSave">
            {{ editingId ? '保存角色' : '创建角色' }}
          </n-button>
        </div>
      </template>
    </AppModal>

    <AppModal
      v-model:show="showGrantConfirm"
      title="确认高影响授权"
      width="min(92vw, 500px)"
      :loading="saving"
      :mask-closable="false"
      @close="cancelGrantConfirm"
    >
      <div class="grant-confirm" role="alert" aria-live="assertive">
        <div class="grant-confirm__icon" aria-hidden="true">
          <n-icon :size="22"><WarningOutline /></n-icon>
        </div>
        <div class="grant-confirm__copy">
          <p class="grant-confirm__lead">
            即将为“{{ pendingGrantRoleName }}”增加高影响权限，请确认这符合该岗位的职责边界。
          </p>
          <ul class="grant-confirm__list">
            <li v-for="item in pendingGrantChanges" :key="item.key">
              <strong>{{ item.label }}</strong>
              <span>{{ item.description }}</span>
            </li>
          </ul>
          <n-alert type="warning" :show-icon="false" class="grant-confirm__notice">
            保存后立即影响该角色下的用户；后端仍会按授权上限校验，超出当前管理员权限的配置不会被接受。
          </n-alert>
        </div>
      </div>

      <template #footer>
        <n-button :disabled="saving" @click="cancelGrantConfirm">返回检查</n-button>
        <n-button type="warning" :loading="saving" @click="confirmGrantSave">确认授权并保存</n-button>
      </template>
    </AppModal>

    <DangerConfirm
      v-model:show="showDeleteConfirm"
      title="删除角色？"
      :subject="pendingDelete?.name || ''"
      description="删除后，该角色及其权限配置无法恢复；请先确认没有账号仍依赖此角色。"
      :loading="deleting"
      @confirm="confirmDelete"
      @cancel="pendingDelete = null"
    />
  </div>
</template>

<script setup>
import { computed, h, onMounted, reactive, ref, watch } from 'vue'
import {
  NAlert, NButton, NCheckbox, NDataTable, NForm, NFormItem, NIcon, NInput,
  NRadio, NRadioGroup, NSelect, NSwitch, NTabPane, NTabs, NTag, useMessage,
} from 'naive-ui'
import {
  AddOutline, ChevronDownOutline, InformationCircleOutline, ShieldCheckmarkOutline, WarningOutline,
} from '@vicons/ionicons5'
import { getRoles, createRole, updateRole, deleteRole, getPermissionCatalog } from '@/api/roles'
import { getKnowledgeBases } from '@/api/knowledge'
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

const ACCESS_ALL_PERMISSION = 'kb:access_all'
// 旧后端尚未返回 superadmin_only 元数据时的兼容兜底；新目录是限制展示的唯一来源。
const FALLBACK_SUPERADMIN_ONLY_DELEGATIONS = new Set([
  'kb:create', 'kb:update', 'kb:delete',
  'settings:write', 'user:manage', 'role:manage',
])
const SCOPE_REQUIRED_CAPABILITIES = new Set([
  'search:use',
  'kb:read', 'kb:create', 'kb:update', 'kb:delete',
  'doc:read', 'doc:create', 'doc:update', 'doc:delete',
])
const MENU_PERMISSION_BY_MODULE = {
  chat: 'menu:chat',
  knowledge: 'menu:knowledge',
  documents: 'menu:documents',
  search_test: 'menu:search_test',
  intent_routing: 'menu:intent_routing',
  settings: 'menu:settings',
  users: 'menu:users',
  roles: 'menu:roles',
  audit_logs: 'menu:login_logs',
}

const PERMISSION_LABELS = {
  'chat:use': '发起智能问答',
  'search:use': '执行检索测试',
  'kb:read': '查看知识库',
  'kb:create': '新增知识库',
  'kb:update': '修改知识库',
  'kb:delete': '删除知识库',
  'doc:read': '查看文档',
  'doc:create': '新增文档',
  'doc:update': '修改文档',
  'doc:delete': '删除文档',
  'settings:read': '查看系统设置',
  'settings:write': '修改系统设置',
  'intent:read': '查看智能路由',
  'intent:manage': '管理智能路由',
  'user:manage': '管理用户账号',
  'role:manage': '管理角色权限',
  'log:read': '查看审计日志',
}

const PERMISSION_DESCRIPTIONS = {
  'chat:use': '允许用户在问答工作台发起会话。',
  'search:use': '允许使用检索测试验证召回结果。',
  'kb:read': '允许查看当前范围内的知识库及其元数据。',
  'kb:create': '允许创建新的知识库；该操作需要全部知识库范围。',
  'kb:update': '允许修改当前范围内知识库的名称、描述和标识。',
  'kb:delete': '允许删除当前范围内且已清空文档的知识库。',
  'doc:read': '允许查看当前范围内的文档内容。',
  'doc:create': '允许向当前范围内的知识库上传图片、文件或手动新增文档。',
  'doc:update': '允许编辑当前范围内文档的内容、标签和启停状态。',
  'doc:delete': '允许删除当前范围内的单个文档或批量删除文档。',
  'settings:read': '允许查看模型、检索和站点配置。',
  'settings:write': '允许修改系统级模型与站点配置。',
  'intent:read': '允许查看智能路由策略和运行日志。',
  'intent:manage': '允许修改路由策略、意图分类和路由反馈。',
  'user:manage': '允许创建、修改、启停和删除用户账号。',
  'role:manage': '允许创建、编辑和删除角色及其授权。',
  'log:read': '允许查看登录与操作审计记录。',
}

// 兼容旧目录：新合同未返回 groups 时，仍保持“按业务域 → 模块 → 能力”的树结构。
const FALLBACK_GROUPS = [
  {
    key: 'workspace',
    label: '问答工作台',
    description: '面向日常提问和检索验证的能力。',
    modules: [
      {
        key: 'chat', label: '智能问答', description: '提供受知识库范围约束的问答能力。',
        permissions: ['chat:use'],
      },
      {
        key: 'search_test', label: '检索测试', description: '用于验证不同检索策略和召回结果。',
        permissions: ['search:use'],
      },
    ],
  },
  {
    key: 'knowledge',
    label: '知识运营',
    description: '管理知识库、文档和智能路由。',
    modules: [
      { key: 'knowledge', label: '知识库管理', description: '分别配置知识库的查看、新增、修改和删除能力。', permissions: ['kb:read', 'kb:create', 'kb:update', 'kb:delete'] },
      { key: 'documents', label: '文档管理', description: '分别配置文档的查看、新增、修改和删除能力。', permissions: ['doc:read', 'doc:create', 'doc:update', 'doc:delete'] },
      { key: 'intent_routing', label: '智能路由', description: '查看或维护问答路由策略。', permissions: ['intent:read', 'intent:manage'] },
    ],
  },
  {
    key: 'system',
    label: '系统管理',
    description: '涉及账号、授权、系统配置与审计。',
    modules: [
      { key: 'settings', label: '系统设置', description: '查看或修改系统配置。', permissions: ['settings:read', 'settings:write'] },
      { key: 'users', label: '用户管理', description: '维护账号及其启用状态。', permissions: ['user:manage'] },
      { key: 'roles', label: '角色管理', description: '维护角色、能力和范围。', permissions: ['role:manage'] },
      { key: 'audit_logs', label: '审计日志', description: '查看登录和操作审计记录。', permissions: ['log:read'] },
    ],
  },
]

// 结构化目录由后端提供；仅在旧部署只返回权限 key 时补充最小依赖，
// 确保前端勾选写操作时同步补齐其读取能力。
const FALLBACK_PERMISSION_REQUIREMENTS = {
  'search:use': ['kb:read'],
  'kb:create': ['kb:read'],
  'kb:update': ['kb:read'],
  'kb:delete': ['kb:read'],
  'doc:read': ['kb:read'],
  'doc:create': ['doc:read', 'kb:read'],
  'doc:update': ['doc:read', 'kb:read'],
  'doc:delete': ['doc:read', 'kb:read'],
}
const FALLBACK_PERMISSION_RISKS = {
  'chat:use': 'medium',
  'search:use': 'medium',
  'kb:read': 'medium',
  'kb:create': 'high',
  'kb:update': 'high',
  'kb:delete': 'high',
  'doc:read': 'medium',
  'doc:create': 'high',
  'doc:update': 'high',
  'doc:delete': 'high',
  'settings:read': 'medium',
  'settings:write': 'critical',
  'intent:read': 'medium',
  'intent:manage': 'high',
  'user:manage': 'critical',
  'role:manage': 'critical',
  'log:read': 'high',
}

const roles = ref([])
const allPermissions = ref([])
const catalogCapabilities = ref([])
const catalogGroups = ref([])
const templates = ref([])
const catalogMenus = ref([])
const catalogScopeModes = ref([])
const kbs = ref([])
const loading = ref(false)
const saving = ref(false)

const showModal = ref(false)
const editingId = ref(null)
const editingIsSystem = ref(false)
const initialGrantState = ref({ permissions: [], scope_mode: 'none', is_assignable: true })
const activeTab = ref('capabilities')
const selectedTemplateKey = ref(null)
const expandedModuleKeys = ref([])
const form = ref(newRoleForm())

const showGrantConfirm = ref(false)
const pendingRolePayload = ref(null)
const pendingGrantChanges = ref([])

const showDeleteConfirm = ref(false)
const pendingDelete = ref(null)
const deleting = ref(false)

const pagination = reactive({
  page: 1,
  pageSize: 10,
  showSizePicker: true,
  pageSizes: [10, 20, 30, 50],
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (page) => { pagination.page = page },
  onUpdatePageSize: (pageSize) => { pagination.pageSize = pageSize; pagination.page = 1 },
})

watch(() => roles.value.length, () => {
  const max = Math.max(1, Math.ceil(roles.value.length / pagination.pageSize))
  if (pagination.page > max) pagination.page = max
})

const kbOptions = computed(() => kbs.value.map(kb => ({ label: kb.name, value: kb.id })))
const isSuperadmin = computed(() => !!authStore.user?.is_superadmin)
const editingIsReservedSuperadmin = computed(() => Boolean(editingId.value && form.value.code === 'superadmin'))
const superadminOnlyCapabilityKeys = computed(() => {
  const definitions = catalogCapabilities.value.filter(item => item && typeof item === 'object')
  const hasDirectoryMetadata = definitions.some(item => Object.prototype.hasOwnProperty.call(item, 'superadmin_only'))
  const declared = definitions
    .filter(item => item.superadmin_only === true)
    .map(item => String(item.key))
  return new Set(hasDirectoryMetadata ? declared : FALLBACK_SUPERADMIN_ONLY_DELEGATIONS)
})
const allScopeSuperadminOnly = computed(() => {
  const definition = catalogScopeModes.value.find(item => item?.key === 'all')
  return definition?.superadmin_only !== false
})
const templateOptions = computed(() => templates.value.map(template => ({
  label: template.label,
  value: template.key,
  description: template.description,
  disabled: !isSuperadmin.value && (
    template.is_assignable === false
    || (template.scope_mode === 'all' && allScopeSuperadminOnly.value)
    || (template.permissions || []).some(key => superadminOnlyCapabilityKeys.value.has(key))
  ),
})))

const capabilityMetadataByKey = computed(() => new Map(
  catalogCapabilities.value
    .filter(item => item && typeof item === 'object' && item.key)
    .map(item => [String(item.key), item])
))

const capabilityGroups = computed(() => {
  const knownKeys = new Set(allPermissions.value)
  const sourceGroups = catalogGroups.value.length ? catalogGroups.value : FALLBACK_GROUPS
  const used = new Set()

  const groups = sourceGroups.map(group => {
    const modules = (group.modules || []).map(module => {
      const permissions = (module.permissions || [])
        .map(raw => normalizeAbility(raw))
        .filter(Boolean)
        .filter(ability => !ability.key.startsWith('menu:') && ability.key !== ACCESS_ALL_PERMISSION)
        .filter(ability => !knownKeys.size || knownKeys.has(ability.key))

      permissions.forEach(ability => used.add(ability.key))
      if (!permissions.length) return null
      return {
        key: module.key,
        label: module.label || module.key,
        description: module.description || '',
        menuPermission: resolveModuleMenuPermission(module),
        permissions,
      }
    }).filter(Boolean)

    if (!modules.length) return null
    return {
      key: group.key,
      label: group.label || group.key,
      description: group.description || '',
      modules,
    }
  }).filter(Boolean)

  const remaining = allPermissions.value
    .filter(key => !key.startsWith('menu:') && key !== ACCESS_ALL_PERMISSION && !used.has(key))
    .map(key => normalizeAbility(key))
    .filter(Boolean)
  if (remaining.length) {
    groups.push({
      key: 'other',
      label: '其他能力',
      description: '尚未归入业务模块的已定义能力。',
      modules: [{ key: 'other', label: '其他能力', description: '按需授权的补充能力。', menuPermission: null, permissions: remaining }],
    })
  }
  return groups
})

const abilityIndex = computed(() => {
  const index = new Map()
  capabilityGroups.value.forEach(group => group.modules.forEach(module => {
    module.permissions.forEach(ability => index.set(ability.key, ability))
  }))
  return index
})

const allModuleKeys = computed(() => capabilityGroups.value.flatMap(group => group.modules.map(module => module.key)))
const selectedCapabilityCount = computed(() => form.value.permissions.filter(key => abilityIndex.value.has(key)).length)
const legacyCapabilities = computed(() => form.value.permissions.filter(key => !key.startsWith('menu:') && !abilityIndex.value.has(key)))
const selectedKnowledgeBases = computed(() => kbs.value.filter(kb => form.value.kb_ids.includes(kb.id)))

const activeModules = computed(() => capabilityGroups.value
  .flatMap(group => group.modules)
  .filter(module => moduleSelectedAbilities(module).length)
)

const highRiskAbilities = computed(() => {
  const selected = new Set(form.value.permissions)
  return [...abilityIndex.value.values()].filter(ability => selected.has(ability.key) && isHighRisk(ability))
})

const hasKnowledgeDataCapability = computed(() => {
  const selected = new Set(form.value.permissions)
  return [
    'chat:use', 'search:use',
    'kb:read', 'kb:create', 'kb:update', 'kb:delete',
    'doc:read', 'doc:create', 'doc:update', 'doc:delete',
  ].some(key => selected.has(key))
})
const hasScopeRequiredCapability = computed(() => form.value.permissions.some(key => SCOPE_REQUIRED_CAPABILITIES.has(key)))
const hasKnowledgeCreateCapability = computed(() => form.value.permissions.includes('kb:create'))

const scopeLabel = computed(() => ({ none: '不授予范围', selected: '指定知识库', all: '全部知识库' })[form.value.scope_mode] || '不授予范围')
const scopeTagType = computed(() => ({ none: 'default', selected: 'info', all: 'warning' })[form.value.scope_mode] || 'default')
const scopeNarrative = computed(() => {
  if (form.value.scope_mode === 'all') return '可访问当前及未来新建的全部知识库。'
  if (form.value.scope_mode === 'selected') {
    if (!selectedKnowledgeBases.value.length) return '范围设为指定知识库，但尚未选择对象。'
    return `仅可访问 ${selectedKnowledgeBases.value.map(kb => kb.name).join('、')}。`
  }
  return '不授予任何知识库数据访问范围。'
})
const naturalLanguageSummary = computed(() => {
  if (!activeModules.value.length) return `尚未授予可执行能力；${scopeNarrative.value}`
  return `可使用 ${activeModules.value.map(module => module.label).join('、')}，${scopeNarrative.value}`
})
const pendingGrantRoleName = computed(() => (
  pendingRolePayload.value?.name || form.value.name.trim() || '未命名角色'
))

const columns = [
  { title: '名称', key: 'name', minWidth: 140, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  { title: '编码', key: 'code', width: 150, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true }, render: row => row.code || '—' },
  { title: '说明', key: 'description', minWidth: 190, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true }, render: row => row.description || '—' },
  {
    title: '数据范围', key: 'scope_mode', width: 118, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', bordered: false, type: scopeTagTypeFor(row) }, () => scopeLabelFor(row)),
  },
  {
    title: '可分配', key: 'is_assignable', width: 92, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { size: 'small', bordered: false, type: row.is_assignable === false ? 'default' : 'success' }, () => row.is_assignable === false ? '仅保留' : '可分配'),
  },
  { title: '能力数', key: 'perm_count', width: 88, align: 'center', titleAlign: 'center', render: row => (row.permissions || []).filter(key => !String(key).startsWith('menu:') && key !== ACCESS_ALL_PERMISSION).length },
  {
    title: '类型', key: 'is_system', width: 90, align: 'center', titleAlign: 'center',
    render: row => h(NTag, { type: row.is_system ? 'info' : 'default', size: 'small', bordered: false }, () => row.is_system ? '系统' : '自定义'),
  },
  {
    title: '操作', key: 'actions', width: 136, align: 'center', titleAlign: 'center',
    render: row => h(RowActions, { label: `角色 ${row.name} 操作` }, {
      default: () => [
        h(NButton, {
          text: true,
          type: 'primary',
          size: 'small',
          disabled: !canManageRoleRow(row),
          title: canManageRoleRow(row) ? '编辑角色' : '该角色只能由超级管理员维护',
          onClick: () => openEdit(row),
        }, () => '编辑'),
        h(NButton, {
          text: true,
          type: 'error',
          size: 'small',
          disabled: row.is_system || !canManageRoleRow(row),
          title: row.is_system ? '系统角色不可删除' : (canManageRoleRow(row) ? '删除角色' : '该角色只能由超级管理员维护'),
          onClick: () => openDelete(row),
        }, () => '删除'),
      ],
    }),
  },
]

onMounted(async () => {
  await Promise.all([loadRoles(), loadCatalog(), loadKbs()])
})

function newRoleForm() {
  return {
    name: '',
    code: '',
    description: '',
    permissions: [],
    scope_mode: 'none',
    kb_ids: [],
    is_assignable: true,
  }
}

function normalizeAbility(raw) {
  const key = typeof raw === 'string' ? raw : raw?.key
  if (!key) return null
  const value = String(key)
  const definition = typeof raw === 'object' ? raw : capabilityMetadataByKey.value.get(value)
  const requires = Array.isArray(definition?.requires)
    ? definition.requires.map(item => String(item)).filter(Boolean)
    : (FALLBACK_PERMISSION_REQUIREMENTS[value] || [])
  return {
    key: value,
    label: definition?.label || PERMISSION_LABELS[value] || value,
    description: definition?.description || PERMISSION_DESCRIPTIONS[value] || '',
    risk: definition?.risk || FALLBACK_PERMISSION_RISKS[value] || 'normal',
    requires,
  }
}

function moduleKey(value) {
  return String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_')
}

function resolveModuleMenuPermission(module) {
  const explicit = module.menu_permission || module.menuPermission || module.menu
  if (typeof explicit === 'string' && explicit.startsWith('menu:')) return explicit
  const key = moduleKey(module.key)
  const fromCatalog = catalogMenus.value.find(item => moduleKey(item.key || item.route) === key)?.permission
  return fromCatalog || MENU_PERMISSION_BY_MODULE[key] || null
}

function isHighRisk(ability) {
  return ['high', 'critical', 'danger'].includes(String(ability?.risk || '').toLowerCase())
}

function isDelegationRestricted(key) {
  return !isSuperadmin.value && superadminOnlyCapabilityKeys.value.has(key)
}

function canManageRoleRow(role) {
  if (isSuperadmin.value) return true
  if (
    role?.is_system
    || role?.is_assignable === false
    || (scopeModeFor(role) === 'all' && allScopeSuperadminOnly.value)
  ) return false
  return !(role?.permissions || []).some(key => superadminOnlyCapabilityKeys.value.has(key))
}

function dependencyLabels(keys) {
  return keys
    .filter(key => !String(key).startsWith('menu:') && key !== ACCESS_ALL_PERMISSION)
    .map(key => abilityIndex.value.get(key)?.label || PERMISSION_LABELS[key] || key)
}

function hasCapability(key) {
  return form.value.permissions.includes(key)
}

function addCapabilityWithDependencies(key, selected, visited = new Set()) {
  if (!key || visited.has(key)) return
  visited.add(key)
  if (!String(key).startsWith('menu:') && key !== ACCESS_ALL_PERMISSION) selected.add(key)
  const ability = abilityIndex.value.get(key)
  ;(ability?.requires || []).forEach(required => addCapabilityWithDependencies(required, selected, visited))
}

function setCapability(key, checked) {
  if (saving.value) return
  const selected = new Set(form.value.permissions.filter(item => !String(item).startsWith('menu:') && item !== ACCESS_ALL_PERMISSION))
  if (checked) {
    addCapabilityWithDependencies(key, selected)
  } else {
    const removed = new Set([key])
    selected.delete(key)
    let changed = true
    while (changed) {
      changed = false
      abilityIndex.value.forEach((ability, candidate) => {
        if (!selected.has(candidate) || !ability.requires?.some(required => removed.has(required))) return
        selected.delete(candidate)
        removed.add(candidate)
        changed = true
      })
    }
  }
  form.value.permissions = [...selected]
}

function normalizeCapabilities(keys) {
  const selected = new Set()
  keys
    .filter(key => !String(key).startsWith('menu:') && key !== ACCESS_ALL_PERMISSION)
    .forEach(key => addCapabilityWithDependencies(key, selected))
  return [...selected]
}

function moduleSelectedAbilities(module) {
  return module.permissions.filter(ability => hasCapability(ability.key))
}

function moduleSelectedCount(module) {
  return moduleSelectedAbilities(module).length
}

function groupSelectedCount(group) {
  return group.modules.reduce((count, module) => count + moduleSelectedCount(module), 0)
}

function isModuleExpanded(key) {
  return expandedModuleKeys.value.includes(key)
}

function toggleModule(key) {
  expandedModuleKeys.value = isModuleExpanded(key)
    ? expandedModuleKeys.value.filter(item => item !== key)
    : [...expandedModuleKeys.value, key]
}

function expandAllModules() {
  expandedModuleKeys.value = [...allModuleKeys.value]
}

function collapseAllModules() {
  expandedModuleKeys.value = []
}

function clearCapabilities() {
  form.value.permissions = []
}

function resetEditorState() {
  activeTab.value = 'capabilities'
  selectedTemplateKey.value = null
  expandedModuleKeys.value = [...allModuleKeys.value]
}

function normalizeScopeMode(value) {
  return ['none', 'selected', 'all'].includes(value) ? value : 'none'
}

function changeScopeMode(value) {
  const mode = normalizeScopeMode(value)
  form.value.scope_mode = mode
  if (mode !== 'selected') form.value.kb_ids = []
}

function scopeModeFor(role) {
  if (['none', 'selected', 'all'].includes(role?.scope_mode)) return role.scope_mode
  if ((role?.permissions || []).includes(ACCESS_ALL_PERMISSION)) return 'all'
  return role?.kb_ids?.length ? 'selected' : 'none'
}

function scopeLabelFor(role) {
  return ({ none: '无范围', selected: '指定知识库', all: '全部知识库' })[scopeModeFor(role)] || '无范围'
}

function scopeTagTypeFor(role) {
  return ({ none: 'default', selected: 'info', all: 'warning' })[scopeModeFor(role)] || 'default'
}

function payloadPermissions() {
  // 菜单与 kb:access_all 都由后端从 capabilities / scope_mode 派生；
  // 只提交目录中可分配的真实能力，避免旧 key 被误写回角色。
  return [...new Set(form.value.permissions.filter(key => abilityIndex.value.has(key)))]
}

function applyTemplate(templateKey) {
  if (!templateKey) return
  const template = templates.value.find(item => item.key === templateKey)
  if (!template) return
  form.value.permissions = normalizeCapabilities(template.permissions || [])
  changeScopeMode(template.scope_mode || 'none')
  if (isSuperadmin.value && typeof template.is_assignable === 'boolean') form.value.is_assignable = template.is_assignable
  activeTab.value = 'capabilities'
}

async function loadRoles() {
  loading.value = true
  try {
    roles.value = await getRoles()
  } catch (error) {
    msg.error(error?.response?.data?.detail || '加载角色列表失败')
  } finally {
    loading.value = false
  }
}

async function loadCatalog() {
  try {
    const catalog = await getPermissionCatalog()
    // v2 directory lives under `catalog`; retain top-level support while older
    // deployments are being upgraded.
    const directory = catalog?.catalog && typeof catalog.catalog === 'object' ? catalog.catalog : (catalog || {})
    const capabilityEntries = catalog?.capabilities ?? directory.capabilities ?? []
    const permissionEntries = catalog?.permissions ?? directory.permissions ?? capabilityEntries
    catalogCapabilities.value = Array.isArray(capabilityEntries) ? capabilityEntries : []
    allPermissions.value = [...new Set([...permissionEntries, ...catalogCapabilities.value]
      .map(item => typeof item === 'string' ? item : item?.key)
      .filter(Boolean))]
    catalogGroups.value = Array.isArray(catalog?.groups)
      ? catalog.groups
      : (Array.isArray(directory.groups) ? directory.groups : [])
    templates.value = Array.isArray(catalog?.templates)
      ? catalog.templates
      : (Array.isArray(directory.templates) ? directory.templates : [])
    catalogMenus.value = Array.isArray(catalog?.menus)
      ? catalog.menus
      : (Array.isArray(directory.menus) ? directory.menus : [])
    catalogScopeModes.value = Array.isArray(catalog?.scope_modes)
      ? catalog.scope_modes
      : (Array.isArray(directory.scope_modes) ? directory.scope_modes : [])
  } catch (error) {
    msg.error(error?.response?.data?.detail || '加载权限目录失败')
  }
}

async function loadKbs() {
  try {
    kbs.value = await getKnowledgeBases()
  } catch (error) {
    kbs.value = []
    msg.warning(error?.response?.data?.detail || '未能加载知识库范围选项')
  }
}

function openCreate() {
  editingId.value = null
  editingIsSystem.value = false
  form.value = newRoleForm()
  initialGrantState.value = { permissions: [], scope_mode: 'none', is_assignable: true }
  resetGrantConfirmation()
  resetEditorState()
  showModal.value = true
}

function openEdit(row) {
  const rawPermissions = [...(row.permissions || [])]
  const scopeMode = scopeModeFor(row)
  const normalizedPermissions = normalizeCapabilities(rawPermissions)
  editingId.value = row.id
  editingIsSystem.value = !!row.is_system
  form.value = {
    name: row.name || '',
    code: row.code || '',
    description: row.description || '',
    permissions: normalizedPermissions,
    scope_mode: scopeMode,
    kb_ids: scopeMode === 'selected' ? [...(row.kb_ids || [])] : [],
    is_assignable: row.is_assignable !== false,
  }
  initialGrantState.value = {
    permissions: [...normalizedPermissions],
    scope_mode: scopeMode,
    is_assignable: row.is_assignable !== false,
  }
  resetGrantConfirmation()
  resetEditorState()
  showModal.value = true
}

function closeEditor() {
  if (saving.value) return
  resetGrantConfirmation()
  showModal.value = false
}

async function handleSave() {
  if (!form.value.name.trim()) {
    msg.warning('请输入角色名称')
    return
  }
  if (hasKnowledgeCreateCapability.value && form.value.scope_mode !== 'all') {
    activeTab.value = 'scope'
    msg.warning('新增知识库需要全部知识库范围')
    return
  }
  if (form.value.scope_mode === 'selected' && !form.value.kb_ids.length) {
    activeTab.value = 'scope'
    msg.warning('请选择至少一个可访问知识库，或改为“不授予知识库访问”')
    return
  }
  if (form.value.scope_mode === 'none' && hasScopeRequiredCapability.value) {
    activeTab.value = 'scope'
    msg.warning('已选择需要知识库数据的能力，请先配置知识库范围')
    return
  }

  const payload = buildRolePayload()
  const grantChanges = findNewHighImpactGrants(payload)
  if (grantChanges.length) {
    pendingRolePayload.value = payload
    pendingGrantChanges.value = grantChanges
    showGrantConfirm.value = true
    return
  }

  await persistRole(payload)
}

function buildRolePayload() {
  const payload = {
    name: form.value.name.trim(),
    description: form.value.description.trim() || null,
    permissions: payloadPermissions(),
    scope_mode: form.value.scope_mode,
    kb_ids: form.value.scope_mode === 'selected' ? [...new Set(form.value.kb_ids)] : [],
  }
  if (isSuperadmin.value) {
    payload.is_assignable = !!form.value.is_assignable
    if (form.value.code.trim()) payload.code = form.value.code.trim()
  }
  return payload
}

function findNewHighImpactGrants(payload) {
  const originalPermissions = new Set(initialGrantState.value.permissions || [])
  const changes = payload.permissions
    .filter(key => superadminOnlyCapabilityKeys.value.has(key) && !originalPermissions.has(key))
    .map(key => ({
      key,
      label: abilityIndex.value.get(key)?.label || PERMISSION_LABELS[key] || key,
      description: abilityIndex.value.get(key)?.description || PERMISSION_DESCRIPTIONS[key] || '该能力会扩大角色可执行的操作范围。',
    }))

  if (payload.scope_mode === 'all' && initialGrantState.value.scope_mode !== 'all') {
    changes.push({
      key: 'scope_mode:all',
      label: '全部知识库范围',
      description: '可访问当前以及未来新建的全部知识库，数据边界会随平台内容持续扩大。',
    })
  }
  if (
    payload.is_assignable === true
    && initialGrantState.value.is_assignable === false
    && (
      (payload.scope_mode === 'all' && allScopeSuperadminOnly.value)
      || payload.permissions.some(key => superadminOnlyCapabilityKeys.value.has(key))
    )
  ) {
    changes.push({
      key: 'is_assignable:enabled',
      label: '允许分配高权限角色',
      description: '该角色将出现在用户管理的角色选项中，可直接影响被分配账号的实际权限。',
    })
  }
  return changes
}

function resetGrantConfirmation() {
  showGrantConfirm.value = false
  pendingRolePayload.value = null
  pendingGrantChanges.value = []
}

function cancelGrantConfirm() {
  if (saving.value) return
  resetGrantConfirmation()
}

async function confirmGrantSave() {
  const payload = pendingRolePayload.value
  if (!payload) return
  await persistRole(payload)
}

async function persistRole(payload) {
  saving.value = true
  try {
    if (editingId.value) {
      await updateRole(editingId.value, payload)
      msg.success('角色已更新')
    } else {
      await createRole(payload)
      msg.success('角色已创建')
    }
    resetGrantConfirmation()
    showModal.value = false
    await loadRoles()
  } catch (error) {
    msg.error(error?.response?.data?.detail || '保存失败，请重试')
  } finally {
    saving.value = false
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
    await deleteRole(row.id)
    roles.value = roles.value.filter(role => role.id !== row.id)
    msg.success('角色已删除')
    showDeleteConfirm.value = false
    pendingDelete.value = null
  } catch (error) {
    msg.error(error?.response?.data?.detail || '删除失败，请重试')
  } finally {
    deleting.value = false
  }
}
</script>

<style scoped>
.role-form-scroll {
  max-height: calc(80vh - 158px);
  padding-right: 4px;
  overflow-y: auto;
}

.role-editor__overview,
.role-template-picker,
.permission-tree-toolbar,
.scope-heading,
.role-summary__hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.role-editor__overview {
  margin-bottom: 18px;
  padding: 14px 16px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: linear-gradient(135deg, var(--ui-primary-subtle), var(--ui-surface));
}

.role-editor__eyebrow {
  color: var(--ui-primary);
  font-size: 10px;
  font-weight: 750;
  letter-spacing: .12em;
}

.role-editor__hint,
.role-template-picker p,
.permission-tree-toolbar p,
.permission-tree-intro,
.scope-heading p,
.scope-selected-panel__head p,
.permission-tree-group__header p,
.permission-module__copy > span:not(.permission-module__title-row),
.capability-option__description,
.capability-option__requires {
  margin: 4px 0 0;
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.55;
}

.role-template-picker {
  margin: 0 0 18px;
  padding: 14px 16px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface-muted);
}

.role-template-picker__copy { min-width: 0; }
.role-template-picker__title { color: var(--ui-text); font-size: 13px; font-weight: 650; }
.role-template-picker__select { width: min(100%, 330px); flex: 0 1 330px; }

.role-assignable-control {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 36px;
}

.role-assignable-control > span {
  color: var(--ui-text-secondary);
  font-size: 12px;
  line-height: 1.45;
}

.role-editor-tabs { margin-top: 4px; }
.role-tab-panel { padding: 16px 1px 2px; }

.permission-tree-toolbar {
  padding: 12px 14px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface-muted);
}

.permission-tree-toolbar__title,
.scope-heading h3,
.role-summary__hero h3 {
  margin: 0;
  color: var(--ui-text);
  font-size: 14px;
  font-weight: 650;
}

.permission-tree-intro {
  margin: 12px 2px;
}

.role-permission-tree {
  display: grid;
  gap: 14px;
}

.permission-tree-group {
  padding: 15px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
}

.permission-tree-group--legacy { border-style: dashed; background: var(--ui-surface-muted); }

.permission-tree-group__header {
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr) auto;
  align-items: start;
  gap: 10px;
}

.permission-tree-group__dot,
.permission-module__dot {
  display: block;
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--ui-primary);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--ui-primary) 13%, transparent);
}

.permission-tree-group__dot { margin-top: 6px; }

.permission-tree-group__header h4 {
  margin: 0;
  color: var(--ui-text);
  font-size: 13px;
  font-weight: 650;
}

.permission-tree__modules {
  position: relative;
  display: grid;
  gap: 8px;
  margin-top: 14px;
  padding-left: 21px;
}

.permission-tree__modules::before {
  position: absolute;
  top: 1px;
  bottom: 17px;
  left: 4px;
  width: 1px;
  content: '';
  background: var(--ui-divider);
}

.permission-module { position: relative; }

.permission-module::before {
  position: absolute;
  top: 20px;
  left: -17px;
  width: 13px;
  height: 1px;
  content: '';
  background: var(--ui-divider);
}

.permission-module__head {
  display: grid;
  width: 100%;
  grid-template-columns: 10px minmax(0, 1fr) 18px;
  align-items: start;
  gap: 9px;
  padding: 9px 10px;
  color: inherit;
  text-align: left;
  border: 1px solid transparent;
  border-radius: var(--ui-radius-control);
  background: transparent;
  cursor: pointer;
  font: inherit;
  transition: background-color .16s ease, border-color .16s ease;
}

.permission-module__head:hover {
  border-color: var(--ui-border);
  background: var(--ui-surface-hover);
}

.permission-module__head:focus-visible {
  outline: 0;
  border-color: var(--ui-border-focus);
  box-shadow: var(--ui-focus-ring);
}

.permission-module__dot {
  width: 6px;
  height: 6px;
  margin-top: 6px;
  background: var(--ui-text-tertiary);
  box-shadow: none;
}

.permission-module__copy { display: grid; min-width: 0; }

.permission-module__title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  color: var(--ui-text);
  font-size: 13px;
  line-height: 1.4;
}

.permission-module__count {
  flex: 0 0 auto;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.permission-module__entry-hint {
  color: var(--ui-primary) !important;
  font-size: 11px !important;
}

.permission-module__chevron {
  margin-top: 3px;
  color: var(--ui-icon);
  transition: transform .16s ease;
}

.permission-module__chevron.is-open { transform: rotate(180deg); }

.permission-module__abilities {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  padding: 8px 0 2px 19px;
}

.capability-option {
  min-width: 0;
  padding: 9px 10px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-control);
  background: var(--ui-surface);
  transition: border-color .16s ease, background-color .16s ease, box-shadow .16s ease;
}

.capability-option:hover { border-color: var(--ui-border-strong); background: var(--ui-surface-hover); }

.capability-option--selected {
  border-color: var(--ui-border-focus);
  background: color-mix(in srgb, var(--ui-primary-subtle) 72%, var(--ui-surface));
}

.capability-option--risk { border-left: 3px solid var(--ui-danger); }

.capability-option :deep(.n-checkbox) {
  display: flex;
  width: 100%;
  align-items: flex-start;
}

.capability-option :deep(.n-checkbox-box) { margin-top: 3px; }
.capability-option__content { display: grid; min-width: 0; }
.capability-option__title-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.capability-option__title { color: var(--ui-text); font-size: 13px; font-weight: 600; }
.capability-option__requires { color: var(--ui-text-tertiary); font-size: 11px; }

.permission-module__empty,
.summary-empty {
  padding: 11px 12px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
  line-height: 1.55;
}

.legacy-capability-list,
.summary-kb-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.scope-heading { margin-bottom: 16px; }
.scope-options { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; width: 100%; }

.scope-option {
  display: flex;
  min-width: 0;
  min-height: 108px;
  margin: 0;
  padding: 13px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
  transition: border-color .16s ease, background-color .16s ease, box-shadow .16s ease;
}

.scope-option:hover { border-color: var(--ui-border-strong); background: var(--ui-surface-hover); }
.scope-option--active { border-color: var(--ui-border-focus); background: var(--ui-primary-subtle); box-shadow: var(--ui-focus-ring); }
.scope-option :deep(.n-radio) { align-items: flex-start; width: 100%; }
.scope-option :deep(.n-radio__dot) { margin-top: 3px; }
.scope-option__copy { display: grid; gap: 5px; min-width: 0; }
.scope-option__copy strong { color: var(--ui-text); font-size: 13px; font-weight: 650; }
.scope-option__copy span { color: var(--ui-text-secondary); font-size: 12px; line-height: 1.55; }

.scope-selected-panel {
  margin-top: 14px;
  padding: 15px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface-muted);
}

.scope-selected-panel__head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.scope-selected-panel__head h4 { margin: 0; color: var(--ui-text); font-size: 13px; font-weight: 650; }
.scope-selected-panel__head > span { color: var(--ui-primary); font-size: 12px; font-weight: 650; white-space: nowrap; }
.scope-alert { margin-top: 14px; }
.scope-consistency-hint { margin: 12px 2px 0; color: var(--ui-warning); font-size: 12px; line-height: 1.55; }

.role-summary { display: grid; gap: 14px; }
.role-summary__hero { padding: 16px; border: 1px solid var(--ui-border); border-radius: var(--ui-radius-card); background: var(--ui-surface-muted); }
.role-summary__hero-icon { display: grid; flex: 0 0 auto; width: 38px; height: 38px; place-items: center; color: var(--ui-primary); border-radius: 12px; background: var(--ui-primary-subtle); }
.role-summary__hero > div { min-width: 0; }
.role-summary__hero p { margin: 5px 0 0; color: var(--ui-text-secondary); font-size: 13px; line-height: 1.6; }
.role-summary__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }

.summary-surface {
  min-width: 0;
  padding: 15px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
}

.summary-surface h4 { margin: 0; color: var(--ui-text); font-size: 13px; font-weight: 650; }
.summary-list { display: grid; gap: 10px; margin: 12px 0 0; padding: 0; list-style: none; }
.summary-list li { display: grid; gap: 3px; padding-bottom: 10px; border-bottom: 1px solid var(--ui-divider); }
.summary-list li:last-child { padding-bottom: 0; border-bottom: 0; }
.summary-list strong { color: var(--ui-text); font-size: 12px; }
.summary-list span,
.summary-scope-text { margin: 0; color: var(--ui-text-secondary); font-size: 12px; line-height: 1.55; }
.summary-scope-text { margin-top: 12px; }
.role-summary__risk { margin: 0; }
.role-summary__risk p { margin: 4px 0 0; line-height: 1.55; }
.role-summary__legacy { display: flex; align-items: center; gap: 7px; color: var(--ui-text-secondary); font-size: 12px; }

.grant-confirm {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}

.grant-confirm__icon {
  display: grid;
  flex: 0 0 auto;
  width: 44px;
  height: 44px;
  place-items: center;
  color: var(--ui-warning);
  border-radius: 14px;
  background: color-mix(in srgb, var(--ui-warning) 14%, var(--ui-surface));
}

.grant-confirm__copy { min-width: 0; flex: 1; }
.grant-confirm__lead { margin: 1px 0 0; color: var(--ui-text); font-size: 14px; line-height: 1.65; }
.grant-confirm__list { display: grid; gap: 8px; margin: 14px 0 0; padding: 0; list-style: none; }
.grant-confirm__list li { display: grid; gap: 3px; padding: 10px 12px; border: 1px solid var(--ui-border); border-radius: var(--ui-radius-control); background: var(--ui-surface-muted); }
.grant-confirm__list strong { color: var(--ui-text); font-size: 13px; font-weight: 650; }
.grant-confirm__list span { color: var(--ui-text-secondary); font-size: 12px; line-height: 1.55; }
.grant-confirm__notice { margin-top: 14px; }

@media (max-width: 767px) {
  .role-editor__overview,
  .role-template-picker,
  .permission-tree-toolbar,
  .scope-heading,
  .role-summary__hero { flex-direction: column; gap: 10px; }

  .role-template-picker__select { width: 100%; flex-basis: auto; }
  .permission-tree-toolbar > div:last-child { justify-content: flex-start; }
  .permission-tree-group__header { grid-template-columns: 10px minmax(0, 1fr); }
  .permission-tree-group__header :deep(.n-tag) { grid-column: 2; justify-self: start; }
  .permission-module__abilities,
  .scope-options,
  .role-summary__grid { grid-template-columns: 1fr; }
  .permission-module__abilities { padding-left: 0; }
  .scope-option { min-height: 0; }
}

@media (max-width: 639px) {
  .role-form-scroll { max-height: calc(82vh - 148px); }
  .role-tab-panel { padding-top: 13px; }
  .permission-tree-group,
  .scope-selected-panel,
  .summary-surface { padding: 13px; }
  .role-assignable-control { align-items: flex-start; flex-direction: column; gap: 6px; }
  .grant-confirm { flex-direction: column; gap: 10px; }
}
</style>
