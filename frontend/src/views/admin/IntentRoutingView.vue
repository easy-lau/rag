<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="space-y-6">
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
              <p class="mt-1 text-xs text-gray-400">模型只表达操作、会话关系和证据需求；回答模式、检索策略和执行授权由服务端确定性编译，并继续受权限校验约束。</p>
            </div>
            <div class="flex items-center gap-3">
              <n-tag :type="routingActive ? 'success' : 'default'" :bordered="false" round>
                {{ routingActive ? '路由已启用' : '路由未启用' }}
              </n-tag>
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
          <div class="grid grid-cols-1 gap-3 border-t border-gray-100 pt-4 text-xs dark:border-gray-700 sm:grid-cols-3">
            <div>
              <div class="text-gray-400">语义协议</div>
              <div class="mt-1 break-all font-mono text-gray-700 dark:text-gray-200">{{ config.route_schema_version || '未记录' }}</div>
            </div>
            <div>
              <div class="text-gray-400">任务合同</div>
              <div class="mt-1 break-all font-mono text-gray-700 dark:text-gray-200">{{ config.contract_schema_version || '未记录' }}</div>
            </div>
            <div>
              <div class="text-gray-400">Prompt 版本</div>
              <div class="mt-1 break-all font-mono text-gray-700 dark:text-gray-200">{{ config.prompt_version || '未记录' }}</div>
            </div>
          </div>
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

          <div class="grid grid-cols-1 gap-6">
          <!-- 在线测试 -->
          <SurfaceCard>
            <div class="flex items-start justify-between gap-3 mb-4">
              <div>
                <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-2">
                  <span class="w-2 h-2 rounded-full bg-orange-500 inline-block"></span>
                  在线测试
                </h3>
                <p class="mt-1 text-xs text-gray-400">模拟当前输入、最近会话和知识库选择状态；只执行路由，不会创建对话、读取知识库或执行真实检索。</p>
              </div>
              <n-button type="primary" :loading="testing" :disabled="!testQuery.trim()" @click="runTest">测试路由</n-button>
            </div>

            <n-form label-placement="top">
              <n-form-item label="当前问题">
                <n-input
                  v-model:value="testQuery" type="textarea" :rows="4"
                  :maxlength="12000" show-count
                  placeholder="例如：住宿标准呢？"
                  @keyup.ctrl.enter="runTest"
                  @keyup.meta.enter="runTest"
                />
                <template #feedback>按 Ctrl / Cmd + Enter 可快速测试。</template>
              </n-form-item>

              <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div class="text-sm font-medium text-gray-700 dark:text-gray-200">最近会话</div>
                  <div class="mt-0.5 text-xs text-gray-400">按真实顺序提供必要上下文，最多 6 条；留空消息不会发送。</div>
                </div>
                <n-button
                  secondary
                  size="small"
                  :disabled="testContextMessages.length >= MAX_TEST_CONTEXT_MESSAGES || testing"
                  @click="addTestContextMessage"
                >
                  添加消息
                </n-button>
              </div>
              <div
                v-if="!testContextMessages.length"
                class="mb-4 rounded-lg border border-dashed border-gray-200 px-4 py-3 text-xs text-gray-400 dark:border-gray-700"
              >
                当前按独立问题测试；如需验证“住宿呢”“补贴呢”等追问，请添加最近会话。
              </div>
              <div v-else class="mb-4 space-y-3">
                <div
                  v-for="(item, index) in testContextMessages"
                  :key="item.id"
                  class="grid grid-cols-1 items-start gap-2 rounded-lg border border-gray-100 p-3 dark:border-gray-700 sm:grid-cols-[120px_minmax(0,1fr)_auto]"
                >
                  <n-select
                    v-model:value="item.role"
                    :options="contextRoleOptions"
                    :disabled="testing"
                    :aria-label="`第 ${index + 1} 条消息角色`"
                  />
                  <n-input
                    v-model:value="item.content"
                    type="textarea"
                    :rows="2"
                    :maxlength="4000"
                    show-count
                    :disabled="testing"
                    :placeholder="item.role === 'assistant' ? '上一轮助手回答' : '上一轮用户问题'"
                    :aria-label="`第 ${index + 1} 条消息内容`"
                  />
                  <n-button
                    text
                    type="error"
                    :disabled="testing"
                    :aria-label="`删除第 ${index + 1} 条会话消息`"
                    @click="removeTestContextMessage(item.id)"
                  >
                    删除
                  </n-button>
                </div>
              </div>

              <n-form-item label="模拟已选择知识库数量">
                <n-input-number
                  v-model:value="testSelectedKbCount"
                  :min="0"
                  :max="100"
                  :step="1"
                  :precision="0"
                  :disabled="testing"
                  class="w-full sm:max-w-xs"
                />
                <template #feedback>仅提供数量作为路由上下文，不读取真实知识库名称、文档或权限范围。</template>
              </n-form-item>
            </n-form>

            <div v-if="testResult" class="mt-2 rounded-lg bg-gray-50 dark:bg-gray-700/40 border border-gray-100 dark:border-gray-700 p-4">
              <div class="flex flex-wrap items-center gap-2 mb-3">
                <span class="text-sm font-medium text-gray-700 dark:text-gray-200">路由结果</span>
                <n-tag type="info" size="small" :bordered="false">{{ operationLabel(semanticOperationFor(testResult)) }}</n-tag>
                <n-tag size="small" :type="readinessTagType(readinessFor(testResult))" :bordered="false">
                  {{ readinessLabel(readinessFor(testResult)) }}
                </n-tag>
                <n-tag size="small" :type="retrievalPolicyTagType(retrievalPolicyFor(testResult))" :bordered="false">
                  {{ retrievalPolicyLabel(retrievalPolicyFor(testResult)) }}
                </n-tag>
              </div>

              <section v-if="testRouteDecision" class="border-t border-gray-200 pt-3 dark:border-gray-600">
                <h4 class="text-xs font-semibold text-gray-600 dark:text-gray-300">模型语义决定</h4>
                <div class="mt-3 grid grid-cols-1 gap-x-4 gap-y-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                  <div><div class="text-gray-400">语义意图</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ operationLabel(semanticOperationFor(testResult)) }}</div></div>
                  <div><div class="text-gray-400">会话关系</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ relationLabel(testRouteDecision.relation) }}</div></div>
                  <div><div class="text-gray-400">准备状态</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ readinessLabel(testRouteDecision.readiness) }}</div></div>
                  <div><div class="text-gray-400">证据范围</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ evidenceScopeLabel(testRouteDecision.evidence_scope) }}</div></div>
                  <div><div class="text-gray-400">查询处理</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ queryResolutionLabel(testRouteDecision.query_resolution?.mode || testRouteDecision.query_resolution_mode) }}</div></div>
                  <div><div class="text-gray-400">绑定上下文</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ testRouteDecision.query_resolution?.context_turn_keys?.length || 0 }} 条</div></div>
                  <div><div class="text-gray-400">置信度</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ formatConfidence(testRouteDecision.confidence) }}</div></div>
                  <div><div class="text-gray-400">协议版本</div><div class="mt-0.5 break-all font-mono text-gray-700 dark:text-gray-200">{{ testRouteDecision.schema_version || '未记录' }}</div></div>
                </div>
                <div v-if="testRequirements.length" class="mt-3">
                  <div class="text-xs text-gray-400">回答需求</div>
                  <div class="mt-2 space-y-2">
                    <div v-for="requirement in testRequirements" :key="requirement.id || requirement.description" class="rounded-lg border border-gray-200 px-3 py-2 text-xs dark:border-gray-600">
                      <div class="flex flex-wrap items-center gap-2">
                        <n-tag size="small" :bordered="false">{{ requirementRoleLabel(requirement.role) }}</n-tag>
                        <span class="text-gray-500 dark:text-gray-400">{{ requirement.origin || '来源未记录' }}</span>
                      </div>
                      <p class="mt-1.5 break-words text-gray-700 dark:text-gray-200">{{ requirement.description || '未提供需求说明' }}</p>
                    </div>
                  </div>
                </div>
                <div v-if="testClarificationQuestion" class="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
                  澄清问题：{{ testClarificationQuestion }}
                  <div v-if="testRouteDecision.clarification?.unresolved?.length" class="mt-1.5 text-amber-700 dark:text-amber-300">
                    未解决项：{{ testRouteDecision.clarification.unresolved.map(unresolvedSlotLabel).join('、') }}
                  </div>
                </div>
              </section>

              <section v-if="testTaskContract" class="mt-4 border-t border-gray-200 pt-3 dark:border-gray-600">
                <h4 class="text-xs font-semibold text-gray-600 dark:text-gray-300">后端编译合同</h4>
                <div class="mt-3 grid grid-cols-1 gap-x-4 gap-y-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                  <div><div class="text-gray-400">回答模式</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ responseModeLabel(responseModeFor(testResult)) }}</div></div>
                  <div><div class="text-gray-400">检索策略</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ retrievalPolicyLabel(retrievalPolicyFor(testResult)) }}</div></div>
                  <div><div class="text-gray-400">最终是否检索</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ booleanDecisionLabel(needsRetrievalValue(testResult)) }}</div></div>
                  <div><div class="text-gray-400">允许执行</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ booleanDecisionLabel(dispatchAuthorizedFor(testResult)) }}</div></div>
                  <div><div class="text-gray-400">合同状态</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ readinessLabel(testTaskContract.readiness) }}</div></div>
                  <div><div class="text-gray-400">会话关系</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ relationLabel(testTaskContract.relation) }}</div></div>
                  <div><div class="text-gray-400">查询处理</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ queryResolutionLabel(testTaskContract.query_mode || testTaskContract.query_resolution?.mode) }}</div></div>
                  <div><div class="text-gray-400">知识库上下文</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ testTaskContract.selected_kb_count ?? 0 }} 个</div></div>
                  <div><div class="text-gray-400">编译需求</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ Array.isArray(testTaskContract.requirements) ? testTaskContract.requirements.length : 0 }} 项</div></div>
                  <div><div class="text-gray-400">合同版本</div><div class="mt-0.5 break-all font-mono text-gray-700 dark:text-gray-200">{{ testTaskContract.schema_version || '未记录' }}</div></div>
                  <div class="sm:col-span-2 lg:col-span-3"><div class="text-gray-400">编译原因</div><div class="mt-0.5 break-words text-gray-700 dark:text-gray-200">{{ decisionReasonLabel(decisionReasonFor(testResult)) }}</div></div>
                </div>
                <div v-if="testTaskContract.clarification?.question" class="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
                  合同澄清：{{ testTaskContract.clarification.question }}
                  <div v-if="testTaskContract.clarification?.unresolved?.length" class="mt-1.5 text-amber-700 dark:text-amber-300">
                    未解决项：{{ testTaskContract.clarification.unresolved.map(unresolvedSlotLabel).join('、') }}
                  </div>
                </div>
              </section>

              <section v-if="!testRouteDecision && !testTaskContract" class="border-t border-gray-200 pt-3 dark:border-gray-600">
                <div class="grid grid-cols-1 gap-x-4 gap-y-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                  <div><div class="text-gray-400">旧版意图</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ testResult.intent_name || testResult.intent_code || testResult.intent || '未记录' }}</div></div>
                  <div><div class="text-gray-400">分类动作</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ actionLabel(testAction) }}</div></div>
                  <div><div class="text-gray-400">回答模式</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ responseModeLabel(responseModeFor(testResult)) }}</div></div>
                  <div><div class="text-gray-400">最终是否检索</div><div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ booleanDecisionLabel(needsRetrievalValue(testResult)) }}</div></div>
                </div>
              </section>

              <section class="mt-4 border-t border-gray-200 pt-3 dark:border-gray-600">
                <h4 class="text-xs font-semibold text-gray-600 dark:text-gray-300">诊断摘要</h4>
                <div class="mt-3 grid grid-cols-1 gap-x-4 gap-y-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <div class="text-gray-400">判定来源</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ sourceLabel(testDiagnostics.source || testResult.decision_source || testResult.source) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">Schema 校验</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ diagnosticBooleanLabel(testDiagnostics.schema_valid) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">输出约束</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ structuredOutputLabel(testDiagnostics) }}</div>
                </div>
                <div>
                  <div class="text-gray-400">格式修复</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ diagnosticBooleanLabel(testDiagnostics.repair_used, '已使用', '未使用') }}</div>
                </div>
                <div>
                  <div class="text-gray-400">备用 / 兜底</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ diagnosticBooleanLabel(testDiagnostics.fallback_used, '已使用', '未使用') }}</div>
                </div>
                <div>
                  <div class="text-gray-400">Prompt 版本</div>
                  <div class="mt-0.5 break-all font-mono text-gray-700 dark:text-gray-200">{{ testDiagnostics.prompt_version || '未记录' }}</div>
                </div>
                <div>
                  <div class="text-gray-400">模型</div>
                  <div class="mt-0.5 break-all text-gray-700 dark:text-gray-200">{{ testDiagnostics.model || testDiagnostics.route_model || '未记录' }}</div>
                </div>
                <div>
                  <div class="text-gray-400">执行状态</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ retrievalExecutionLabel(testResult, true) }}</div>
                </div>
                <div v-if="testDiagnostics.latency_ms !== undefined && testDiagnostics.latency_ms !== null">
                  <div class="text-gray-400">耗时</div>
                  <div class="mt-0.5 text-gray-700 dark:text-gray-200">{{ testDiagnostics.latency_ms }} ms</div>
                </div>
              </div>
              </section>
              <div v-if="testReason && !testTaskContract" class="mt-3 border-t border-gray-200 pt-3 text-xs leading-relaxed text-gray-500 dark:border-gray-600 dark:text-gray-400">
                <span class="text-gray-400">策略原因：</span>
                <span :title="testReason">{{ decisionReasonLabel(testReason) }}</span>
              </div>
            </div>
          </SurfaceCard>

          <!-- 运行说明 -->
          <section class="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-800 rounded-xl border border-blue-100 dark:border-gray-700 p-5 sm:p-6">
            <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-200">安全的路由边界</h3>
            <div class="mt-4 space-y-3 text-sm text-gray-600 dark:text-gray-300">
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">1</span><span>规则和模型只产生语义决定，不能直接控制接口、知识库或越过权限边界。</span></div>
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">2</span><span>后端编译器结合语义决定与知识库选择状态，独立确定回答模式、检索策略和执行授权。</span></div>
              <div class="flex gap-3"><span class="text-blue-500 font-semibold">3</span><span>日志同时记录语义决定、编译合同和证据状态，便于区分“等待澄清”“跳过检索”与“检索无命中”。</span></div>
            </div>
          </section>
          </div>
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
  getIntentRoutingConfig, testIntentRouting, updateIntentCategory,
  updateIntentRoutingConfig,
} from '@/api/intentRouting'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import RowActions from '@/components/ui/RowActions.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'
import {
  evidenceStatusLabel as evidenceStatusContractLabel,
  evidenceStatusTagType as evidenceStatusContractTagType,
  normalizeEvidenceStatus,
} from '@/utils/evidenceStatus'

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
const initialLoading = ref(false)
const categoriesLoading = ref(false)
const savingConfig = ref(false)
const savingCategory = ref(false)
const testing = ref(false)
const testQuery = ref('')
const testResult = ref(null)
const testContextMessages = ref([])
const testSelectedKbCount = ref(0)

const MAX_TEST_CONTEXT_MESSAGES = 6
let testContextMessageSequence = 0

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
const contextRoleOptions = [
  { label: '用户', value: 'user' },
  { label: '助手', value: 'assistant' },
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

const testRouteDecision = computed(() => explicitRouteDecisionFor(testResult.value))
const testTaskContract = computed(() => taskContractFor(testResult.value))
const testDiagnostics = computed(() => diagnosticsFor(testResult.value))
const testRequirements = computed(() => (
  Array.isArray(testRouteDecision.value?.requirements) ? testRouteDecision.value.requirements : []
))
const testClarificationQuestion = computed(() => (
  testRouteDecision.value?.clarification?.question || testRouteDecision.value?.clarification_question || ''
))
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
    await Promise.all([loadConfig(), loadCategories()])
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

function addTestContextMessage() {
  if (testContextMessages.value.length >= MAX_TEST_CONTEXT_MESSAGES) return
  testContextMessageSequence += 1
  testContextMessages.value.push({
    id: `context-${testContextMessageSequence}`,
    role: testContextMessages.value.at(-1)?.role === 'user' ? 'assistant' : 'user',
    content: '',
  })
}

function removeTestContextMessage(id) {
  testContextMessages.value = testContextMessages.value.filter(item => item.id !== id)
}

async function runTest() {
  const query = testQuery.value.trim()
  if (!query) return
  testing.value = true
  testResult.value = null
  try {
    const contextMessages = testContextMessages.value
      .map(item => ({ role: item.role, content: item.content.trim() }))
      .filter(item => item.content)
    const data = await testIntentRouting({
      question: query,
      current_input: query,
      context_messages: contextMessages,
      selected_kb_count: Math.max(0, Math.trunc(Number(testSelectedKbCount.value) || 0)),
    })
    testResult.value = normalizeTestResult(data)
  } catch (error) {
    showError(error, '测试路由失败')
  } finally {
    testing.value = false
  }
}

function normalizeTestResult(data) {
  const routeDecision = explicitRouteDecisionFor(data)
  const taskContract = taskContractFor(data)
  const diagnostics = {
    ...diagnosticsFor(data),
    latency_ms: diagnosticsFor(data).latency_ms ?? data?.latency_ms ?? null,
  }
  const legacyDecision = data?.decision && !routeDecision
    ? data.decision
    : (!routeDecision && !taskContract ? data : {})
  return {
    ...(legacyDecision || {}),
    route_decision: routeDecision,
    task_contract: taskContract,
    diagnostics,
    latency_ms: diagnostics.latency_ms,
    retrieval_executed: data?.retrieval_executed ?? legacyDecision?.retrieval_executed,
    evidence_status: data?.evidence_status ?? legacyDecision?.evidence_status,
    hit_count: data?.hit_count ?? legacyDecision?.hit_count,
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

function explicitRouteDecisionFor(result) {
  if (!result || typeof result !== 'object') return null
  const nested = result.route_decision || result.semantic_decision || result.decision?.route_decision
  if (nested && typeof nested === 'object') return nested
  const nestedDecisionVersion = String(result.decision?.schema_version || '')
  if (nestedDecisionVersion.startsWith('rag_route_decision.') || nestedDecisionVersion.startsWith('route_decision.')) {
    return result.decision
  }
  const schemaVersion = String(result.schema_version || '')
  return schemaVersion.startsWith('rag_route_decision.') || schemaVersion.startsWith('route_decision.')
    ? result
    : null
}

function taskContractFor(result) {
  if (!result || typeof result !== 'object') return null
  const contract = result.task_contract || result.contract || result.execution_contract
    || result.route_summary || result.decision?.task_contract
  if (contract && typeof contract === 'object') return contract
  const nestedDecisionVersion = String(result.decision?.schema_version || '')
  if (nestedDecisionVersion.startsWith('rag_task_contract.') || nestedDecisionVersion.startsWith('task_contract.')) {
    return result.decision
  }
  const schemaVersion = String(result.schema_version || '')
  return schemaVersion.startsWith('rag_task_contract.') || schemaVersion.startsWith('task_contract.')
    ? result
    : null
}

function diagnosticsFor(result) {
  if (!result || typeof result !== 'object') return {}
  const raw = result.diagnostics && typeof result.diagnostics === 'object' ? result.diagnostics : {}
  const fallbackSignals = [
    raw.fallback_model_used,
    raw.safe_fallback_used,
    result.fallback_model_used,
    result.safe_fallback_used,
  ].filter(value => typeof value === 'boolean')
  return {
    ...raw,
    source: raw.source ?? taskContractFor(result)?.source ?? result.decision_source ?? result.source,
    schema_valid: raw.schema_valid ?? result.schema_valid ?? result.route_schema_valid,
    strict_schema_used: raw.strict_schema_used ?? result.strict_schema_used,
    json_object_fallback_used: raw.json_object_fallback_used ?? result.json_object_fallback_used,
    repair_used: raw.repair_used ?? raw.repair_attempted ?? result.repair_used,
    fallback_used: raw.fallback_used ?? result.fallback_used
      ?? (fallbackSignals.length ? fallbackSignals.some(Boolean) : undefined),
    latency_ms: raw.latency_ms ?? result.latency_ms,
  }
}

function semanticOperationFor(result) {
  const decision = explicitRouteDecisionFor(result)
  return decision?.operation || decision?.intent_code || result?.operation || result?.intent_code || result?.intent || ''
}

function relationFor(result) {
  const decision = explicitRouteDecisionFor(result)
  return decision?.relation || taskContractFor(result)?.relation
    || result?.relation || result?.conversation_relation || ''
}

function readinessFor(result) {
  return taskContractFor(result)?.readiness || semanticReadinessFor(result)
}

function semanticReadinessFor(result) {
  const decision = explicitRouteDecisionFor(result)
  return decision?.readiness || taskContractFor(result)?.readiness || result?.readiness || ''
}

function evidenceScopeFor(result) {
  const decision = explicitRouteDecisionFor(result)
  return decision?.evidence_scope || taskContractFor(result)?.evidence_scope || result?.evidence_scope || ''
}

function queryResolutionModeFor(result) {
  const decision = explicitRouteDecisionFor(result)
  return decision?.query_resolution?.mode || decision?.query_resolution_mode
    || taskContractFor(result)?.query_mode || result?.query_resolution_mode || result?.query_mode || ''
}

function operationLabel(operation) {
  return {
    knowledge_qa: '知识库问答',
    general_chat: '通用交流',
    writing: '写作 / 润色',
    platform_help: '平台帮助',
    system_help: '平台帮助',
    other: '未识别问题',
  }[operation] || operation || '未记录'
}

function relationLabel(relation) {
  return {
    new: '新任务',
    standalone: '独立问题',
    followup: '追问',
    refinement: '细化追问',
    correction: '修正',
    continuation: '继续任务',
  }[relation] || relation || '未记录'
}

function readinessLabel(readiness) {
  return {
    ready: '可编译',
    needs_clarification: '需要澄清',
    blocked: '不可执行',
  }[readiness] || readiness || '未记录'
}

function readinessTagType(readiness) {
  return {
    ready: 'success',
    needs_clarification: 'warning',
    blocked: 'error',
  }[readiness] || 'default'
}

function evidenceScopeLabel(scope) {
  return {
    enterprise_kb: '企业知识库',
    current_input: '当前输入',
    general_world: '通用知识',
    platform_self: '当前平台',
    mixed: '混合范围',
  }[scope] || scope || '未记录'
}

function queryResolutionLabel(mode) {
  return {
    current: '使用当前问题',
    contextualize: '结合上下文解析',
    use_current: '使用当前问题',
    rewrite_with_context: '结合上下文改写',
    use_context: '使用会话上下文',
    clarify: '先澄清',
  }[mode] || mode || '未记录'
}

function requirementRoleLabel(role) {
  return {
    answer: '回答需求',
    bridge: '推理桥接',
    constraint: '约束',
  }[role] || role || '未分类'
}

function unresolvedSlotLabel(item) {
  const role = item?.role
  const reason = item?.reason
  const roleLabel = {
    query_execution: '查询执行条件',
    knowledge_base: '知识库范围',
    context_object: '追问对象',
    context_turn: '上下文轮次',
    subject: '查询对象',
    user_grade: '适用职级',
  }[role] || role || '未命名项'
  const reasonLabel = {
    missing: '缺少信息',
    ambiguous: '存在歧义',
    unavailable: '暂不可用',
  }[reason] || reason || '未说明'
  return `${roleLabel}（${reasonLabel}）`
}

function responseModeFor(result) {
  if (!result) return ''
  const contract = taskContractFor(result)
  const contractMode = contract?.response_mode || contract?.execution?.response_mode
  if (contractMode) return contractMode
  if (result.response_mode) return result.response_mode
  if (explicitRouteDecisionFor(result)) return ''
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
  const contract = taskContractFor(result)
  const contractPolicy = contract?.retrieval_policy || contract?.retrieval?.policy || contract?.execution?.retrieval_policy
  if (contractPolicy) return contractPolicy
  if (result.retrieval_policy) return result.retrieval_policy
  if (explicitRouteDecisionFor(result)) return ''
  const needRetrieval = needsRetrievalValue(result)
  if (needRetrieval === true) return 'required'
  if (needRetrieval === false) return 'skip'
  return ''
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
  const contract = taskContractFor(result)
  return contract?.decision_reason || contract?.reason || result?.decision_reason || result?.reason || result?.message || ''
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
  const status = normalizeEvidenceStatus(result?.evidence_status)
  if (status) return status
  if (result?.retrieval_executed === false) return 'skipped'
  if (result?.retrieval_executed === true && Number(result?.hit_count) > 0) return 'hit'
  if (result?.retrieval_executed === true && result?.hit_count !== undefined && result?.hit_count !== null) return 'no_hit'
  if (Number(result?.hit_count) > 0) return 'hit'
  return 'unverified'
}

function evidenceStatusLabel(status, simulation = false) {
  if (!status && simulation) return '未执行（仅测试路由策略）'
  return evidenceStatusContractLabel(status, status || '未记录')
}

function evidenceStatusTagType(status) {
  return evidenceStatusContractTagType(status)
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

function formatConfidence(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : '—'
}

function needsRetrievalValue(result) {
  if (!result) return null
  const contract = taskContractFor(result)
  const contractValue = contract?.need_retrieval ?? contract?.needs_retrieval
    ?? contract?.retrieval?.required ?? contract?.execution?.need_retrieval
  if (typeof contractValue === 'boolean') return contractValue
  if (typeof result.need_retrieval === 'boolean') return result.need_retrieval
  if (typeof result.needs_retrieval === 'boolean') return result.needs_retrieval
  if (explicitRouteDecisionFor(result) || contract) return null
  const action = result.action || result.route_action
  return action ? action === 'retrieve' : null
}

function dispatchAuthorizedFor(result) {
  const contract = taskContractFor(result)
  const value = contract?.dispatch_authorized ?? contract?.dispatchAuthorized
    ?? contract?.execution?.dispatch_authorized ?? result?.dispatch_authorized
  return typeof value === 'boolean' ? value : null
}

function booleanDecisionLabel(value) {
  if (value === true) return '是'
  if (value === false) return '否'
  return '未记录'
}

function diagnosticBooleanLabel(value, trueLabel = '通过', falseLabel = '未通过') {
  if (value === true) return trueLabel
  if (value === false) return falseLabel
  return '未记录'
}

function structuredOutputLabel(diagnostics) {
  if (diagnostics?.json_object_fallback_used === true) return 'JSON Object 兼容模式'
  if (diagnostics?.strict_schema_used === true) return 'Strict JSON Schema'
  return '未记录'
}

function showError(error, fallback) {
  msg.error(error?.response?.data?.detail || fallback)
}
</script>
