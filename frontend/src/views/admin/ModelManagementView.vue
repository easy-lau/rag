<template>
  <div class="model-management h-full overflow-y-auto p-4 sm:p-6">
    <n-spin :show="settingsStore.loading">
      <div class="mx-auto max-w-6xl space-y-5">
        <PageHeader
          title="模型管理"
          description="集中配置对话、向量和多模态模型服务。配置保存在数据库中，服务器环境变量不再承载模型密钥。"
        >
          <template #meta>
            <n-tag :type="canWrite ? 'success' : 'warning'" :bordered="false" round>
              {{ canWrite ? '可编辑' : '仅查看' }}
            </n-tag>
          </template>
        </PageHeader>

        <div v-if="!canWrite" class="flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/30 dark:text-amber-200">
          <n-icon :size="17"><LockClosedOutline /></n-icon>
          当前账号仅可查看模型配置，不能修改或测试模型服务。
        </div>

        <div class="grid grid-cols-1 gap-5 xl:grid-cols-2">
          <SurfaceCard v-for="service in services" :key="service.key" padding="lg" class="model-card">
            <div class="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 class="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                  <span class="h-2 w-2 rounded-full" :class="service.dotClass"></span>
                  {{ service.title }}
                </h3>
                <p v-if="service.description" class="mt-1 text-xs leading-5 text-gray-400">{{ service.description }}</p>
              </div>
              <n-tag size="small" :bordered="false" round>{{ service.badge }}</n-tag>
            </div>

            <n-form :model="form" :disabled="!canWrite" label-placement="top" class="model-form">
              <n-form-item label="Base URL">
                <n-input v-model:value="form[service.baseUrlField]" placeholder="https://api.openai.com/v1" />
              </n-form-item>
              <n-form-item label="API Key">
                <n-input v-model:value="serviceKeys[service.key]" type="password" show-password-on="click" :placeholder="keyPlaceholder(settingsStore.data[service.savedKeyField])" />
              </n-form-item>
              <n-form-item :label="service.modelLabel">
                <div class="model-picker-field">
                  <div class="model-picker-row">
                    <n-auto-complete
                      v-model:value="form[service.modelField]"
                      :options="modelOptions(service.key)"
                      :loading="modelLists[service.key].loading"
                      :disabled="!canWrite"
                      :placeholder="modelSelectPlaceholder"
                      :get-show="showModelOptions"
                      class="model-picker-input"
                    />
                    <n-button
                      secondary
                      class="model-picker-button"
                      :loading="modelLists[service.key].loading"
                      :disabled="!canWrite || modelLists[service.key].loading"
                      @click="loadModels(service.key)"
                    >
                      <template #icon><n-icon><RefreshOutline /></n-icon></template>
                      获取模型
                    </n-button>
                  </div>
                  <p class="model-help min-h-5 text-xs leading-5" :class="modelListHelpClass(service.key)" aria-live="polite">
                    {{ modelListHelpText(service.key) }}
                  </p>
                </div>
              </n-form-item>

              <div v-if="service.key === 'llm'" class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <n-form-item label="Temperature">
                  <n-input-number v-model:value="form.temperature" :min="0" :max="2" :step="0.1" class="w-full" />
                </n-form-item>
                <n-form-item label="Max Tokens">
                  <n-input-number v-model:value="form.max_tokens" :min="256" :max="8192" :step="256" class="w-full" />
                </n-form-item>
              </div>
            </n-form>

            <div class="mt-1 flex flex-wrap items-center gap-3">
              <n-button
                secondary
                size="small"
                :loading="connectionTests[service.key].loading"
                :disabled="!canWrite || connectionTests[service.key].loading"
                @click="handleTestConnection(service.key)"
              >
                测试当前填写
              </n-button>
              <span class="text-xs text-gray-400">{{ service.testHint }}</span>
            </div>
            <n-alert
              v-if="connectionTests[service.key].result"
              class="mt-4"
              :type="connectionTests[service.key].result.type"
              :show-icon="true"
              aria-live="polite"
            >
              {{ connectionTests[service.key].result.message }}
            </n-alert>
          </SurfaceCard>
        </div>

        <div v-if="canWrite" class="flex justify-end pb-2">
          <n-button type="primary" size="large" class="px-10" :loading="saving" @click="handleSave">保存模型配置</n-button>
        </div>
      </div>
    </n-spin>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { NAlert, NAutoComplete, NButton, NForm, NFormItem, NIcon, NInput, NInputNumber, NSpin, NTag, useMessage } from 'naive-ui'
import { LockClosedOutline, RefreshOutline } from '@vicons/ionicons5'
import { getAvailableModels, testModelConnection } from '@/api/settings'
import { useAuthStore } from '@/stores/auth'
import { useSettingsStore } from '@/stores/settings'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'

const settingsStore = useSettingsStore()
const authStore = useAuthStore()
const msg = useMessage()
const saving = ref(false)
const form = ref({ ...settingsStore.data })
const canWrite = computed(() => authStore.hasPerm('settings:write'))
const serviceKeys = reactive({ llm: '', embedding: '', vision: '' })
const connectionTests = ref({
  llm: { loading: false, result: null },
  embedding: { loading: false, result: null },
  vision: { loading: false, result: null },
})

const keyPlaceholder = (status) => status ? `${status}，留空则不修改` : 'sk-...'
const modelSelectPlaceholder = '输入模型 ID，或获取后选择'
const modelLists = reactive({
  llm: { models: [], loading: false, loaded: false, error: '', requestSequence: 0 },
  embedding: { models: [], loading: false, loaded: false, error: '', requestSequence: 0 },
  vision: { models: [], loading: false, loaded: false, error: '', requestSequence: 0 },
})
const modelServiceConfig = {
  llm: { baseUrlField: 'llm_base_url', modelField: 'chat_model', savedKeyField: 'llm_api_key' },
  embedding: { baseUrlField: 'embedding_base_url', modelField: 'embedding_model', savedKeyField: 'embedding_api_key' },
  vision: { baseUrlField: 'vision_base_url', modelField: 'vision_model', savedKeyField: 'vision_api_key' },
}
const services = [
  { key: 'llm', title: '大语言模型', badge: 'Chat', dotClass: 'bg-blue-500', modelLabel: '对话模型', baseUrlField: 'llm_base_url', modelField: 'chat_model', savedKeyField: 'llm_api_key', testHint: '测试不会保存配置；变更 Base URL 时请重新填写密钥。' },
  { key: 'embedding', title: '向量模型', badge: 'Embedding', dotClass: 'bg-purple-500', modelLabel: 'Embedding 模型', baseUrlField: 'embedding_base_url', modelField: 'embedding_model', savedKeyField: 'embedding_api_key', testHint: '测试会校验向量维度；保存变更时也会校验。' },
  { key: 'vision', title: '多模态模型', description: '用于把上传的图片 / 截图通过视觉模型转写为可编辑文本。', badge: 'Vision', dotClass: 'bg-orange-500', modelLabel: '视觉模型', baseUrlField: 'vision_base_url', modelField: 'vision_model', savedKeyField: 'vision_api_key', testHint: '测试不会保存配置；变更 Base URL 时请重新填写密钥。' },
]

function showModelOptions() { return true }
function normalizeBaseUrl(value) { return String(value || '').trim().replace(/\/+$/, '') }
function currentServiceBaseUrl(service) { return normalizeBaseUrl(form.value[modelServiceConfig[service].baseUrlField]) }
function savedServiceBaseUrl(service) { return normalizeBaseUrl(settingsStore.data[modelServiceConfig[service].baseUrlField]) }
function typedServiceKey(service) { return String(serviceKeys[service] || '').trim() }
function hasSavedServiceKey(service) { return Boolean(String(settingsStore.data[modelServiceConfig[service].savedKeyField] || '').trim()) }
function modelListPrerequisite(service) {
  if (!canWrite.value) return '当前账号仅可查看，无法读取模型列表。'
  if (!currentServiceBaseUrl(service)) return '请先填写 Base URL，再获取可用模型。'
  if (savedServiceBaseUrl(service) && savedServiceBaseUrl(service) !== currentServiceBaseUrl(service) && !typedServiceKey(service)) return 'Base URL 已修改，请重新填写对应的 API Key 后再获取模型。'
  if (!typedServiceKey(service) && !hasSavedServiceKey(service)) return '请先填写 API Key，再获取可用模型；模型名称仍可直接输入。'
  return ''
}
function modelOptions(service) {
  const names = [...new Set(modelLists[service].models)]
  const current = String(form.value[modelServiceConfig[service].modelField] || '').trim().toLocaleLowerCase()
  return (current ? names.filter(name => name.toLocaleLowerCase().includes(current)) : names).map(name => ({ label: name, value: name }))
}
function modelListHelpText(service) {
  const state = modelLists[service]
  if (state.loading) return '正在从当前模型服务读取可用模型…'
  if (state.error) return state.error
  if (state.loaded) return state.models.length ? `已获取 ${state.models.length} 个可用模型，可搜索、选择或继续手动输入。` : '当前服务没有返回可用模型，可继续手动输入模型 ID。'
  return modelListPrerequisite(service) || '可直接输入模型 ID，或点击“获取模型”读取服务商列表。'
}
function modelListHelpClass(service) {
  const state = modelLists[service]
  if (state.error) return 'text-[var(--ui-danger)]'
  if (modelListPrerequisite(service)) return 'text-[var(--ui-warning)]'
  return 'text-[var(--ui-text-tertiary)]'
}
function modelNameFromResponse(value) {
  if (typeof value === 'string') return value.trim()
  if (value && typeof value === 'object') return String(value.id || value.name || value.model || '').trim()
  return ''
}
function resetModelList(service) {
  const state = modelLists[service]
  state.requestSequence += 1
  state.models = []
  state.loading = false
  state.loaded = false
  state.error = ''
}
function createModelListPayload(service) {
  const payload = { service, base_url: currentServiceBaseUrl(service) }
  const apiKey = typedServiceKey(service)
  if (apiKey) payload.api_key = apiKey
  return payload
}
async function loadModels(service) {
  const prerequisite = modelListPrerequisite(service)
  if (prerequisite) { if (canWrite.value) msg.warning(prerequisite); return }
  const state = modelLists[service]
  const requestId = ++state.requestSequence
  state.loading = true
  state.error = ''
  try {
    const result = await getAvailableModels(createModelListPayload(service))
    if (requestId !== state.requestSequence) return
    state.models = [...new Set((Array.isArray(result?.models) ? result.models : []).map(modelNameFromResponse).filter(Boolean))]
    state.loaded = true
  } catch (error) {
    if (requestId !== state.requestSequence) return
    state.error = error?.response?.data?.detail || '无法获取模型列表，请检查 Base URL、API Key 与网络连接后重试。'
    state.loaded = false
    msg.error(state.error)
  } finally {
    if (requestId === state.requestSequence) state.loading = false
  }
}
onMounted(async () => {
  await settingsStore.fetch()
  form.value = { ...settingsStore.data }
  serviceKeys.llm = ''
  serviceKeys.embedding = ''
  serviceKeys.vision = ''
})
for (const service of Object.keys(modelServiceConfig)) {
  watch(() => [currentServiceBaseUrl(service), typedServiceKey(service)], ([nextUrl, nextKey], [previousUrl, previousKey]) => {
    if (nextUrl !== previousUrl || nextKey !== previousKey) resetModelList(service)
  })
}
function createConnectionPayload(service) {
  const config = modelServiceConfig[service]
  const payload = { service }
  const apiKey = typedServiceKey(service)
  const baseUrl = form.value[config.baseUrlField]
  const model = form.value[config.modelField]
  if (apiKey) payload.api_key = apiKey
  if (baseUrl?.trim()) payload.base_url = baseUrl.trim()
  if (model?.trim()) payload.model = model.trim()
  return payload
}
function resetConnectionTests() { Object.values(connectionTests.value).forEach(state => { state.loading = false; state.result = null }) }
async function handleTestConnection(service) {
  if (!canWrite.value) return
  const state = connectionTests.value[service]
  state.loading = true
  state.result = null
  try {
    const result = await testModelConnection(createConnectionPayload(service))
    const success = result.ok === true
    const dimensions = success && result.embedding_dimensions ? `，实际向量维度为 ${result.embedding_dimensions}` : ''
    state.result = { type: success ? 'success' : 'error', message: `${result.message || (success ? '模型连接成功' : '连接测试失败，请检查模型配置后重试。')}${dimensions}` }
  } catch (error) {
    state.result = { type: 'error', message: error?.response?.data?.detail || '连接测试失败，请检查模型配置后重试。' }
  } finally { state.loading = false }
}
async function handleSave() {
  if (!canWrite.value) return
  saving.value = true
  try {
    const payload = {
      chat_model: form.value.chat_model,
      temperature: form.value.temperature,
      max_tokens: form.value.max_tokens,
      embedding_model: form.value.embedding_model,
      vision_model: form.value.vision_model,
      llm_base_url: form.value.llm_base_url,
      embedding_base_url: form.value.embedding_base_url,
      vision_base_url: form.value.vision_base_url,
    }
    if (serviceKeys.llm.trim()) payload.llm_api_key = serviceKeys.llm.trim()
    if (serviceKeys.embedding.trim()) payload.embedding_api_key = serviceKeys.embedding.trim()
    if (serviceKeys.vision.trim()) payload.vision_api_key = serviceKeys.vision.trim()
    await settingsStore.save(payload)
    msg.success('模型配置已保存')
    form.value = { ...settingsStore.data }
    serviceKeys.llm = ''
    serviceKeys.embedding = ''
    serviceKeys.vision = ''
    resetConnectionTests()
  } catch { msg.error('保存失败，请检查配置') } finally { saving.value = false }
}
</script>

<style scoped>
.model-picker-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 8px;
  width: 100%;
}
.model-picker-input,
.model-picker-button { min-width: 0; height: 36px; }
:deep(.model-picker-input .n-input),
:deep(.model-picker-input .n-input-wrapper) { width: 100%; }
.model-picker-button { white-space: nowrap; }
@media (max-width: 560px) {
  .model-picker-row { grid-template-columns: 1fr; }
  .model-picker-button { width: 100%; }
}
</style>
