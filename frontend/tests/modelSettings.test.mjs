import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  normalizeOptionalModel,
  resolveOptionalLlmModel,
} from '../src/utils/modelSettings.js'

const modelManagementSource = readFileSync(
  new URL('../src/views/admin/ModelManagementView.vue', import.meta.url),
  'utf8',
)
const settingsViewSource = readFileSync(
  new URL('../src/views/SettingsView.vue', import.meta.url),
  'utf8',
)

test('独立重排模型会去除首尾空白并保留模型 ID', () => {
  assert.equal(normalizeOptionalModel('  fast-reranker  '), 'fast-reranker')
})

test('重排模型留空时回退到对话模型', () => {
  assert.equal(resolveOptionalLlmModel('   ', '  chat-model  '), 'chat-model')
})

test('重排模型已配置时不会被对话模型覆盖', () => {
  assert.equal(
    resolveOptionalLlmModel('rerank-model', 'chat-model'),
    'rerank-model',
  )
})

test('模型管理页把重排模型接入 LLM 模型列表、连接测试和保存请求', () => {
  assert.match(modelManagementSource, /v-model:value="form\.rerank_model"/)
  assert.match(modelManagementSource, /loadModels\('llm', 'rerank'\)/)
  assert.match(modelManagementSource, /handleTestRerankModel/)
  assert.match(modelManagementSource, /rerank_model: normalizeOptionalModel\(form\.value\.rerank_model\)/)
})

test('重排模型操作继续服从 settings write 权限', () => {
  assert.match(modelManagementSource, /authStore\.hasPerm\('settings:write'\)/)
  assert.match(modelManagementSource, /:disabled="!canWrite \|\| connectionTests\.rerank\.loading"/)
})

test('系统设置提供三档知识库未命中兜底策略并随设置保存', () => {
  assert.match(settingsViewSource, /v-model:value="form\.rag_general_fallback_mode"/)
  assert.ok(settingsViewSource.includes("value: 'off'"))
  assert.ok(settingsViewSource.includes("value: 'no_hit'"))
  assert.ok(settingsViewSource.includes("value: 'no_hit_or_insufficient'"))
  assert.ok(settingsViewSource.includes('rag_general_fallback_mode: form.value.rag_general_fallback_mode'))
  assert.match(settingsViewSource, /v-model:value="form\.rag_general_fallback_model"/)
  assert.ok(settingsViewSource.includes('rag_general_fallback_model: form.value.rag_general_fallback_model'))
  assert.ok(settingsViewSource.includes("authStore.hasPerm('settings:write')"))
})
