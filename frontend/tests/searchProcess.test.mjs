import test from 'node:test'
import assert from 'node:assert/strict'
import { createPinia, setActivePinia } from 'pinia'

import { useSearchStore } from '../src/stores/search.js'

function store() {
  setActivePinia(createPinia())
  return useSearchStore()
}

test('direct 执行计划只展示问题分析和生成', () => {
  const search = store()
  assert.equal(search.setProcessPlan({
    type: 'search_process',
    execution_path: 'direct',
    steps: [
      { key: 'analyze', label: '问题分析' },
      { key: 'generate', label: '生成' },
    ],
  }), true)

  search.updateStep('analyze', 'done')
  search.updateStep('generate', 'active')

  assert.equal(search.executionPath, 'direct')
  assert.deepEqual(
    search.steps.map(step => [step.key, step.status]),
    [['analyze', 'done'], ['generate', 'active']],
  )
})

test('目录查询使用服务端标签且不会伪造查询扩展和重排', () => {
  const search = store()
  search.setProcessPlan({
    execution_path: 'catalog',
    steps: [
      { key: 'analyze', label: '问题分析' },
      { key: 'retrieve', label: '目录查询' },
      { key: 'generate', label: '生成' },
    ],
  })

  assert.deepEqual(search.steps.map(step => step.label), ['问题分析', '目录查询', '生成'])
  assert.equal(search.steps.some(step => step.key === 'expand'), false)
  assert.equal(search.steps.some(step => step.key === 'rerank'), false)
})

test('RAG 保留完整流程，异常会落在真实活动步骤', () => {
  const search = store()
  search.setProcessPlan({
    execution_path: 'rag',
    steps: [
      { key: 'analyze', label: '问题分析' },
      { key: 'expand', label: '查询扩展' },
      { key: 'retrieve', label: '检索' },
      { key: 'rerank', label: '重排' },
      { key: 'generate', label: '生成' },
    ],
  })
  search.updateStep('retrieve', 'active')
  search.failActiveStep('连接失败')

  assert.deepEqual(search.steps.map(step => step.key), [
    'analyze', 'expand', 'retrieve', 'rerank', 'generate',
  ])
  assert.equal(search.steps.find(step => step.key === 'retrieve').status, 'error')
})

test('结果引用读取显示结果读取，而不是查询扩展和重排', () => {
  const search = store()
  search.setProcessPlan({
    execution_path: 'result_reference',
    steps: [
      { key: 'analyze', label: '问题分析' },
      { key: 'retrieve', label: '结果读取' },
      { key: 'generate', label: '生成' },
    ],
  })

  assert.deepEqual(search.steps.map(step => step.label), ['问题分析', '结果读取', '生成'])
})
