import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const apiSource = readFileSync(
  new URL('../src/api/terminology.js', import.meta.url),
  'utf8',
)
const menuSource = readFileSync(
  new URL('../src/router/menus.js', import.meta.url),
  'utf8',
)
const routerSource = readFileSync(
  new URL('../src/router/index.js', import.meta.url),
  'utf8',
)
const viewSource = readFileSync(
  new URL('../src/views/admin/TerminologyView.vue', import.meta.url),
  'utf8',
)

test('受控术语 API 覆盖注册表、概念、术语与作用域绑定的真实后端路径', () => {
  assert.match(apiSource, /http\.get\(basePath\(kbId\), \{ params: \{ include_inactive: includeInactive \} \}\)/)
  assert.match(apiSource, /http\.post\(`\$\{basePath\(kbId\)\}\/concepts`, data\)/)
  assert.match(apiSource, /http\.put\(`\$\{basePath\(kbId\)\}\/concepts\/\$\{encodeURIComponent\(conceptId\)\}`, data\)/)
  assert.match(apiSource, /\/concepts\/\$\{encodeURIComponent\(conceptId\)\}\/terms/)
  assert.match(apiSource, /http\.post\(`\$\{basePath\(kbId\)\}\/bindings`, data\)/)
  assert.match(apiSource, /http\.put\(`\$\{basePath\(kbId\)\}\/bindings\/\$\{encodeURIComponent\(bindingId\)\}`, data\)/)
  assert.doesNotMatch(apiSource, /http\.delete\(/)
})

test('受控术语菜单、路由与旧书签重定向使用同一个派生菜单权限', () => {
  assert.match(menuSource, /to: '\/admin\/terminology', label: '受控术语'.*permission: 'menu:terminology'/)
  assert.match(routerSource, /path: 'terminology'.*name: 'terminology'.*permission: 'menu:terminology'/)
  assert.match(routerSource, /path: '\/terminology', redirect: legacyRedirect\('terminology'\)/)
})

test('受控术语页区分读取、维护和文档范围权限，且不要求输入 UUID', () => {
  assert.match(viewSource, /authStore\.hasPerm\('terminology:read'\)/)
  assert.match(viewSource, /authStore\.hasPerm\('terminology:manage'\)/)
  assert.match(viewSource, /当前仅查看。创建、编辑和停用术语的入口已禁用/)
  assert.match(viewSource, /getAllDocuments/)
  assert.match(viewSource, /if \(!canReadDocuments\.value && bindingForm\.value\.document_id !== KB_WIDE_SCOPE\)/)
  assert.match(viewSource, /当前没有文档查看权限，不能创建或编辑文档范围绑定/)
  assert.match(viewSource, /不提供删除操作/)
  assert.doesNotMatch(viewSource, /placeholder=".*UUID/)
})
