import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'


const loginViewSource = readFileSync(
  new URL('../src/views/LoginView.vue', import.meta.url),
  'utf8',
)


test('登录回车忽略中文输入法合成事件', () => {
  assert.ok(loginViewSource.includes('@keydown.enter="handleLoginEnter"'))
  assert.doesNotMatch(loginViewSource, /@keyup\.enter="handleLogin"/)
  assert.ok(loginViewSource.includes('event.isComposing || event.keyCode === 229'))
})


test('账号阶段的普通回车先聚焦密码框而不是提交登录', () => {
  assert.ok(loginViewSource.includes('ref="passwordInputRef"'))
  assert.ok(loginViewSource.includes('if (!form.value.password)'))
  assert.ok(loginViewSource.includes('passwordInputRef.value?.focus()'))
  assert.ok(loginViewSource.includes('void handleLogin()'))
})
