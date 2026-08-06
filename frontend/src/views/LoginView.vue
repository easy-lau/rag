<template>
  <main class="auth-page">
    <section class="auth-story" aria-label="产品说明">
      <div class="auth-story__content">
        <div class="brand-mark" aria-hidden="true">
          <img v-if="siteStore.site_logo" :src="siteStore.site_logo" alt="" />
          <n-icon v-else><ShieldCheckmarkOutline /></n-icon>
        </div>
        <p class="auth-eyebrow">{{ siteStore.site_title || 'RAG KNOWLEDGE' }}</p>
        <h1>让企业知识有据可查，<br />让每次回答都有出处。</h1>
        <p class="auth-story__description">
          {{ siteStore.site_description || '面向团队内部的安全知识问答平台。权限、范围与证据在回答前完成校验。' }}
        </p>
        <div class="trust-note">内部系统 · 仅限授权成员使用</div>
      </div>
    </section>

    <section class="auth-panel" aria-labelledby="login-title">
      <div class="auth-card">
        <header class="auth-heading">
          <p class="auth-eyebrow">欢迎回来</p>
          <h2 id="login-title">登录知识中心</h2>
          <p>使用管理员为你创建的内部账号登录。</p>
        </header>

        <n-form
          ref="formRef"
          :model="form"
          :rules="rules"
          label-placement="top"
          @keydown.enter="handleLoginEnter"
        >
          <n-form-item path="username" label="登录账号">
            <n-input
              v-model:value="form.username"
              placeholder="请输入账号"
              size="large"
              :input-props="{ autocomplete: 'username' }"
            >
              <template #prefix><n-icon><PersonOutline /></n-icon></template>
            </n-input>
          </n-form-item>
          <n-form-item path="password" label="密码">
            <n-input
              ref="passwordInputRef"
              v-model:value="form.password"
              type="password"
              show-password-on="click"
              placeholder="请输入密码"
              size="large"
              :input-props="{ autocomplete: 'current-password' }"
            >
              <template #prefix><n-icon><LockClosedOutline /></n-icon></template>
            </n-input>
          </n-form-item>
          <n-button
            type="primary"
            block
            size="large"
            class="auth-submit"
            :loading="loading"
            :disabled="cooldownSeconds > 0"
            @click="handleLogin"
          >
            {{ loginButtonText }}
          </n-button>
        </n-form>

        <p class="support-copy">忘记密码请联系系统管理员重置。</p>
      </div>

      <footer v-if="siteStore.site_copyright" class="auth-footer">
        {{ siteStore.site_copyright }}
      </footer>
    </section>
  </main>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NForm, NFormItem, NInput, NButton, NIcon, useMessage } from 'naive-ui'
import { PersonOutline, LockClosedOutline, ShieldCheckmarkOutline } from '@vicons/ionicons5'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { defaultWorkspaceRoute } from '@/router/menus'

const router = useRouter()
const route = useRoute()
const message = useMessage()
const authStore = useAuthStore()
const siteStore = useSiteStore()

const formRef = ref(null)
const passwordInputRef = ref(null)
const loading = ref(false)
const cooldownSeconds = ref(0)
let cooldownTimer = null
const form = ref({ username: '', password: '' })
const rules = {
  username: { required: true, message: '请输入用户名', trigger: 'blur' },
  password: { required: true, message: '请输入密码', trigger: 'blur' },
}

const loginButtonText = computed(() => {
  if (loading.value) return '正在验证身份'
  if (cooldownSeconds.value > 0) return `${cooldownSeconds.value} 秒后重试`
  return '安全登录'
})

function startCooldown(seconds) {
  const safeSeconds = Math.min(86400, Math.max(1, Number.parseInt(seconds, 10) || 60))
  cooldownSeconds.value = safeSeconds
  if (cooldownTimer) window.clearInterval(cooldownTimer)
  cooldownTimer = window.setInterval(() => {
    cooldownSeconds.value = Math.max(0, cooldownSeconds.value - 1)
    if (!cooldownSeconds.value && cooldownTimer) {
      window.clearInterval(cooldownTimer)
      cooldownTimer = null
    }
  }, 1000)
}

onBeforeUnmount(() => {
  if (cooldownTimer) window.clearInterval(cooldownTimer)
})

function handleLoginEnter(event) {
  // 中文输入法使用 Enter 确认候选时不能提交表单。keydown 阶段的
  // isComposing / 229 比 keyup 更可靠，因为 keyup 前合成通常已经结束。
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  if (!form.value.password) {
    passwordInputRef.value?.focus()
    return
  }
  void handleLogin()
}

async function handleLogin() {
  if (loading.value || cooldownSeconds.value > 0) return
  try {
    await formRef.value?.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    await authStore.login(form.value.username, form.value.password)
    const redirect = route.query.redirect
    const target = typeof redirect === 'string'
      && redirect.startsWith('/')
      && !redirect.startsWith('//')
      && !redirect.startsWith('/login')
      ? redirect
      : defaultWorkspaceRoute(authStore)
    router.push(target)
  } catch (e) {
    const detail = e?.response?.data?.detail
    if (e?.response?.status === 429) {
      const retryAfter = e.response.headers?.['retry-after']
      startCooldown(retryAfter)
      message.warning(detail || '登录请求过于频繁，请稍后再试')
    } else {
      message.error(detail || '登录失败，请检查用户名或密码')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-page {
  --auth-brand-start: #0047ff;
  --auth-brand-end: #4b7dff;
  --auth-brand-hover-start: #165dff;
  --auth-brand-hover-end: #6590ff;
  --auth-brand-pressed-start: #003ad9;
  --auth-brand-pressed-end: #366ef4;
  --auth-accent: #0047ff;
  --auth-ink: #1d2129;
  --auth-muted: #636e80;
  min-height: 100%;
  display: grid;
  grid-template-columns: minmax(360px, 1.1fr) minmax(420px, 0.9fr);
  background:
    radial-gradient(circle at 100% 0%, rgba(0, 75, 255, 0.055), transparent 30%),
    #fbfbff;
  color: var(--ui-text);
}

.auth-story {
  position: relative;
  display: flex;
  overflow: hidden;
  align-items: center;
  padding: clamp(48px, 8vw, 120px);
  color: var(--auth-ink);
  background:
    radial-gradient(circle at 18% 12%, rgba(150, 202, 255, 0.5), transparent 33%),
    linear-gradient(155deg, #edf5ff 0%, #fafdff 52%, #e8eef8 100%);
}

.auth-story::before {
  position: absolute;
  right: -180px;
  bottom: 8%;
  width: 540px;
  height: 290px;
  border: 1px solid rgba(75, 125, 255, 0.18);
  border-radius: 56px;
  background: linear-gradient(135deg, rgba(75, 125, 255, 0.34), rgba(150, 202, 255, 0.12));
  box-shadow: 0 34px 80px rgba(44, 100, 220, 0.14);
  transform: rotate(-11deg);
  content: '';
}

.auth-story::after {
  position: absolute;
  right: 13%;
  bottom: -118px;
  width: 275px;
  height: 275px;
  border: 1px solid rgba(191, 215, 255, 0.9);
  border-radius: 52px;
  background: rgba(255, 255, 255, 0.68);
  box-shadow: 0 26px 62px rgba(49, 94, 251, 0.12);
  transform: rotate(16deg);
  content: '';
}

.auth-story__content {
  position: relative;
  z-index: 1;
  max-width: 680px;
}

.brand-mark {
  width: 56px;
  height: 56px;
  display: grid;
  overflow: hidden;
  place-items: center;
  border: 1px solid rgba(0, 71, 255, 0.12);
  border-radius: 14px;
  background: linear-gradient(180deg, var(--auth-brand-start) 0%, var(--auth-brand-end) 100%);
  color: #ffffff;
  box-shadow: 0 12px 28px rgba(0, 71, 255, 0.16);
  font-size: 28px;
}

.brand-mark img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.auth-eyebrow {
  margin: 0;
  color: var(--auth-accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.auth-story .auth-eyebrow {
  margin-top: 36px;
  color: #0047ff;
}

.auth-story h1 {
  max-width: 680px;
  margin: 24px 0;
  color: inherit;
  font-size: clamp(38px, 4.6vw, 66px);
  font-weight: 700;
  line-height: 1.14;
  letter-spacing: -0.04em;
}

.auth-story__description {
  max-width: 680px;
  margin: 0;
  color: var(--auth-muted);
  font-size: 17px;
  line-height: 1.8;
}

.trust-note {
  margin-top: 36px;
  color: #707481;
  font-size: 13px;
  letter-spacing: 0.08em;
}

.auth-panel {
  position: relative;
  display: grid;
  min-width: 0;
  place-items: center;
  padding: 48px;
  border-left: 1px solid var(--ui-border-strong);
  background: transparent;
}

.auth-card {
  width: min(100%, 420px);
}

.auth-heading h2 {
  margin: 10px 0 8px;
  color: var(--ui-text);
  font-size: 30px;
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.035em;
}

.auth-heading > p:last-child {
  margin: 0;
  color: var(--ui-text-secondary);
  font-size: 14px;
  line-height: 1.7;
}

.auth-card :deep(.n-form) {
  margin-top: 28px;
}

.auth-card :deep(.n-form-item) {
  margin-bottom: 20px;
}

.auth-card :deep(.n-form-item-label) {
  padding-bottom: 8px;
  color: var(--ui-text);
  font-weight: 600;
}

.auth-card :deep(.n-input) {
  --n-border-focus: 1px solid var(--auth-accent) !important;
  --n-border-hover: 1px solid color-mix(in srgb, var(--auth-accent) 72%, var(--ui-border)) !important;
  --n-box-shadow-focus: 0 0 0 3px color-mix(in srgb, var(--auth-accent) 16%, transparent) !important;
  min-height: 46px;
  border-radius: 8px;
}

.auth-card :deep(.n-input-wrapper) {
  padding-right: 13px;
  padding-left: 13px;
}

.auth-card :deep(.n-input__prefix) {
  margin-right: 8px;
  color: var(--ui-icon);
}

.auth-submit {
  --n-color: transparent !important;
  --n-color-hover: transparent !important;
  --n-color-pressed: transparent !important;
  --n-color-focus: transparent !important;
  --n-border: 0 !important;
  --n-border-hover: 0 !important;
  --n-border-pressed: 0 !important;
  --n-border-focus: 0 !important;
  height: 48px;
  margin-top: 2px;
  border-radius: 8px;
  background: linear-gradient(180deg, var(--auth-brand-start) 0%, var(--auth-brand-end) 100%);
  box-shadow: 0 6px 16px rgba(0, 71, 255, 0.16);
  font-weight: 650;
  transition: box-shadow 0.2s ease, filter 0.2s ease;
}

.auth-submit:hover {
  background: linear-gradient(180deg, var(--auth-brand-hover-start) 0%, var(--auth-brand-hover-end) 100%);
  box-shadow: 0 8px 20px rgba(0, 71, 255, 0.2);
  filter: saturate(1.04);
}

.auth-submit:active {
  background: linear-gradient(180deg, var(--auth-brand-pressed-start) 0%, var(--auth-brand-pressed-end) 100%);
  box-shadow: 0 3px 10px rgba(0, 71, 255, 0.16);
  filter: none;
}

.dark .auth-page {
  --auth-accent: #7aa9ff;
  background:
    radial-gradient(circle at 100% 0%, rgba(77, 127, 255, 0.1), transparent 30%),
    var(--ui-bg-subtle);
}

.support-copy {
  margin: 20px 0 0;
  color: var(--ui-text-tertiary);
  font-size: 13px;
  text-align: center;
}

.auth-footer {
  position: absolute;
  right: 32px;
  bottom: 24px;
  left: 32px;
  color: var(--ui-text-tertiary);
  font-size: 11px;
  text-align: center;
}

@media (max-width: 860px) {
  .auth-page {
    min-height: 100%;
    grid-template-columns: 1fr;
  }

  .auth-story {
    min-height: 330px;
    padding: 48px 28px;
  }

  .auth-story h1 {
    font-size: 38px;
  }

  .auth-story__description,
  .trust-note {
    display: none;
  }

  .auth-panel {
    min-height: 520px;
    align-items: start;
    padding: 42px 24px 64px;
    border-top: 1px solid var(--ui-border-strong);
    border-left: 0;
  }
}

@media (max-width: 430px) {
  .auth-story {
    min-height: 280px;
    padding: 36px 20px;
  }

  .brand-mark {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    font-size: 24px;
  }

  .auth-story .auth-eyebrow {
    margin-top: 26px;
  }

  .auth-story h1 {
    margin: 18px 0 0;
    font-size: 32px;
  }

  .auth-panel {
    min-height: 500px;
    padding-right: 20px;
    padding-left: 20px;
  }

  .auth-heading h2 {
    font-size: 27px;
  }

  .auth-footer {
    right: 20px;
    left: 20px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .auth-page *,
  .auth-page *::before,
  .auth-page *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }
}
</style>
