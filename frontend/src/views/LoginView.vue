<template>
  <main class="login-page">
    <div class="login-page__glow login-page__glow--top" aria-hidden="true"></div>
    <div class="login-page__glow login-page__glow--bottom" aria-hidden="true"></div>

    <section class="login-shell">
      <aside class="login-hero">
        <div class="login-hero__grid" aria-hidden="true"></div>
        <div class="login-hero__content">
          <div class="login-hero__brand">
            <img
              v-if="siteStore.site_logo"
              :src="siteStore.site_logo"
              class="login-hero__brand-logo"
              alt=""
            />
            <div v-else class="login-hero__mark" aria-hidden="true">
              <span></span><span></span><span></span>
            </div>
            <span>{{ siteStore.site_title }}</span>
          </div>

          <div class="login-hero__headline">
            <p class="login-hero__eyebrow"><i></i> INTELLIGENCE WORKSPACE</p>
            <h1>让每一份知识<br /><em>即时产生价值</em></h1>
            <p class="login-hero__description">
              {{ siteStore.site_description || '知识增强·精准问答' }}。汇聚分散信息，让答案有迹可循。
            </p>
          </div>

          <div class="login-hero__capabilities" aria-label="产品能力">
            <div><span class="capability-icon capability-icon--search"></span><p>混合检索<small>Hybrid retrieval</small></p></div>
            <div><span class="capability-icon capability-icon--spark"></span><p>可信问答<small>Grounded answers</small></p></div>
            <div><span class="capability-icon capability-icon--shield"></span><p>安全可控<small>Role-based access</small></p></div>
          </div>
        </div>

        <div class="login-hero__orbit login-hero__orbit--one" aria-hidden="true"></div>
        <div class="login-hero__orbit login-hero__orbit--two" aria-hidden="true"></div>
        <div class="login-hero__status"><span></span> Knowledge engine online</div>
      </aside>

      <section class="login-panel" aria-labelledby="login-title">
        <div class="login-panel__inner">
          <div class="login-panel__mobile-brand">
            <img v-if="siteStore.site_logo" :src="siteStore.site_logo" alt="" />
            <div v-else class="login-panel__mobile-mark">{{ (siteStore.site_title || 'R')[0] }}</div>
            <span>{{ siteStore.site_title }}</span>
          </div>

          <header class="login-form-heading">
            <p>WELCOME BACK</p>
            <h2 id="login-title">登录工作空间</h2>
            <span>请输入账号信息，继续你的知识探索。</span>
          </header>

          <n-form ref="formRef" :model="form" :rules="rules" label-placement="top" @keyup.enter="handleLogin">
            <n-form-item path="username" label="账号">
              <n-input
                v-model:value="form.username"
                placeholder="请输入用户名"
                size="large"
                :input-props="{ autocomplete: 'username' }"
              >
                <template #prefix><n-icon><PersonOutline /></n-icon></template>
              </n-input>
            </n-form-item>
            <n-form-item path="password" label="密码">
              <n-input
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
            <n-button type="primary" block size="large" class="login-submit" :loading="loading" @click="handleLogin">
              <span>{{ loading ? '正在验证身份' : '安全登录' }}</span>
              <span v-if="!loading" class="login-submit__arrow">→</span>
            </n-button>
          </n-form>

          <div class="login-security-note">
            <span class="login-security-note__lock"></span>
            登录受组织权限保护，信息将被安全加密传输
          </div>
        </div>

        <footer v-if="siteStore.site_copyright" class="login-panel__footer">
          {{ siteStore.site_copyright }}
        </footer>
      </section>
    </section>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NForm, NFormItem, NInput, NButton, NIcon, useMessage } from 'naive-ui'
import { PersonOutline, LockClosedOutline } from '@vicons/ionicons5'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { defaultWorkspaceRoute } from '@/router/menus'

const router = useRouter()
const route = useRoute()
const message = useMessage()
const authStore = useAuthStore()
const siteStore = useSiteStore()

const formRef = ref(null)
const loading = ref(false)
const form = ref({ username: '', password: '' })
const rules = {
  username: { required: true, message: '请输入用户名', trigger: 'blur' },
  password: { required: true, message: '请输入密码', trigger: 'blur' },
}

async function handleLogin() {
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
    message.error(detail || '登录失败，请检查用户名或密码')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  --ink: #152240;
  --muted: #71809a;
  --line: #e5eaf2;
  min-height: 100%;
  position: relative;
  display: grid;
  place-items: center;
  overflow: hidden;
  background: #f4f7fb;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.login-page__glow { position: absolute; border-radius: 50%; filter: blur(4px); pointer-events: none; }
.login-page__glow--top { width: 34rem; height: 34rem; top: -22rem; right: -9rem; background: rgba(83, 140, 255, .16); }
.login-page__glow--bottom { width: 30rem; height: 30rem; left: -18rem; bottom: -20rem; background: rgba(65, 207, 199, .10); }

.login-shell {
  width: min(1120px, calc(100vw - 48px));
  min-height: min(720px, calc(100vh - 48px));
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.12fr) minmax(390px, .88fr);
  overflow: hidden;
  border: 1px solid rgba(207, 216, 231, .88);
  border-radius: 28px;
  background: white;
  box-shadow: 0 28px 80px rgba(30, 56, 103, .14), 0 3px 12px rgba(30, 56, 103, .05);
}

.login-hero { position: relative; overflow: hidden; color: white; background: radial-gradient(circle at 80% 23%, #3275df 0, #174aab 35%, #0d3178 76%, #09265f 100%); }
.login-hero::before { content: ''; position: absolute; width: 580px; height: 580px; left: -178px; bottom: -292px; border-radius: 50%; border: 1px solid rgba(211, 229, 255, .22); box-shadow: 0 0 0 52px rgba(211, 229, 255, .055), 0 0 0 105px rgba(211, 229, 255, .035); }
.login-hero::after { content: ''; position: absolute; inset: 0; opacity: .42; background: linear-gradient(126deg, transparent 46%, rgba(151, 204, 255, .15) 46.1%, transparent 46.4%), linear-gradient(126deg, transparent 57%, rgba(151, 204, 255, .08) 57.1%, transparent 57.4%); }
.login-hero__grid { position: absolute; inset: 0; opacity: .25; background-image: linear-gradient(rgba(207, 231, 255, .16) 1px, transparent 1px), linear-gradient(90deg, rgba(207, 231, 255, .16) 1px, transparent 1px); background-size: 52px 52px; mask-image: linear-gradient(to bottom, black, transparent 70%); }
.login-hero__content { position: relative; z-index: 1; display: flex; flex-direction: column; height: 100%; padding: 53px 56px; }
.login-hero__brand { display: flex; align-items: center; gap: 11px; font-size: 15px; font-weight: 650; letter-spacing: .01em; }
.login-hero__brand-logo { width: 31px; height: 31px; object-fit: cover; border-radius: 9px; }
.login-hero__mark { width: 31px; height: 31px; display: flex; align-items: end; justify-content: center; gap: 3px; padding-bottom: 8px; border: 1px solid rgba(255,255,255,.32); border-radius: 9px; background: rgba(255,255,255,.11); box-shadow: inset 0 1px rgba(255,255,255,.18); }
.login-hero__mark span { width: 4px; border-radius: 4px; background: #d9ecff; }.login-hero__mark span:nth-child(1) { height: 7px; }.login-hero__mark span:nth-child(2) { height: 13px; }.login-hero__mark span:nth-child(3) { height: 19px; }
.login-hero__headline { margin: auto 0; max-width: 410px; }
.login-hero__eyebrow { display: flex; align-items: center; gap: 9px; margin: 0 0 19px; color: #b9d5ff; font-size: 10px; font-weight: 700; letter-spacing: .16em; }.login-hero__eyebrow i { width: 24px; height: 1px; background: #91bcff; }
.login-hero h1 { margin: 0; color: #fff; font-size: clamp(32px, 3.3vw, 46px); font-weight: 650; letter-spacing: -.055em; line-height: 1.22; }.login-hero h1 em { font-style: normal; color: #a7ccff; }
.login-hero__description { max-width: 358px; margin: 24px 0 0; color: rgba(229, 240, 255, .78); font-size: 14px; line-height: 1.9; }
.login-hero__capabilities { display: flex; gap: 23px; }.login-hero__capabilities > div { display: flex; align-items: center; gap: 9px; }.login-hero__capabilities p { margin: 0; color: #dceaff; font-size: 12px; line-height: 1.35; }.login-hero__capabilities small { display: block; margin-top: 2px; color: #91b9ed; font-size: 9px; letter-spacing: .02em; }
.capability-icon { position: relative; width: 27px; height: 27px; flex: 0 0 auto; border: 1px solid rgba(189, 220, 255, .42); border-radius: 8px; }.capability-icon--search::before { content: ''; position: absolute; width: 8px; height: 8px; top: 7px; left: 7px; border: 1.5px solid #bedaff; border-radius: 50%; }.capability-icon--search::after { content: ''; position: absolute; width: 6px; height: 1.5px; left: 15px; top: 16px; transform: rotate(45deg); background: #bedaff; }.capability-icon--spark::before { content: '✦'; position: absolute; inset: 2px; color: #bedaff; font-size: 18px; text-align: center; }.capability-icon--shield::before { content: '⌁'; position: absolute; inset: 0; color: #bedaff; font-size: 21px; line-height: 24px; text-align: center; }
.login-hero__orbit { position: absolute; z-index: 0; border-radius: 50%; border: 1px solid rgba(173, 211, 255, .22); }.login-hero__orbit--one { width: 300px; height: 300px; right: -155px; top: 115px; }.login-hero__orbit--two { width: 215px; height: 215px; right: -92px; top: 157px; border-color: rgba(173, 211, 255, .15); }
.login-hero__status { position: absolute; z-index: 1; bottom: 24px; left: 56px; display: flex; align-items: center; gap: 7px; color: rgba(201, 224, 255, .7); font-size: 10px; letter-spacing: .06em; }.login-hero__status span { width: 6px; height: 6px; border-radius: 50%; background: #55d6bc; box-shadow: 0 0 0 4px rgba(85,214,188,.12); }

.login-panel { position: relative; display: flex; min-width: 0; background: rgba(255,255,255,.96); }.login-panel__inner { width: min(100%, 360px); margin: auto; padding: 56px 0; }.login-panel__mobile-brand { display: none; }
.login-form-heading { margin-bottom: 32px; }.login-form-heading p { margin: 0 0 10px; color: #4d82e8; font-size: 10px; font-weight: 750; letter-spacing: .14em; }.login-form-heading h2 { margin: 0; color: var(--ink); font-size: 28px; font-weight: 700; letter-spacing: -.045em; line-height: 1.25; }.login-form-heading span { display: block; margin-top: 12px; color: var(--muted); font-size: 13px; line-height: 1.6; }
.login-panel :deep(.n-form-item) { margin-bottom: 21px; }.login-panel :deep(.n-form-item-label) { padding-bottom: 8px; color: #45536b; font-size: 13px; font-weight: 620; }.login-panel :deep(.n-input) { --n-border: 1px solid #e0e6ef !important; --n-border-hover: 1px solid #8fb5f7 !important; --n-border-focus: 1px solid #4c84ec !important; --n-box-shadow-focus: 0 0 0 3px rgba(76, 132, 236, .11) !important; --n-color: #fbfcfe !important; border-radius: 10px; }.login-panel :deep(.n-input-wrapper) { padding-left: 13px; }.login-panel :deep(.n-input__prefix) { margin-right: 8px; color: #88a0c1; }.login-panel :deep(.n-input__input-el) { color: var(--ink); font-size: 14px; }.login-panel :deep(.n-input__placeholder) { color: #acb8c9; }
.login-submit { height: 47px; margin-top: 3px; border-radius: 10px; font-weight: 650; letter-spacing: .02em; box-shadow: 0 10px 18px rgba(54, 116, 220, .21); transition: transform .18s ease, box-shadow .18s ease; }.login-submit:hover { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(54, 116, 220, .28); }.login-submit__arrow { margin-left: 10px; font-size: 19px; font-weight: 400; line-height: 0; }
.login-security-note { display: flex; justify-content: center; align-items: center; gap: 7px; margin-top: 24px; color: #9ba8ba; font-size: 11px; }.login-security-note__lock { position: relative; width: 10px; height: 8px; border: 1.3px solid #a5b2c3; border-radius: 2px; }.login-security-note__lock::before { content: ''; position: absolute; left: 1px; bottom: 6px; width: 5px; height: 5px; border: 1.3px solid #a5b2c3; border-bottom: 0; border-radius: 5px 5px 0 0; }
.login-panel__footer { position: absolute; right: 32px; bottom: 25px; left: 32px; color: #a8b2c0; font-size: 11px; text-align: center; }

/* 登录页有独立的品牌视觉，仍需跟随全局深色模式，避免 Naive 控件变暗而页面面板保持纯白。 */
.dark .login-page { background: #0f1727; }
.dark .login-page__glow--top { background: rgba(63, 119, 230, .16); }
.dark .login-page__glow--bottom { background: rgba(51, 168, 163, .09); }
.dark .login-shell { border-color: rgba(67, 86, 117, .8); background: #172235; box-shadow: 0 28px 80px rgba(0, 0, 0, .32), 0 3px 12px rgba(0, 0, 0, .18); }
.dark .login-panel { background: #172235; }
.dark .login-form-heading h2 { color: #edf3fc; }
.dark .login-form-heading span { color: #9daec4; }
.dark .login-panel :deep(.n-form-item-label) { color: #c0cede; }
.dark .login-panel :deep(.n-input) { --n-border: 1px solid #3b4d67 !important; --n-border-hover: 1px solid #6796e9 !important; --n-border-focus: 1px solid #76a7f4 !important; --n-box-shadow-focus: 0 0 0 3px rgba(104, 157, 239, .16) !important; --n-color: #202d42 !important; --n-text-color: #e6eef8 !important; --n-placeholder-color: #7789a3 !important; }
.dark .login-panel :deep(.n-input__input-el) { color: #e6eef8; }
.dark .login-panel :deep(.n-input__placeholder) { color: #7789a3; }
.dark .login-panel :deep(.n-input__prefix) { color: #91a8c8; }
.dark .login-security-note { color: #8293aa; }
.dark .login-security-note__lock, .dark .login-security-note__lock::before { border-color: #8a9bb1; }
.dark .login-panel__mobile-brand { color: #dce8f8; }
.dark .login-panel__footer { color: #73849a; }

@media (max-width: 800px) { .login-page { display: block; overflow: auto; background: linear-gradient(155deg, #eef5ff, #f8fbff 50%, #f5f8fc); }.login-shell { width: 100%; min-height: 100%; display: block; border: 0; border-radius: 0; box-shadow: none; }.login-hero { display: none; }.login-panel { min-height: 100vh; padding: 28px 24px 68px; }.login-panel__inner { padding: 28px 0; }.login-panel__mobile-brand { display: flex; align-items: center; gap: 9px; margin-bottom: 64px; color: #203458; font-size: 15px; font-weight: 680; }.login-panel__mobile-brand img, .login-panel__mobile-mark { width: 30px; height: 30px; border-radius: 9px; object-fit: cover; }.login-panel__mobile-mark { display: grid; place-items: center; color: white; background: linear-gradient(135deg, #558ce9, #2460c3); font-size: 15px; }.login-panel__footer { bottom: 22px; } }
@media (max-width: 430px) { .login-panel { padding-right: 20px; padding-left: 20px; }.login-panel__mobile-brand { margin-bottom: 48px; }.login-form-heading h2 { font-size: 26px; }.login-panel__footer { right: 20px; left: 20px; }.login-security-note { font-size: 10px; white-space: nowrap; } }
</style>
