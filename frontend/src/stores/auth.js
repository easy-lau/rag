import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { login as loginApi, getMe } from '@/api/auth'

const TOKEN_KEY = 'token'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem(TOKEN_KEY) || '')
  const user = ref(null)

  // 持久化 token；为空时清除 localStorage
  watch(token, val => {
    if (val) localStorage.setItem(TOKEN_KEY, val)
    else localStorage.removeItem(TOKEN_KEY)
  })

  const isAuthenticated = computed(() => !!token.value)
  const permissions = computed(() => user.value?.permissions || [])
  const menus = computed(() => user.value?.menus || [])

  // 超管恒为 true，否则按 permissions 判断
  function hasPerm(key) {
    if (user.value?.is_superadmin) return true
    return permissions.value.includes(key)
  }

  async function login(username, password) {
    const res = await loginApi({ username, password })
    token.value = res.access_token
    user.value = res.user
    return res.user
  }

  function logout() {
    token.value = ''
    user.value = null
    // 用 window.location 避免与 router 形成循环依赖
    if (window.location.pathname !== '/login') {
      window.location.assign('/login')
    }
  }

  async function fetchMe() {
    if (!token.value) return null
    try {
      user.value = await getMe()
      return user.value
    } catch (e) {
      // token 失效/无效：清理本地状态（不强制跳转，由调用方/守卫决定）
      token.value = ''
      user.value = null
      return null
    }
  }

  return {
    token, user, isAuthenticated, permissions, menus,
    hasPerm, login, logout, fetchMe,
  }
})
