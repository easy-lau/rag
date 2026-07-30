import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 60000
})

// 请求拦截器：直接从 localStorage 读取 token，避免 import auth store 造成循环依赖
http.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

http.interceptors.response.use(
  res => res.config?.returnFullResponse ? res : res.data,
  err => {
    // token 失效/未授权：清除 token 并跳转登录页（用 window.location 避免循环依赖）
    if (err?.response?.status === 401) {
      localStorage.removeItem('token')
      if (window.location.pathname !== '/login') {
        window.location.assign('/login')
      }
    }
    // 让调用方自行处理错误，避免在模块加载时初始化 Naive UI 组件
    return Promise.reject(err)
  }
)

export default http
