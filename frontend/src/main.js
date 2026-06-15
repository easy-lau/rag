import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
// 先引 highlight.js 主题，再引 main.css —— 让 main.css 里自定义的 .hljs 背景/暗色配色
// 覆盖 github 主题（否则代码块底色被 github.css 覆盖成白色，预览里看不出代码块）。
import 'highlight.js/styles/github.css'
import './assets/styles/main.css'

createApp(App).use(createPinia()).use(router).mount('#app')
