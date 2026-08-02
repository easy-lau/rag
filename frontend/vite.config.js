import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': resolve(__dirname, 'src') }
  },
  build: {
    rollupOptions: {
      output: {
        // 框架和 UI 库的更新频率远低于业务代码。显式分包可使浏览器长期缓存它们，
        // 也避免把所有依赖集中在入口 chunk，影响首次更新时的下载粒度。
        manualChunks(id) {
          const normalizedId = id.replaceAll(String.fromCharCode(92), '/')
          if (!normalizedId.includes('/node_modules/')) return

          if (
            normalizedId.includes('/node_modules/vue/') ||
            normalizedId.includes('/node_modules/vue-router/') ||
            normalizedId.includes('/node_modules/pinia/')
          ) {
            return 'vue-vendor'
          }
        }
      }
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('Accept-Encoding', 'identity')
          })
        }
      }
    }
  }
})
