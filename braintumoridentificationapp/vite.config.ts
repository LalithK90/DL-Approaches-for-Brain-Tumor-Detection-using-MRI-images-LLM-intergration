/// <reference types="vitest" />

import legacy from '@vitejs/plugin-legacy'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import type { ClientRequest } from 'http'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    legacy()
  ],
  server: {
    cors: true,
    host: true,
    port: 5173,
    strictPort: false,
    proxy: {
      '/login': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        // Normalize Origin/Referer so back-end that strictly expects http://localhost:8100 accepts proxied requests
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq: ClientRequest) => {
            proxyReq.setHeader('origin', 'http://localhost:8100')
            proxyReq.setHeader('referer', 'http://localhost:8100/')
          })
        }
      },
      '/predict': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq: ClientRequest) => {
            proxyReq.setHeader('origin', 'http://localhost:8100')
            proxyReq.setHeader('referer', 'http://localhost:8100/')
          })
        }
      },
      '/chat': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq: ClientRequest) => {
            proxyReq.setHeader('origin', 'http://localhost:8100')
            proxyReq.setHeader('referer', 'http://localhost:8100/')
          })
        }
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  }
})
