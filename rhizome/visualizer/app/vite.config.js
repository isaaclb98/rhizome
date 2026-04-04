import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/traverse': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/domains': {
        target: 'http://localhost:8000',
        timeout: 120_000,
      },
    },
  },
  build: {
    outDir: '/app/visualizer/static',
    emptyOutDir: true,
  },
});
