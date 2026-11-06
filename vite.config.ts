/// <reference types="vitest/config" />
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve, normalize, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Same headers the service worker injects in production (src/sw.ts). Only WASM threads need them.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// Dev only: static dirs with Range. /models/ = $MODELS_DIR (default weights/dist, core dev A's artifacts, git-ignored),
// so VITE_MODEL_BASE_URL=/models/ works without uploading anything. /__dev/* feed debug.html (test audio, references).
const devDirs: Record<string, string> = {
  '/models': process.env.MODELS_DIR ?? 'weights/dist',
  '/__dev/external': 'external',
  '/__dev/reference': 'weights/reference',
};

function serveDirs(): Plugin {
  return {
    name: 'serve-dev-dirs',
    configureServer(server) {
      for (const [mount, dir] of Object.entries(devDirs)) {
        const root = resolve(import.meta.dirname, dir);
        server.middlewares.use(mount, (req, res, next) => {
          const file = normalize(resolve(root, '.' + decodeURIComponent((req.url ?? '/').split('?')[0])));
          if (!file.startsWith(root + sep) || !existsSync(file) || !statSync(file).isFile()) return next();
          const size = statSync(file).size;
          const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
          const start = m ? Number(m[1]) : 0;
          const end = m && m[2] ? Number(m[2]) : size - 1;
          res.setHeader('Accept-Ranges', 'bytes');
          res.setHeader('Content-Length', end - start + 1);
          res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
          if (m) {
            res.statusCode = 206;
            res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
          }
          createReadStream(file, { start, end }).pipe(res);
        });
      }
    },
  };
}

export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [
    react(),
    serveDirs(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.ts',
      injectRegister: false, // registered by src/coi.ts so the isolation reload logic stays in one place
      manifest: false,
      injectManifest: { globPatterns: ['**/*.{js,mjs,css,html,wasm,svg,png}'], maximumFileSizeToCacheInBytes: 64 * 1024 * 1024 },
      devOptions: { enabled: false },
    }),
  ],
  resolve: { alias: { '@': resolve(import.meta.dirname, 'src') } },
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  build: { target: 'es2022' },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    testTimeout: 120_000,
    alias: { 'onnxruntime-web/webgpu': 'onnxruntime-web' },
  },
});
