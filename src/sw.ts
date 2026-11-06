/// <reference lib="webworker" />
// Service worker: app-shell precache (offline, PRD 8.5) + COOP/COEP on same-origin responses so WASM threads
// work on GitHub Pages (ARCHITECTURE §13). Header injection follows coi-serviceworker.js (MIT).
import { precacheAndRoute } from 'workbox-precaching';

declare const self: ServiceWorkerGlobalScope & { __WB_MANIFEST: Array<{ url: string; revision: string | null }> };

self.addEventListener('install', () => void self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

function isolate(r: Response): Response {
  if (r.status === 0) return r; // opaque
  const h = new Headers(r.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'cross-origin');
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}

// Registered before precacheAndRoute so this handler answers first for same-origin requests.
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (new URL(req.url).origin !== self.location.origin) return; // model downloads: plain CORS fetch, untouched
  e.respondWith((async () => isolate((await caches.match(req, { ignoreSearch: true })) ?? (await fetch(req))))());
});

precacheAndRoute(self.__WB_MANIFEST);
