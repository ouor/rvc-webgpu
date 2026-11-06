// Registers the service worker and reloads once so the page becomes crossOriginIsolated (ARCHITECTURE §13).
// Without isolation the app still works with single-threaded WASM.
const FLAG = 'web-rvc:coi-reloaded';

export async function ensureIsolation(): Promise<boolean> {
  if (globalThis.crossOriginIsolated) { sessionStorage.removeItem(FLAG); return true; }
  if (import.meta.env.DEV || !('serviceWorker' in navigator) || !window.isSecureContext) return false;
  try {
    const reg = await navigator.serviceWorker.register(import.meta.env.BASE_URL + 'sw.js', { type: 'module', scope: import.meta.env.BASE_URL });
    await navigator.serviceWorker.ready;
    if (!sessionStorage.getItem(FLAG) && (reg.active || navigator.serviceWorker.controller)) {
      sessionStorage.setItem(FLAG, '1'); // loop guard (coi-serviceworker's coiReloadedBySelf)
      location.reload();
    }
  } catch {
    // registration failed: stay non-isolated
  }
  return false;
}
