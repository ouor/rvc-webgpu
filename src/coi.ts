// Registers the service worker and reloads once so the page becomes crossOriginIsolated (ARCHITECTURE §13).
// Without isolation the app still works with single-threaded WASM.
const FLAG = 'web-rvc:coi-reloaded';

export async function ensureIsolation(): Promise<boolean> {
  const isolated = globalThis.crossOriginIsolated;
  if (isolated) sessionStorage.removeItem(FLAG);
  if (import.meta.env.DEV || !('serviceWorker' in navigator) || !window.isSecureContext) return isolated;
  const register = () => navigator.serviceWorker.register(import.meta.env.BASE_URL + 'sw.js', { type: 'module', scope: import.meta.env.BASE_URL });
  // QA B14: hosts that send COOP/COEP themselves still need the worker for the offline cache (no reload needed)
  if (isolated) { register().catch(() => {}); return true; }
  try {
    const reg = await register();
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
