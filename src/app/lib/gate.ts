/**
 * Proxy whose methods (also on nested objects such as engine.cache) await `wait()` before calling through.
 * `skip` names top-level methods that run at once.
 */
export function gateCalls<T extends object>(target: T, wait: () => Promise<unknown>, skip: readonly string[] = []): T {
  return new Proxy(target, {
    get(o, k, r) {
      const v: unknown = Reflect.get(o, k, r);
      if (typeof k !== 'string' || skip.includes(k)) return v;
      if (typeof v === 'function') return async (...a: unknown[]) => { await wait(); return (v as (...x: unknown[]) => unknown).apply(o, a); };
      if (v && typeof v === 'object') return gateCalls(v, wait);
      return v;
    },
  });
}
