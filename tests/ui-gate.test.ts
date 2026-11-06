// Engine calls made before init settles wait for it (file dropped right after load).
import { describe, expect, it } from 'vitest';
import { gateCalls } from '../src/app/lib/gate';

describe('gateCalls', () => {
  it('holds methods and nested methods until the gate opens; skipped ones run at once', async () => {
    const log: string[] = [];
    let open!: () => void;
    const gate = new Promise<void>((r) => { open = r; });
    const api = gateCalls({ init: () => log.push('init'), load: async (x: number) => { log.push(`load ${x}`); return x * 2; }, cache: { list: () => log.push('list') } }, () => gate, ['init']);
    const loaded = api.load(2);
    void api.cache.list();
    api.init();
    await Promise.resolve();
    expect(log).toEqual(['init']);
    open();
    expect(await loaded).toBe(4);
    expect(log).toEqual(['init', 'load 2', 'list']);
  });
});
