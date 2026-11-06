// Engine plumbing without models: tiny fake registry served by a fake fetch, MemoryStore.
import { describe, expect, it } from 'vitest';
import { createEngine } from '../src/engine/engine';
import { MemoryStore, sha256Hex } from '../src/models/store';
import type { Progress } from '../src/shared/types';
import { fakeFlatIndex, fakeRvcPth, SIG_V2_F0_40K } from './helpers';

const enc = (s: string) => new TextEncoder().encode(s);
const FILES: Record<string, Uint8Array> = {
  'rmvpe/rmvpe.onnx': enc('graph-bytes'), 'rmvpe/rmvpe.manifest.json': enc('{}'), 'rmvpe/rmvpe.pth': enc('w'.repeat(1000)),
  'contentvec/contentvec.onnx': enc('g2'), 'contentvec/contentvec.manifest.json': enc('{"a":1}'), 'contentvec/contentvec.pth': enc('x'.repeat(500)),
  'rvc/v2-f0-40k.onnx': enc('tpl'), 'rvc/v2-f0-40k.manifest.json': enc('{}'),
};

async function setup() {
  const files: Record<string, { bytes: number; sha256: string }> = {};
  for (const [p, b] of Object.entries(FILES)) files[p] = { bytes: b.length, sha256: await sha256Hex(b.slice().buffer) };
  const m = (d: string) => ({ graph: `${d}/${d}.onnx`, manifest: `${d}/${d}.manifest.json`, weights: `${d}/${d}.pth` });
  const registry = { format: 1, revision: 'r1', files, models: { rmvpe: m('rmvpe'), contentvec: m('contentvec') },
    rvcTemplates: { [SIG_V2_F0_40K]: { id: 'v2-f0-40k', graph: 'rvc/v2-f0-40k.onnx', manifest: 'rvc/v2-f0-40k.manifest.json' } } };
  const fetchFn = (async (url: string | URL) => {
    const path = new URL(String(url)).pathname.replace(/^\/models\//, '');
    const body = path === 'registry.json' ? enc(JSON.stringify(registry)) : FILES[path];
    return body ? new Response(body.slice()) : new Response(null, { status: 404 });
  }) as typeof fetch;
  const store = new MemoryStore();
  const engine = createEngine({ store, fetch: fetchFn, registry: registry as never });
  const env = await engine.init({ backend: 'wasm', modelBaseUrl: 'http://x/models/' });
  return { engine, store, env };
}

const OLD = '0123456789abcdef'; // a saved entry from before savedAt/lastUsed and the info.json version

describe('engine (no models)', () => {
  it('reports the registry revision', async () => {
    const { env } = await setup();
    expect(env.modelRevision).toBe('r1');
  });

  it('download reports per-item progress', async () => {
    const { engine } = await setup();
    const ev: Progress[] = [];
    await engine.download(['rmvpe', 'contentvec'], (p) => ev.push(p));
    const last = (id: string) => ev.filter((p) => p.item === id).at(-1)!;
    expect(last('rmvpe').itemDone).toBe(last('rmvpe').itemTotal);
    expect(last('rmvpe').itemTotal).toBe(11 + 2 + 1000);
    expect(last('contentvec').itemTotal).toBe(2 + 7 + 500);
    expect(ev.at(-1)!.done).toBe(ev.at(-1)!.total);
  });

  it('cache.list status, verify detects corruption, importFiles fills the cache', async () => {
    const { engine, store } = await setup();
    await engine.download(['rmvpe'], () => {});
    expect(await engine.cache.list()).toEqual([expect.objectContaining({ id: 'rmvpe', name: 'RMVPE', status: 'verified' })]);
    await store.write('public/r1/rmvpe/rmvpe.pth', [enc('v'.repeat(1000))]); // same size, wrong bytes
    expect(await engine.cache.verify('rmvpe')).toBe('corrupt');
    expect((await engine.cache.list())[0].status).toBe('incomplete'); // the bad file was deleted

    const r = await engine.cache.importFiles([
      new File([FILES['rmvpe/rmvpe.pth'].slice()], 'rmvpe.pth'),
      new File([enc('nope')], 'contentvec.pth'),
      new File([enc('?')], 'unrelated.bin'),
    ]);
    expect(r.map((x) => x.status)).toEqual(['imported', 'mismatch', 'unknown']);
    expect(await engine.cache.verify('rmvpe')).toBe('verified');
  });

  it('my-models: list by last use, rename, verify', async () => {
    const { engine, store } = await setup();
    const model = enc('pth-bytes');
    const id = (await sha256Hex(model.slice().buffer)).slice(0, 16);
    const info = (x: object) => [enc(JSON.stringify({ name: 'm', source: 'pth', version: 'v2', f0: true, sr: 40000, nSpk: 1, bytes: 9, ...x }))];
    await store.write(`my-models/${id}/model.pth`, [model]);
    await store.write(`my-models/${id}/info.json`, info({ id, savedAt: 1, lastUsed: 5 }));
    await store.write(`my-models/${OLD}/info.json`, info({ id: OLD })); // saved before savedAt/lastUsed existed
    expect((await engine.myModels.list()).map((m) => m.id)).toEqual([id, OLD]);
    await engine.myModels.rename(id, '  Kyubey  ');
    expect((await engine.myModels.list())[0].name).toBe('Kyubey');
    expect(await engine.cache.verify(id)).toBe('verified');
    expect(await engine.cache.verify(OLD)).toBe('incomplete');
    await expect(engine.myModels.rename('missing', 'x')).rejects.toMatchObject({ code: 'NOT_FOUND' });
  });

  // R1 H3 / R3 A1: MemoryStore has OPFS semantics (truncate on write, stale File snapshots), so these would fail
  // if save() rewrote a stored file from its own OPFS File.
  it('my-models: open, attach another index, save keeps the stored model intact', async () => {
    const { engine, store } = await setup();
    const pth = await fakeRvcPth();
    const m = await engine.loadRvcModel(new File([pth], 'k.pth'));
    const ix1 = await engine.loadIndex(new File([fakeFlatIndex(768, 4, 1)], 'a.index'));
    await engine.myModels.save(m.id, ix1.id);

    const opened = await engine.myModels.open(m.id);
    expect(opened.index?.nVectors).toBe(4);
    await engine.myModels.save(m.id); // same model, same index: rewrites only info.json
    const ix2 = await engine.loadIndex(new File([fakeFlatIndex(768, 6, 2)], 'b.index'));
    await engine.myModels.save(m.id, ix2.id);

    expect(await store.size(`my-models/${m.id}/model.pth`)).toBe(pth.size);
    expect(await engine.cache.verify(m.id)).toBe('verified');
    expect((await engine.myModels.list())[0].index?.nVectors).toBe(6);
    const again = await engine.myModels.open(m.id);
    expect(again.index?.id).toBe(ix2.id);
  });

  it('my-models: remove and save again while the opened model is loaded', async () => {
    const { engine } = await setup();
    const m = await engine.loadRvcModel(new File([await fakeRvcPth()], 'k.pth'));
    const ix = await engine.loadIndex(new File([fakeFlatIndex(768, 4)], 'a.index'));
    await engine.myModels.save(m.id, ix.id);
    await engine.myModels.open(m.id);
    await engine.myModels.remove(m.id);
    expect(await engine.myModels.list()).toEqual([]);
    await engine.myModels.save(m.id); // the loaded copy survives the removal
    expect(await engine.cache.verify(m.id)).toBe('verified');
  });

  it('MemoryStore: a read Blob goes stale after its file is rewritten (like OPFS)', async () => {
    const store = new MemoryStore();
    await store.write('a', [enc('hello')]);
    const b = await store.read('a');
    await store.write('a', [enc('x')]);
    await expect(b.text()).rejects.toMatchObject({ name: 'NotReadableError' });
  });
});
