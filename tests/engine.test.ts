// Engine plumbing without models: tiny fake registry served by a fake fetch, MemoryStore.
import { describe, expect, it } from 'vitest';
import { createEngine, type EngineDeps } from '../src/engine/engine';
import type { Stages } from '../src/engine/run';
import { throttleProgress } from '../src/engine/ctx';
import { changeRms } from '../src/dsp/rms';
import { MemoryStore, sha256Hex } from '../src/models/store';
import { ModelRepo } from '../src/models/registry';
import { AppError } from '../src/shared/errors';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type JobSpec, type Stereo } from '../src/shared/types';
import type { Progress } from '../src/shared/types';
import { fakeFlatIndex, fakeRvcPth, SIG_V2_F0_40K } from './helpers';

const enc = (s: string) => new TextEncoder().encode(s);
const FILES: Record<string, Uint8Array> = {
  'rmvpe/rmvpe.onnx': enc('graph-bytes'), 'rmvpe/rmvpe.manifest.json': enc('{}'), 'rmvpe/rmvpe.pth': enc('w'.repeat(1000)),
  'contentvec/contentvec.onnx': enc('g2'), 'contentvec/contentvec.manifest.json': enc('{"a":1}'), 'contentvec/contentvec.pth': enc('x'.repeat(500)),
  'uvr/uvr.onnx': enc('g3'), 'uvr/uvr.manifest.json': enc('{}'), 'uvr/uvr.pth': enc('u'), 'uvr/uvr.params.json': enc('{}'),
  'rvc/v2-f0-40k.onnx': enc('tpl'), 'rvc/v2-f0-40k.manifest.json': enc('{}'),
};

async function setup(stages?: EngineDeps['stages']) {
  const files: Record<string, { bytes: number; sha256: string }> = {};
  for (const [p, b] of Object.entries(FILES)) files[p] = { bytes: b.length, sha256: await sha256Hex(b.slice().buffer) };
  const m = (d: string) => ({ graph: `${d}/${d}.onnx`, manifest: `${d}/${d}.manifest.json`, weights: `${d}/${d}.pth` });
  const registry = { format: 1, revision: 'r1', files, models: { rmvpe: m('rmvpe'), contentvec: m('contentvec'), 'uvr-hp-vocal': { ...m('uvr'), params: 'uvr/uvr.params.json' } },
    rvcTemplates: { [SIG_V2_F0_40K]: { id: 'v2-f0-40k', graph: 'rvc/v2-f0-40k.onnx', manifest: 'rvc/v2-f0-40k.manifest.json' } } };
  const fetchFn = (async (url: string | URL) => {
    const path = new URL(String(url)).pathname.replace(/^\/models\//, '');
    const body = path === 'registry.json' ? enc(JSON.stringify(registry)) : FILES[path];
    return body ? new Response(body.slice()) : new Response(null, { status: 404 });
  }) as typeof fetch;
  const store = new MemoryStore();
  const engine = createEngine({ store, fetch: fetchFn, registry: registry as never, stages });
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

  it('a verified public file is not re-hashed on load; unmarked (old) caches are hashed once (QA B6)', async () => {
    const store = new MemoryStore(), w = enc('w'.repeat(1000));
    const reg = { revision: 'r1', files: { 'a.pth': { bytes: 1000, sha256: await sha256Hex(w.slice().buffer) } } };
    const repo = new ModelRepo(reg as never, store, 'http://x/', (async () => new Response(w.slice())) as typeof fetch);
    await repo.download('a.pth', {});
    await store.write('public/r1/a.pth', [enc('v'.repeat(1000))]); // same size, wrong bytes
    expect(await (await repo.blob('a.pth')).text()).toBe('v'.repeat(1000)); // trusted: size + marker only
    await store.remove('public/r1/a.pth' + ModelRepo.MARK); // a cache from before the marker
    await expect(repo.blob('a.pth')).rejects.toMatchObject({ code: 'CACHE_INTEGRITY' });
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
    expect((await engine.myModels.list())[1].memo).toBe(''); // old entries have none (I-10)
    expect(await engine.myModels.setMemo(id, '  breathy, key +2 works best  ')).toBe('breathy, key +2 works best');
    await engine.myModels.rename(id, 'Kyubey 2');
    expect((await engine.myModels.list())[0]).toMatchObject({ name: 'Kyubey 2', memo: 'breathy, key +2 works best' });
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

// R1 H4: run() orchestration with fake stages (no ONNX): F2 reuse keys, mode switches, retry, busy, cancel.
describe('engine run (fake stages)', () => {
  const calls = { separate: 0, f0: 0, features: 0, net: 0 };
  let hook: { separate?: (ctx: Parameters<Stages['separate']>[2]) => Promise<void>; features?: () => void } = {};
  const fake: Stages = {
    async separate(audio, _agg, ctx) {
      calls.separate++;
      await hook.separate?.(ctx);
      return { vocals: audio, instrumental: { sr: audio.sr, l: audio.l.map((x) => -x), r: audio.r } };
    },
    async f0(padded) { calls.f0++; return new Float64Array(Math.floor(padded.length / 160) + 1).fill(220); },
    async features(chunks) {
      calls.features++;
      hook.features?.();
      return chunks.map((c) => {
        const frames = Math.floor(c.length / 320);
        return { data: new Float32Array(frames * 768).fill(0.1), frames, dim: 768 as const };
      });
    },
    async net(info) {
      calls.net++;
      return { info, backend: 'wasm', run: async (i) => new Float32Array(i.frames * 400).fill(0.1), release: async () => {} };
    },
  };

  async function ready() {
    for (const k of Object.keys(calls) as (keyof typeof calls)[]) calls[k] = 0;
    hook = {};
    const { engine } = await setup(fake);
    await engine.download(['rmvpe', 'contentvec', 'uvr-hp-vocal', 'v2-f0-40k'], () => {});
    const sr = 16000, n = 3 * sr;
    const l = new Float32Array(n).map((_, i) => Math.sin(i / 20) * 0.3);
    const audio: Stereo = { sr, l, r: l.slice() };
    const inputId = await engine.setInput(audio, { name: 'a.wav', size: 1, lastModified: 1 });
    const model = await engine.loadRvcModel(new File([await fakeRvcPth()], 'k.pth'));
    const job = (o: Partial<JobSpec> = {}, c: Partial<JobSpec['convert']> = {}): JobSpec =>
      ({ inputId, mode: 'song', separate: DEFAULT_SEPARATE, modelId: model.id, convert: { ...DEFAULT_CONVERT, ...c }, ...o });
    return { engine, job };
  }

  it('reuses separate/f0/features for the same job; key and aggressiveness changes redo the right stages', async () => {
    const { engine, job } = await ready();
    const first = await engine.run(job(), () => {});
    expect(first.reused).toEqual([]);
    expect(first.converted.data.every(Number.isFinite)).toBe(true);
    expect((await engine.run(job(), () => {})).reused).toEqual(['separate', 'f0', 'features']);
    expect((await engine.run(job({}, { f0UpKey: 5 }), () => {})).reused).toEqual(['separate', 'f0', 'features']);
    expect(calls).toMatchObject({ separate: 1, f0: 1, features: 1, net: 3 });
    expect((await engine.run(job({ separate: { ...DEFAULT_SEPARATE, aggressiveness: 3 } }), () => {})).reused).toEqual([]);
    expect(calls).toMatchObject({ separate: 2, f0: 2, features: 2 });
  });

  it('voice mode after song mode returns no stems (H2)', async () => {
    const { engine, job } = await ready();
    expect((await engine.run(job(), () => {})).instrumental).toBeDefined();
    const v = await engine.run(job({ mode: 'voice' }), () => {});
    expect(v.instrumental).toBeUndefined();
    expect(v.vocals).toBeUndefined();
  });

  it('stems mode: no separation, padded stems come back, no separator needed (I-20)', async () => {
    const { engine, job } = await ready();
    const sr = 16000, v = new Float32Array(3 * sr).fill(0.2), i = new Float32Array(2 * sr).fill(0.1);
    const inputId = await engine.setStemsInput({ sr, l: v, r: v.slice() }, { sr, l: i, r: i.slice() }, { name: 's', size: 1, lastModified: 1 });
    const ev: Progress[] = [];
    const r = await engine.run(job({ inputId, mode: 'stems' }), (p) => ev.push(p));
    expect(calls.separate).toBe(0);
    expect(ev.find((p) => p.stage === 'separate')).toMatchObject({ skipped: true });
    expect(r.instrumental!.l.length).toBe(3 * sr);
    expect(r.instrumental!.l[2 * sr]).toBe(0);
    expect((await engine.plan(job({ inputId, mode: 'stems' }))).map((d) => d.id)).not.toContain('uvr-hp-vocal');
    const plain = await engine.setInput({ sr, l: v, r: v.slice() }, { name: 'p', size: 1, lastModified: 1 });
    await expect(engine.run(job({ inputId: plain, mode: 'stems' }), () => {})).rejects.toMatchObject({ code: 'NOT_FOUND' });
  });

  it('rejects an index of the wrong dimension before separating (H1)', async () => {
    const { engine, job } = await ready();
    const ix = await engine.loadIndex(new File([fakeFlatIndex(256, 4)], 'v1.index'));
    await expect(engine.run(job({ indexId: ix.id }), () => {})).rejects.toMatchObject({ code: 'INDEX_DIM_MISMATCH' });
    expect(calls.separate).toBe(0);
  });

  it('rejects ranges under 1 s (M3)', async () => {
    const { engine, job } = await ready();
    await expect(engine.run(job({ range: { start: 1, end: 1.3 } }), () => {})).rejects.toMatchObject({ code: 'AUDIO_TOO_SHORT' });
  });

  it('retries once with the low chunk profile on OOM', async () => {
    const { engine, job } = await ready();
    let n = 0;
    hook.features = () => { if (n++ === 0) throw new AppError('OOM', 'out of memory'); };
    await engine.run(job(), () => {});
    expect(calls.features).toBe(2);
    hook.features = () => { throw new AppError('OOM', 'out of memory'); };
    await expect(engine.run(job({ mode: 'voice' }), () => {})).rejects.toMatchObject({ code: 'OOM' });
    expect(calls.features).toBe(4); // normal + exactly one low retry
  });

  it('BUSY while running, cancel() ends with CANCELLED', async () => {
    const { engine, job } = await ready();
    let release!: () => void;
    hook.separate = (ctx) => new Promise<void>((r) => { release = r; }).then(() => ctx('wasm').throwIfCancelled());
    const running = engine.run(job(), () => {});
    await new Promise((r) => setTimeout(r, 0));
    await expect(engine.run(job(), () => {})).rejects.toMatchObject({ code: 'BUSY' });
    engine.cancel();
    release();
    await expect(running).rejects.toMatchObject({ code: 'CANCELLED' });
    hook = {};
    await expect(engine.run(job(), () => {})).resolves.toBeDefined(); // the next run starts uncancelled
  });

  it('download and run exclude each other (M5)', async () => {
    const { engine, job } = await ready();
    let release!: () => void;
    hook.separate = () => new Promise<void>((r) => { release = r; });
    const running = engine.run(job(), () => {});
    await new Promise((r) => setTimeout(r, 0));
    await expect(engine.download(['rmvpe'], () => {})).rejects.toMatchObject({ code: 'BUSY' });
    release();
    await running;
  });

  it('keeps one loaded model and one index (M4)', async () => {
    const { engine } = await ready();
    const a = await engine.loadRvcModel(new File([await fakeRvcPth({ salt: 'a' })], 'a.pth'));
    const ia = await engine.loadIndex(new File([fakeFlatIndex(768, 4, 1)], 'a.index'));
    const b = await engine.loadRvcModel(new File([await fakeRvcPth({ salt: 'b' })], 'b.pth'));
    await engine.loadIndex(new File([fakeFlatIndex(768, 4, 2)], 'b.index'));
    await expect(engine.myModels.save(a.id)).rejects.toMatchObject({ code: 'NOT_FOUND' });
    await expect(engine.myModels.save(b.id, ia.id)).rejects.toMatchObject({ code: 'NOT_FOUND' });
  });

  it('MODELS_MISSING before download', async () => {
    const { engine } = await setup(fake);
    const id = await engine.setInput({ sr: 16000, l: new Float32Array(16000), r: new Float32Array(16000) }, { name: 'a', size: 1, lastModified: 1 });
    const m = await engine.loadRvcModel(new File([await fakeRvcPth()], 'k.pth'));
    await expect(engine.run({ inputId: id, mode: 'voice', separate: DEFAULT_SEPARATE, modelId: m.id, convert: DEFAULT_CONVERT }, () => {}))
      .rejects.toMatchObject({ code: 'MODELS_MISSING' });
  });
});

describe('throttleProgress (QA B1)', () => {
  it('passes step changes and the end at once, rate-limits the rest, delivers the last dropped event', async () => {
    const got: Progress[] = [];
    const on = throttleProgress((p) => got.push(p), 50);
    on({ stage: 'download', done: 0, total: 1000, file: 'a' });
    for (let i = 1; i < 500; i++) on({ stage: 'download', done: i, total: 1000, file: 'a' });
    on({ stage: 'download', done: 0, total: 10, file: 'b' }); // new file: immediate
    for (let i = 1; i < 9; i++) on({ stage: 'download', done: i, total: 10, file: 'b' });
    expect(got.map((p) => `${p.file}${p.done}`)).toEqual(['a0', 'b0']);
    await new Promise((r) => setTimeout(r, 80));
    expect(got.at(-1)).toMatchObject({ file: 'b', done: 8 }); // trailing
    on({ stage: 'download', done: 10, total: 10, file: 'b' });
    expect(got.at(-1)!.done).toBe(10);
  });
});

describe('dsp padding (M3)', () => {
  it('changeRms stays finite on 0.3 s input (pad longer than the signal)', () => {
    const a = new Float64Array(4800).map((_, i) => Math.sin(i / 7));
    const b = new Float32Array(12000).map((_, i) => Math.sin(i / 9) * 0.5);
    expect(changeRms(a, 16000, b, 40000, 0.25).every(Number.isFinite)).toBe(true);
  });
});
