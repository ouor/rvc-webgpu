// Engine API (ARCHITECTURE §6.2): job planning, stage execution, F2 reuse cache, cancellation.
// No DOM here; storage and fetch are injected so the same code runs in vitest (Node).
import { AppError, toAppError } from '../shared/errors';
import type {
  Backend, CacheEntry, Engine, IndexInfo, JobResult, JobSpec, Mono, OnProgress, RvcModelInfo, StageId, Stereo,
} from '../shared/types';
import { createPrng } from '../dsp/prng';
import { resample } from '../dsp/resample';
import { readFaissIndex, type IvfIndex } from '../formats/faiss';
import { readModelArchive } from '../formats/archive';
import { readOnnxInfo, type OnnxInfo } from '../formats/onnx-meta';
import { readRvcCheckpoint, type RvcCheckpoint } from '../formats/rvc-checkpoint';
import { configureOrt, pickBackend, probeGpu, releaseSession, type OrtSession } from '../models/ort';
import { createSessionFromParts, tensorMap } from '../models/loader';
import { loadRegistry, ModelRepo, type Opened, type PublicId } from '../models/registry';
import { MemoryStore, sha256Hex, type BlobStore } from '../models/store';
import { CancelFlag, type RunCtx } from './ctx';
import { checkVrParams, separateVR, type VrBandParams, type VrMeta } from './separate/vr';
import { extractF0 } from './rvc/f0';
import { extractFeatures } from './rvc/features';
import { createOnnxNet, createTemplateNet, describeOnnx, type RvcNet } from './rvc/net';
import { prepareConvert, synthesizeAll, type ConvertCache } from './rvc/pipeline';

export interface EngineDeps { store?: BlobStore; fetch?: typeof fetch }

type LoadedModel =
  | { kind: 'pth'; info: RvcModelInfo; ckpt: RvcCheckpoint; file: Blob; indexFile?: Blob }
  | { kind: 'onnx'; info: RvcModelInfo; bytes: Uint8Array; onnx: OnnxInfo; file: Blob; indexFile?: Blob };

const MAX_SECONDS = 20 * 60; // IN-3 hard limit
const idOf = async (b: Blob) => (await sha256Hex(b)).slice(0, 16);
const baseName = (n: string) => n.replace(/\.[^.]+$/, '');

export function createEngine(deps: EngineDeps = {}): Engine {
  const fetchFn = deps.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  let store: BlobStore = deps.store ?? new MemoryStore();
  let repo: ModelRepo | undefined;
  let backend: Backend = 'wasm';
  const cancel = new CancelFlag();
  let busy = false;

  const inputs = new Map<string, Stereo>();
  const models = new Map<string, LoadedModel>();
  const indexes = new Map<string, { info: IndexInfo; index: IvfIndex; file: Blob }>();
  let sep: { key: string; vocals: Stereo; instrumental: Stereo } | undefined;
  let conv: { key: string; cache: ConvertCache } | undefined;

  const needRepo = () => { if (!repo) throw new AppError('INTERNAL', 'engine.init() not called'); return repo; };
  const get = <T>(m: Map<string, T>, id: string | undefined, what: string): T => {
    const v = id ? m.get(id) : undefined;
    if (!v) throw new AppError('NOT_FOUND', `unknown ${what} ${id}`);
    return v;
  };

  /** Public model ids a job needs (CA-6: no separation model in voice mode). */
  function requiredIds(job: JobSpec): string[] {
    const m = get(models, job.modelId, 'model');
    const ids = ['contentvec'];
    if (m.info.f0) ids.push('rmvpe');
    if (job.mode === 'song') ids.push('uvr-hp-vocal');
    if (m.info.templateId) ids.push(m.info.templateId);
    return ids;
  }

  async function openPublic(id: PublicId, stage: StageId, on: OnProgress): Promise<{ s: OrtSession; opened: Opened; backend: Backend }> {
    on({ stage, done: 0, total: 1, phase: 'load' });
    const opened = await needRepo().open(id);
    const r = await createSessionFromParts({ graph: opened.graph, manifest: opened.manifest, pth: opened.pth!, tensors: tensorMap(opened.pth!.root) }, backend, 'CACHE_INTEGRITY');
    return { s: r.session, opened, backend: r.backend };
  }

  async function createNet(m: LoadedModel): Promise<RvcNet> {
    if (m.kind === 'onnx') return createOnnxNet(m.info, m.bytes, m.onnx, backend);
    const t = await needRepo().open(m.info.templateId!);
    return createTemplateNet(m.info, m.ckpt, t, backend);
  }

  function crop(a: Stereo, range?: { start: number; end: number }): Stereo {
    if (!range) return a;
    const s = Math.max(0, Math.floor(range.start * a.sr)), e = Math.min(a.l.length, Math.ceil(range.end * a.sr));
    if (e <= s) throw new AppError('INTERNAL', 'empty range');
    return { sr: a.sr, l: a.l.subarray(s, e), r: a.r.subarray(s, e) };
  }

  async function runOnce(job: JobSpec, profile: JobSpec['convert']['chunkProfile'], on: OnProgress): Promise<JobResult> {
    const input = get(inputs, job.inputId, 'input');
    const model = get(models, job.modelId, 'model');
    const index = job.indexId ? get(indexes, job.indexId, 'index') : undefined;
    const missing = (await needRepo().plan(requiredIds(job))).filter((d) => !d.cached);
    if (missing.length) throw new AppError('MODELS_MISSING', 'public models not downloaded', missing.map((d) => d.id).join(','), true);

    const audio = crop(input, job.range);
    const reused: StageId[] = [];
    const timings: Partial<Record<StageId, number>> = {};
    const rng = createPrng(job.convert.seed);
    const ctxFor = (stage: StageId, used: Backend): RunCtx => ({
      backend: used, rng, throwIfCancelled: () => cancel.check(),
      progress: (done, total) => on({ stage, done, total, phase: 'run', backend: used }),
    });
    const reuse = (stage: StageId) => { reused.push(stage); on({ stage, done: 1, total: 1, reused: true }); };
    const timed = async <T>(stage: StageId, f: () => Promise<T>) => { const t0 = performance.now(); try { return await f(); } finally { timings[stage] = performance.now() - t0; } };
    const rangeKey = job.range ? `${job.range.start}-${job.range.end}` : 'all';

    // separate
    let source: Stereo = audio;
    let sepKey = 'voice';
    if (job.mode === 'song') {
      sepKey = `${job.inputId}|${rangeKey}|${job.separate.modelId}|${job.separate.aggressiveness}`;
      if (sep?.key === sepKey) reuse('separate');
      else {
        sep = undefined;
        conv = undefined;
        await timed('separate', async () => {
          const { s, opened, backend: used } = await openPublic('uvr-hp-vocal', 'separate', on);
          try {
            const mp = opened.params as VrBandParams;
            checkVrParams(mp);
            const meta = opened.manifest.meta as unknown as VrMeta;
            const a44 = audio.sr === mp.sr ? audio : { sr: mp.sr, l: resample(audio.l, audio.sr, mp.sr), r: resample(audio.r, audio.sr, mp.sr) };
            const r = await separateVR(a44, s, mp, meta, { aggressiveness: job.separate.aggressiveness }, ctxFor('separate', used));
            sep = { key: sepKey, ...r };
          } finally { await releaseSession(s); }
        });
      }
      source = sep!.vocals;
    }

    // 16 kHz preprocessing + chunk plan (cached with f0/features)
    const convKey = `${job.inputId}|${rangeKey}|${sepKey}|${profile}`;
    if (conv?.key !== convKey) {
      const mono = new Float32Array(source.l.length);
      for (let i = 0; i < mono.length; i++) mono[i] = (source.l[i] + source.r[i]) / 2;
      conv = { key: convKey, cache: prepareConvert(resample(mono, source.sr, 16000), profile) };
    }
    const cache = conv.cache;

    // f0
    if (model.info.f0) {
      if (cache.f0Hz) reuse('f0');
      else await timed('f0', async () => {
        const { s, opened, backend: used } = await openPublic('rmvpe', 'f0', on);
        try { cache.f0Hz = await extractF0(cache.padded, s, ctxFor('f0', used), opened.manifest.meta as { threshold?: number; centsOffset?: number }); }
        finally { await releaseSession(s); }
      });
    }

    // features
    const v = model.info.version;
    if (cache.feats[v]) reuse('features');
    else await timed('features', async () => {
      const { s, backend: used } = await openPublic('contentvec', 'features', on);
      const ctx = ctxFor('features', used);
      try {
        const out = [];
        for (let c = 0; c < cache.chunks.length; c++) {
          ctx.throwIfCancelled();
          const ch = cache.chunks[c];
          out.push(await extractFeatures(cache.padded.slice(ch.start, ch.end), s, v));
          ctx.progress(c + 1, cache.chunks.length);
        }
        cache.feats[v] = out;
      } finally { await releaseSession(s); }
    });

    // convert
    const converted: Mono = await timed('convert', async () => {
      on({ stage: 'convert', done: 0, total: 1, phase: 'load' });
      const net = await createNet(model);
      try {
        return await synthesizeAll(cache, net, job.convert, ctxFor('convert', net.backend), index && { id: job.indexId!, index: index.index });
      } finally { await net.release(); }
    });
    return { vocals: sep?.vocals, instrumental: sep?.instrumental, converted, reused, timings };
  }

  async function loadIndexBlob(file: Blob): Promise<IndexInfo> {
    const index = await readFaissIndex(file);
    if (index.d !== 256 && index.d !== 768) throw new AppError('INDEX_DIM_MISMATCH', `index dimension ${index.d}`);
    const id = await idOf(file);
    const info: IndexInfo = { id, nVectors: index.nVectors, dim: index.d, nlist: index.nlist, nprobe: index.nprobe, downsampled: index.nVectors < index.ntotal ? index.nVectors : undefined };
    indexes.set(id, { info, index, file });
    return info;
  }

  async function loadModelBlob(file: Blob, name: string): Promise<RvcModelInfo> {
    if (/\.zip$/i.test(name)) {
      const a = await readModelArchive(file);
      const info = await loadModelBlob(a.model, a.model.name);
      if (a.index) {
        try {
          const ii = await loadIndexBlob(a.index);
          if (ii.dim === (info.version === 'v1' ? 256 : 768)) { info.indexId = ii.id; models.get(info.id)!.indexFile = a.index; }
        } catch { /* a bad index in a zip does not block the model; UI shows "no index" */ }
      }
      return info;
    }
    const id = await idOf(file);
    if (/\.onnx$/i.test(name)) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let onnx: OnnxInfo;
      try { onnx = readOnnxInfo(bytes); } catch (e) { throw new AppError('ONNX_UNRECOGNIZED', 'not an ONNX model', String(e)); }
      const d = describeOnnx(onnx);
      const info: RvcModelInfo = { id, name: baseName(name), source: 'onnx', version: d.version, f0: d.f0, sr: d.sr ?? 40000, srUnknown: !d.sr, nSpk: 1, bytes: file.size };
      models.set(id, { kind: 'onnx', info, bytes, onnx, file });
      return info;
    }
    const ckpt = await readRvcCheckpoint(file);
    const t = needRepo().templateFor(ckpt.signature);
    if (!t) throw new AppError('PTH_UNSUPPORTED_ARCH', 'no template for this model configuration', ckpt.signature);
    const root = ckpt.pth.root as Map<string, unknown>;
    const info: RvcModelInfo = {
      id, name: typeof root.get('model_name') === 'string' ? (root.get('model_name') as string) : baseName(name),
      source: 'pth', version: ckpt.version, f0: ckpt.f0, sr: ckpt.sr, nSpk: ckpt.nSpk, templateId: t.id, bytes: file.size, meta: ckpt.meta,
    };
    models.set(id, { kind: 'pth', info, ckpt, file });
    return info;
  }

  const myDir = (id: string) => `my-models/${id}`;

  const engine: Engine = {
    async init({ backend: pref, modelBaseUrl }) {
      if (!deps.store && typeof navigator !== 'undefined' && navigator.storage && 'getDirectory' in navigator.storage) {
        const { OpfsStore } = await import('../models/store');
        store = new OpfsStore();
      }
      const gpu = await probeGpu();
      backend = pickBackend(pref, gpu);
      const threads = configureOrt();
      repo = new ModelRepo(await loadRegistry(modelBaseUrl, fetchFn), store, modelBaseUrl, fetchFn);
      await repo.pruneOldRevisions().catch(() => {});
      const est = (await navigator.storage?.estimate?.().catch(() => undefined)) ?? {};
      const persisted = (await navigator.storage?.persisted?.().catch(() => false)) ?? false;
      return {
        backend, webgpu: gpu.webgpu, shaderF16: gpu.shaderF16,
        crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true, wasmThreads: threads,
        storage: { usage: est.usage ?? 0, quota: est.quota ?? 0, persisted },
      };
    },

    async setInput(audio, meta) {
      if (audio.l.length / audio.sr > MAX_SECONDS) throw new AppError('AUDIO_TOO_LONG', `input longer than ${MAX_SECONDS} s`);
      const id = `${meta.name}|${meta.size}|${meta.lastModified}|${audio.l.length}`;
      if (!inputs.has(id)) { inputs.clear(); sep = undefined; conv = undefined; } // one input at a time (memory)
      inputs.set(id, audio);
      return id;
    },

    loadRvcModel: (file) => loadModelBlob(file, file.name),

    async setOnnxModelSr(modelId, sr) {
      const m = get(models, modelId, 'model');
      if (m.kind !== 'onnx') throw new AppError('INTERNAL', 'sample rate is fixed for .pth models');
      m.info.sr = sr;
      m.info.srUnknown = false;
    },

    loadIndex: (file) => loadIndexBlob(file),

    async plan(job) { return needRepo().plan(requiredIds(job)); },

    async download(ids, on) {
      const r = needRepo();
      const files = [...new Set(ids.flatMap((id) => r.files(id)))];
      const todo = [];
      for (const f of files) if (!(await r.isCached(f))) todo.push(f);
      const total = todo.reduce((a, f) => a + r.reg.files[f].bytes, 0);
      if (!todo.length) return;
      await navigator.storage?.persist?.().catch(() => false);
      const est = await navigator.storage?.estimate?.().catch(() => undefined);
      if (est?.quota && est.quota - (est.usage ?? 0) < total) throw new AppError('STORAGE_QUOTA', 'not enough storage', `${total} bytes needed`, true);
      cancel.reset();
      let base = 0;
      for (const f of todo) {
        await r.download(f, {
          onBytes: (n) => on({ stage: 'download', done: base + n, total, file: f, phase: n >= r.reg.files[f].bytes ? 'verify' : 'run' }),
          throwIfCancelled: () => cancel.check(),
        });
        base += r.reg.files[f].bytes;
      }
      on({ stage: 'download', done: total, total });
    },

    async run(job, on) {
      if (busy) throw new AppError('BUSY', 'a job is already running');
      busy = true;
      cancel.reset();
      try {
        try {
          return await runOnce(job, job.convert.chunkProfile, on);
        } catch (e) {
          // §9.2-3: one automatic retry with the low-memory chunk profile
          if (e instanceof AppError && e.code === 'OOM' && job.convert.chunkProfile === 'normal') return await runOnce(job, 'low', on);
          throw e;
        }
      } catch (e) {
        throw toAppError(e);
      } finally {
        busy = false;
      }
    },

    async getStems() { return sep ? { vocals: sep.vocals, instrumental: sep.instrumental } : null; },

    cancel() { cancel.cancel(); },

    cache: {
      async list() {
        const r = needRepo();
        const ids = [...Object.keys(r.reg.models), ...Object.values(r.reg.rvcTemplates).map((t) => t.id)];
        const out: CacheEntry[] = [];
        for (const id of ids) {
          let bytes = 0;
          for (const f of r.files(id)) if (await r.isCached(f)) bytes += r.reg.files[f].bytes;
          if (bytes) out.push({ id, bytes, kind: 'public' });
        }
        const mine = new Map<string, number>();
        for (const f of await store.list('my-models/')) { const id = f.path.split('/')[1]; mine.set(id, (mine.get(id) ?? 0) + f.bytes); }
        for (const [id, bytes] of mine) out.push({ id, bytes, kind: 'my-model' });
        return out;
      },
      async remove(id) { // public model id or my-model id, as cache.list() returns them
        const r = needRepo();
        let files: string[];
        try { files = r.files(id); } catch { return store.remove(myDir(id)); }
        for (const f of files) await store.remove(`public/${r.reg.revision}/${f}`);
      },
      async clear() { await store.remove('public'); },
    },

    myModels: {
      async save(modelId, indexId) {
        const m = get(models, modelId, 'model');
        const idx = indexId ? get(indexes, indexId, 'index').file : m.indexFile;
        const dir = myDir(modelId);
        const stream = async function* (b: Blob) { const r = b.stream().getReader(); for (;;) { const { done, value } = await r.read(); if (done) return; yield value; } };
        await store.write(`${dir}/model.${m.info.source}`, stream(m.file));
        if (idx) await store.write(`${dir}/model.index`, stream(idx));
        else await store.remove(`${dir}/model.index`);
        const info = { ...m.info, indexId: idx ? (indexId ?? m.info.indexId) : undefined };
        await store.write(`${dir}/info.json`, [new TextEncoder().encode(JSON.stringify(info))]);
      },
      async list() {
        const out: RvcModelInfo[] = [];
        for (const f of await store.list('my-models/')) {
          if (!f.path.endsWith('/info.json')) continue;
          try { out.push(JSON.parse(await (await store.read(f.path)).text())); } catch { /* skip broken entry */ }
        }
        return out;
      },
      async open(id) {
        const dir = myDir(id);
        const saved = JSON.parse(await (await store.read(`${dir}/info.json`)).text()) as RvcModelInfo;
        const info = await loadModelBlob(await store.read(`${dir}/model.${saved.source}`), `${saved.name}.${saved.source}`);
        info.name = saved.name;
        if (saved.source === 'onnx' && !saved.srUnknown) { info.sr = saved.sr; info.srUnknown = false; }
        if (await store.has(`${dir}/model.index`)) {
          const f = await store.read(`${dir}/model.index`);
          info.indexId = (await loadIndexBlob(f)).id;
          models.get(info.id)!.indexFile = f;
        }
        return info;
      },
      remove: (id) => store.remove(myDir(id)),
    },
  };
  return engine;
}
