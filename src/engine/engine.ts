// Engine API (ARCHITECTURE §6.2): job planning, stage execution, F2 reuse cache, cancellation.
// No DOM here; storage and fetch are injected so the same code runs in vitest (Node).
import { AppError, toAppError } from '../shared/errors';
import { PUBLIC_IDS } from '../shared/types';
import type { Backend, CacheEntry, Engine, ImportResult, IndexInfo, JobResult, JobSpec, OnProgress, RvcModelInfo, Stereo } from '../shared/types';
import { resample } from '../dsp/resample';
import { readFaissIndex, type IvfIndex } from '../formats/faiss';
import { readModelArchive } from '../formats/archive';
import { readOnnxInfo, type OnnxInfo } from '../formats/onnx-meta';
import { readRvcCheckpoint, type RvcCheckpoint } from '../formats/rvc-checkpoint';
import { configureOrt, pickBackend, probeGpu, releaseSession, type OrtSession } from '../models/ort';
import { createSessionFromParts, tensorMap } from '../models/loader';
import { loadRegistry, ModelRepo, type Opened, type PublicId, type Registry } from '../models/registry';
import { MemoryStore, type BlobStore } from '../models/store';
import { modelIdOf, MyModelStore } from '../models/my-models';
import { publicDir } from '../models/layout';
import { CancelFlag } from './ctx';
import { JobRunner, type Stages } from './run';
import { checkVrParams, separateVR, type VrBandParams, type VrMeta } from './separate/vr';
import { extractF0 } from './rvc/f0';
import { extractFeatures, featDim } from './rvc/features';
import { createOnnxNet, createTemplateNet, describeOnnx, type RvcNet } from './rvc/net';

/** Injection points for tests: storage, network, a registry in place of the bundled/fetched one, model stages. */
export interface EngineDeps { store?: BlobStore; fetch?: typeof fetch; registry?: Registry; stages?: Partial<Stages> }

// `stored` = my-model id when `file` is that saved model's OPFS file (see myModels.remove).
type LoadedModel =
  | { kind: 'pth'; info: RvcModelInfo; ckpt: RvcCheckpoint; file: Blob; indexFile?: Blob; stored?: string }
  | { kind: 'onnx'; info: RvcModelInfo; bytes: Uint8Array; onnx: OnnxInfo; file: Blob; indexFile?: Blob; stored?: string };

const MAX_SECONDS = 20 * 60; // IN-3 hard limit
const MIN_SECONDS = 1; // shorter than the RMS window (0.5 s) and the 1 s chunk padding gives nothing usable (R1 M3)
const idOf = modelIdOf;
const baseName = (n: string) => n.replace(/\.[^.]+$/, '');

export function createEngine(deps: EngineDeps = {}): Engine {
  const fetchFn = deps.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  let store: BlobStore = deps.store ?? new MemoryStore();
  let repo: ModelRepo | undefined;
  let backend: Backend = 'wasm';
  let active: CancelFlag | undefined; // the running job or download; one at a time, each with its own flag (R1 M5)

  const inputs = new Map<string, Stereo>();
  const models = new Map<string, LoadedModel>();
  const indexes = new Map<string, { info: IndexInfo; index: IvfIndex; file: Blob }>();
  const mine = new MyModelStore(() => store);

  const needRepo = () => { if (!repo) throw new AppError('INTERNAL', 'engine.init() not called'); return repo; };
  const get = <T>(m: Map<string, T>, id: string | undefined, what: string): T => {
    const v = id ? m.get(id) : undefined;
    if (!v) throw new AppError('NOT_FOUND', `unknown ${what} ${id}`);
    return v;
  };

  /** Public model ids a job needs (CA-6: no separation model in voice mode). */
  function requiredIds(job: JobSpec): string[] {
    const m = get(models, job.modelId, 'model');
    const ids: string[] = [PUBLIC_IDS.contentvec];
    if (m.info.f0) ids.push(PUBLIC_IDS.rmvpe);
    if (job.mode === 'song') ids.push(PUBLIC_IDS.separator);
    if (m.info.templateId) ids.push(m.info.templateId);
    return ids;
  }

  async function openPublic(id: PublicId): Promise<{ s: OrtSession; opened: Opened; backend: Backend }> {
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
    const s = range ? Math.max(0, Math.floor(range.start * a.sr)) : 0;
    const e = range ? Math.min(a.l.length, Math.ceil(range.end * a.sr)) : a.l.length;
    if (e - s < MIN_SECONDS * a.sr) throw new AppError('INTERNAL', `range shorter than ${MIN_SECONDS} s`, 'AUDIO_TOO_SHORT', true); // shortcut: no own ErrorCode until the UI has strings for it
    return range ? { sr: a.sr, l: a.l.subarray(s, e), r: a.r.subarray(s, e) } : a;
  }

  const stages: Stages = {
    async separate(audio, aggressiveness, ctx) {
      const { s, opened, backend: used } = await openPublic(PUBLIC_IDS.separator);
      try {
        const mp = opened.params as VrBandParams;
        checkVrParams(mp);
        const meta = opened.manifest.meta as unknown as VrMeta;
        const a = audio.sr === mp.sr ? audio : { sr: mp.sr, l: resample(audio.l, audio.sr, mp.sr), r: resample(audio.r, audio.sr, mp.sr) };
        return await separateVR(a, s, mp, meta, { aggressiveness }, ctx(used));
      } finally { await releaseSession(s); }
    },
    async f0(padded, ctx) {
      const { s, opened, backend: used } = await openPublic(PUBLIC_IDS.rmvpe);
      try { return await extractF0(padded, s, ctx(used), opened.manifest.meta as { threshold?: number; centsOffset?: number }); }
      finally { await releaseSession(s); }
    },
    async features(chunks, v, ctx) {
      const { s, backend: used } = await openPublic(PUBLIC_IDS.contentvec);
      const c = ctx(used);
      try {
        const out = [];
        for (let i = 0; i < chunks.length; i++) {
          c.throwIfCancelled();
          out.push(await extractFeatures(chunks[i].slice(), s, v)); // own buffer for the ORT tensor
          c.progress(i + 1, chunks.length);
        }
        return out;
      } finally { await releaseSession(s); }
    },
    net: (info) => createNet(get(models, info.id, 'model')),
    ...deps.stages,
  };
  const runner = new JobRunner(stages);

  async function exclusive<T>(f: (flag: CancelFlag) => Promise<T>): Promise<T> {
    if (active) throw new AppError('BUSY', 'a job or download is already running');
    const flag = (active = new CancelFlag());
    try { return await f(flag); } finally { active = undefined; }
  }

  /** One loaded model and one index at a time (R1 M4); a running job keeps its own until the next load. */
  const dropAll = (m: Map<string, unknown>) => { if (!active) m.clear(); };

  function runOnce(job: JobSpec, profile: JobSpec['convert']['chunkProfile'], on: OnProgress, cancel: CancelFlag): Promise<JobResult> {
    const input = get(inputs, job.inputId, 'input');
    const model = get(models, job.modelId, 'model');
    const index = job.indexId ? get(indexes, job.indexId, 'index') : undefined;
    return runner.run({ spec: job, audio: crop(input, job.range), model: model.info, index: index && { id: job.indexId!, index: index.index }, profile }, on, cancel);
  }

  async function loadIndexBlob(file: Blob): Promise<IndexInfo> {
    const index = await readFaissIndex(file); // rejects dimensions other than 256/768
    const id = await idOf(file);
    const info: IndexInfo = { id, nVectors: index.nVectors, dim: index.d as 256 | 768, nlist: index.nlist, nprobe: index.nprobe, downsampled: index.nVectors < index.ntotal ? index.nVectors : undefined };
    dropAll(indexes);
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
          if (ii.dim === featDim(info.version)) { info.indexId = ii.id; info.index = ii; models.get(info.id)!.indexFile = a.index; }
          else indexes.delete(ii.id);
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
      dropAll(models);
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
    dropAll(models);
    models.set(id, { kind: 'pth', info, ckpt, file });
    return info;
  }

  const engine: Engine = {
    async init({ backend: pref, modelBaseUrl }) {
      if (!deps.store && typeof navigator !== 'undefined' && navigator.storage && 'getDirectory' in navigator.storage) {
        const { OpfsStore } = await import('../models/store');
        store = new OpfsStore();
      }
      const gpu = await probeGpu();
      backend = pickBackend(pref, gpu);
      const threads = configureOrt();
      const base = new URL(modelBaseUrl, globalThis.location?.href).href; // '/models/' is relative to the app origin
      repo = new ModelRepo(deps.registry ?? (await loadRegistry(base, fetchFn)), store, base, fetchFn);
      await repo.pruneOldRevisions().catch(() => {});
      const est = (await navigator.storage?.estimate?.().catch(() => undefined)) ?? {};
      const persisted = (await navigator.storage?.persisted?.().catch(() => false)) ?? false;
      return {
        backend, webgpu: gpu.webgpu, shaderF16: gpu.shaderF16, gpuAdapter: gpu.adapter, modelRevision: repo.reg.revision,
        crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true, wasmThreads: threads,
        storage: { usage: est.usage ?? 0, quota: est.quota ?? 0, persisted },
      };
    },

    async setInput(audio, meta) {
      if (audio.l.length / audio.sr > MAX_SECONDS) throw new AppError('AUDIO_TOO_LONG', `input longer than ${MAX_SECONDS} s`);
      const id = `${meta.name}|${meta.size}|${meta.lastModified}|${audio.l.length}`;
      if (!inputs.has(id)) { inputs.clear(); runner.clear(); } // one input at a time (memory)
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

    download: (ids, on) => exclusive(async (cancel) => {
      const r = needRepo();
      const owner = new Map<string, string>(); // file -> first item that needs it
      for (const id of ids) for (const f of r.files(id)) if (!owner.has(f)) owner.set(f, id);
      const todo = [];
      for (const f of owner.keys()) if (!(await r.isCached(f))) todo.push(f);
      const bytesOf = (f: string) => r.reg.files[f].bytes;
      const total = todo.reduce((a, f) => a + bytesOf(f), 0);
      const itemTotal = new Map<string, number>(), itemBase = new Map<string, number>();
      for (const f of todo) itemTotal.set(owner.get(f)!, (itemTotal.get(owner.get(f)!) ?? 0) + bytesOf(f));
      if (!todo.length) return;
      await navigator.storage?.persist?.().catch(() => false);
      const est = await navigator.storage?.estimate?.().catch(() => undefined);
      if (est?.quota && est.quota - (est.usage ?? 0) < total) throw new AppError('STORAGE_QUOTA', 'not enough storage', `${total} bytes needed`, true);
      let base = 0;
      for (const f of todo) {
        const item = owner.get(f)!, ib = itemBase.get(item) ?? 0;
        await r.download(f, {
          onBytes: (n) => on({
            stage: 'download', done: base + n, total, file: f, phase: n >= bytesOf(f) ? 'verify' : 'run',
            item, itemDone: ib + n, itemTotal: itemTotal.get(item),
          }),
          throwIfCancelled: () => cancel.check(),
        });
        base += bytesOf(f);
        itemBase.set(item, ib + bytesOf(f));
      }
      on({ stage: 'download', done: total, total });
    }),

    run: (job, on) => exclusive(async (cancel) => {
      try {
        const missing = (await needRepo().plan(requiredIds(job))).filter((d) => !d.cached);
        if (missing.length) throw new AppError('MODELS_MISSING', 'public models not downloaded', missing.map((d) => d.id).join(','), true);
        try {
          return await runOnce(job, job.convert.chunkProfile, on, cancel);
        } catch (e) {
          // §9.2-3: one automatic retry with the low-memory chunk profile
          const a = toAppError(e); // JS allocation failures count too (R1 M2)
          if (a.code === 'OOM' && job.convert.chunkProfile === 'normal') return await runOnce(job, 'low', on, cancel);
          throw a;
        }
      } catch (e) {
        throw toAppError(e);
      }
    }),

    async getStems() { return runner.stems ?? null; },

    cancel() { active?.cancel(); },

    cache: {
      async list() {
        const r = needRepo();
        const out: CacheEntry[] = [];
        const pub = publicDir(r.reg.revision);
        const stored = new Map((await store.list(pub)).map((f) => [f.path, f.bytes]));
        for (const id of r.ids()) {
          let bytes = 0, complete = true;
          for (const f of r.files(id)) {
            const b = stored.get(pub + f) ?? 0;
            bytes += b;
            if (b !== r.reg.files[f].bytes) complete = false;
          }
          if (bytes) out.push({ id, name: r.label(id), bytes, kind: 'public', status: complete ? 'verified' : 'incomplete' });
        }
        for (const m of await mine.sizes()) out.push({ id: m.id, name: m.name, bytes: m.bytes, kind: 'my-model', status: m.complete ? 'verified' : 'incomplete' });
        return out;
      },
      async verify(id) {
        const r = needRepo();
        if (r.ids().includes(id)) return r.verify(id);
        return mine.verify(id);
      },
      async importFiles(files) {
        const r = needRepo();
        const out: ImportResult[] = [];
        for (const f of files) {
          const name = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
          out.push({ name: f.name, ...(await r.importFile(f, name)) });
        }
        return out;
      },
      async remove(id) { // public model id or my-model id, as cache.list() returns them
        const r = needRepo();
        if (r.ids().includes(id)) { for (const f of r.files(id)) await store.remove(publicDir(r.reg.revision) + f); }
        else await engine.myModels.remove(id);
      },
      async clear() { await store.remove('public'); },
    },

    myModels: {
      async save(modelId, indexId) {
        const m = get(models, modelId, 'model');
        const ix = indexId ? get(indexes, indexId, 'index') : m.info.indexId ? indexes.get(m.info.indexId) : undefined;
        if (ix && ix.info.dim !== featDim(m.info.version)) throw new AppError('INDEX_DIM_MISMATCH', `index dimension ${ix.info.dim}`, undefined, true);
        await mine.save({ info: m.info, file: m.file }, ix && { info: ix.info, file: ix.file });
      },
      list: () => mine.list(),
      async rename(id, name) {
        const n = await mine.rename(id, name);
        const loaded = models.get(id);
        if (loaded) loaded.info.name = n;
      },
      async open(id) {
        const f = await mine.files(id);
        const info = await loadModelBlob(f.model, `${f.saved.name}.${f.saved.source}`);
        const m = models.get(info.id)!;
        m.stored = id;
        info.name = f.saved.name;
        if (f.saved.source === 'onnx' && !f.saved.srUnknown) { info.sr = f.saved.sr; info.srUnknown = false; }
        const ii = f.index && (await loadIndexBlob(f.index));
        if (ii && ii.dim === featDim(info.version)) { info.index = ii; info.indexId = ii.id; m.indexFile = f.index; }
        else if (ii) indexes.delete(ii.id);
        await mine.touch(id, info);
        return info;
      },
      async remove(id) {
        // Loaded blobs that are this model's OPFS files would become unreadable: keep a memory copy first,
        // so the model still runs and can be saved again.
        for (const m of models.values()) {
          if (m.stored !== id) continue;
          m.file = new Blob([await m.file.arrayBuffer()]);
          if (m.indexFile) {
            const copy = new Blob([await m.indexFile.arrayBuffer()]);
            for (const ix of indexes.values()) if (ix.file === m.indexFile) ix.file = copy;
            m.indexFile = copy;
          }
          m.stored = undefined;
        }
        await mine.remove(id);
      },
    },
  };
  return engine;
}
