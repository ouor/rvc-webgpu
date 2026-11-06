// Engine API (ARCHITECTURE §6.2): job planning, stage execution, F2 reuse cache, cancellation.
// No DOM here; storage and fetch are injected so the same code runs in vitest (Node).
import { AppError, toAppError } from '../shared/errors';
import { PUBLIC_IDS } from '../shared/types';
import type { Backend, CacheEntry, Engine, ImportResult, JobResult, JobSpec, OnProgress, Stereo } from '../shared/types';
import { resample } from '../dsp/resample';
import { configureOrt, pickBackend, probeGpu, releaseSession, type OrtSession } from '../models/ort';
import { createSessionFromParts, tensorMap } from '../models/loader';
import { loadRegistry, ModelRepo, type Opened, type PublicId, type Registry } from '../models/registry';
import { MemoryStore, type BlobStore } from '../models/store';
import { MyModelStore } from '../models/my-models';
import { publicDir } from '../models/layout';
import { CancelFlag } from './ctx';
import { JobRunner, type Stages } from './run';
import { Library, type LoadedModel } from './library';
import { checkVrParams, separateVR, type VrBandParams } from './separate/vr';
import { checkMeta, checkRegistry } from '../models/contract';
import { extractF0 } from './rvc/f0';
import { extractFeatures, featDim } from './rvc/features';
import { createOnnxNet, createTemplateNet, type RvcNet } from './rvc/net';

/** Injection points for tests: storage, network, a registry in place of the bundled/fetched one, model stages. */
export interface EngineDeps { store?: BlobStore; fetch?: typeof fetch; registry?: Registry; stages?: Partial<Stages> }

const MAX_SECONDS = 20 * 60; // IN-3 hard limit
const MIN_SECONDS = 1; // shorter than the RMS window (0.5 s) and the 1 s chunk padding gives nothing usable (R1 M3)

export function createEngine(deps: EngineDeps = {}): Engine {
  const fetchFn = deps.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  let store: BlobStore = deps.store ?? new MemoryStore();
  let repo: ModelRepo | undefined;
  let backend: Backend = 'wasm';
  let active: CancelFlag | undefined; // the running job or download; one at a time, each with its own flag (R1 M5)

  const inputs = new Map<string, Stereo>();
  const lib = new Library((sig) => needRepo().templateFor(sig), () => !!active);
  const mine = new MyModelStore(() => store);

  const needRepo = () => { if (!repo) throw new AppError('INTERNAL', 'engine.init() not called'); return repo; };

  /** Public model ids a job needs (CA-6: no separation model in voice mode). */
  function requiredIds(job: JobSpec): string[] {
    const m = lib.model(job.modelId);
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
        const meta = checkMeta(PUBLIC_IDS.separator, opened.manifest.meta);
        const a = audio.sr === mp.sr ? audio : { sr: mp.sr, l: resample(audio.l, audio.sr, mp.sr), r: resample(audio.r, audio.sr, mp.sr) };
        return await separateVR(a, s, mp, meta, { aggressiveness }, ctx(used));
      } finally { await releaseSession(s); }
    },
    async f0(padded, ctx) {
      const { s, opened, backend: used } = await openPublic(PUBLIC_IDS.rmvpe);
      try { return await extractF0(padded, s, ctx(used), checkMeta(PUBLIC_IDS.rmvpe, opened.manifest.meta)); }
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
    net: (info) => createNet(lib.model(info.id)),
    ...deps.stages,
  };
  const runner = new JobRunner(stages);

  async function exclusive<T>(f: (flag: CancelFlag) => Promise<T>): Promise<T> {
    if (active) throw new AppError('BUSY', 'a job or download is already running');
    const flag = (active = new CancelFlag());
    try { return await f(flag); } finally { active = undefined; }
  }

  function runOnce(job: JobSpec, profile: JobSpec['convert']['chunkProfile'], on: OnProgress, cancel: CancelFlag): Promise<JobResult> {
    const input = inputs.get(job.inputId);
    if (!input) throw new AppError('NOT_FOUND', `unknown input ${job.inputId}`);
    const model = lib.model(job.modelId);
    const index = job.indexId ? lib.index(job.indexId) : undefined;
    return runner.run({ spec: job, audio: crop(input, job.range), model: model.info, index: index && { id: job.indexId!, index: index.index }, profile }, on, cancel);
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
      repo = new ModelRepo(deps.registry ? checkRegistry(deps.registry) : await loadRegistry(base, fetchFn), store, base, fetchFn);
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

    loadRvcModel: (file) => lib.loadModel(file, file.name),

    async setOnnxModelSr(modelId, sr) {
      const m = lib.model(modelId);
      if (m.kind !== 'onnx') throw new AppError('INTERNAL', 'sample rate is fixed for .pth models');
      m.info.sr = sr;
      m.info.srUnknown = false;
    },

    loadIndex: (file) => lib.loadIndex(file),

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
        const m = lib.model(modelId);
        const ix = indexId ? lib.index(indexId) : m.info.indexId ? lib.loadedIndex(m.info.indexId) : undefined;
        if (ix && ix.info.dim !== featDim(m.info.version)) throw new AppError('INDEX_DIM_MISMATCH', `index dimension ${ix.info.dim}`, undefined, true);
        await mine.save({ info: m.info, file: m.file }, ix && { info: ix.info, file: ix.file });
      },
      list: () => mine.list(),
      async rename(id, name) {
        const n = await mine.rename(id, name);
        const loaded = lib.loaded(id);
        if (loaded) loaded.info.name = n;
      },
      async open(id) {
        const f = await mine.files(id);
        const info = await lib.loadModel(f.model, `${f.saved.name}.${f.saved.source}`, { index: f.index, stored: id });
        info.name = f.saved.name;
        if (f.saved.source === 'onnx' && !f.saved.srUnknown) { info.sr = f.saved.sr; info.srUnknown = false; }
        await mine.touch(id, info);
        return info;
      },
      async remove(id) {
        await lib.detachStored(id); // its loaded OPFS Files would become unreadable
        await mine.remove(id);
      },
    },
  };
  return engine;
}
