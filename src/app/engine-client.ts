// The only worker boundary for the UI (ARCHITECTURE §4 rule 1). Wraps Comlink so callers pass plain callbacks.
import * as Comlink from 'comlink';
import { installErrorTransfer } from '../shared/rpc';
import type { Engine, JobSpec, OnProgress, Stereo } from '../shared/types';

export type EngineClient = Omit<Engine, 'cancel'> & {
  /** setInput / setStemsInput transfer the sample buffers to the worker (they are detached here afterwards). */
  setInput(audio: Stereo, meta: { name: string; size: number; lastModified: number }): Promise<string>;
  cancel(): Promise<void>;
  terminate(): void;
};

export function createEngineClient(): EngineClient {
  installErrorTransfer();
  const worker = new Worker(new URL('../worker/engine.worker.ts', import.meta.url), { type: 'module', name: 'engine' });
  const r = Comlink.wrap<Engine>(worker);
  // Comlink proxies nested objects at runtime; its types model them as promises, hence the casts.
  const cache = r.cache as unknown as Comlink.Remote<Engine['cache']>;
  const mine = r.myModels as unknown as Comlink.Remote<Engine['myModels']>;
  return {
    init: (o) => r.init(o),
    setInput: (audio, meta) => r.setInput(Comlink.transfer(audio, [audio.l.buffer, audio.r.buffer]), meta),
    setStemsInput: (v, i, meta) => r.setStemsInput(v, Comlink.transfer(i, [v.l.buffer, v.r.buffer, i.l.buffer, i.r.buffer]), meta),
    loadRvcModel: (f) => r.loadRvcModel(f),
    setOnnxModelSr: (id, sr) => r.setOnnxModelSr(id, sr),
    loadIndex: (f) => r.loadIndex(f),
    plan: (job: JobSpec) => r.plan(job),
    download: (ids, on: OnProgress) => r.download(ids, Comlink.proxy(on)),
    run: (job, on: OnProgress) => r.run(job, Comlink.proxy(on)),
    getStems: () => r.getStems(),
    cancel: () => r.cancel(),
    cache: {
      list: () => cache.list(), verify: (id) => cache.verify(id), importFiles: (files) => cache.importFiles(files),
      remove: (id) => cache.remove(id), clear: () => cache.clear(),
    },
    myModels: {
      save: (id, ix) => mine.save(id, ix), list: () => mine.list(),
      open: (id) => mine.open(id), rename: (id, name) => mine.rename(id, name), remove: (id) => mine.remove(id),
    },
    terminate: () => worker.terminate(),
  };
}
