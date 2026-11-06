// One conversion job and the F2 reuse cache (ARCHITECTURE §6.3, §9.2). Model sessions sit behind `Stages`, so the
// orchestration (reuse keys, mode switches, retries) is testable without ONNX models.
import { AppError } from '../shared/errors';
import type { Backend, ConvertParams, JobResult, JobSpec, Mono, OnProgress, RvcModelInfo, StageId, Stereo } from '../shared/types';
import { createPrng } from '../dsp/prng';
import { resample } from '../dsp/resample';
import type { IvfIndex } from '../formats/faiss';
import type { CancelFlag, RunCtx } from './ctx';
import { featDim, type Feats } from './rvc/features';
import type { RvcNet } from './rvc/net';
import { prepareConvert, synthesizeAll, type ConvertCache } from './rvc/pipeline';

/** A stage calls this with the backend its session actually got (it may have fallen back to WASM). */
export type StageCtx = (used: Backend) => RunCtx;
export interface Stems { vocals: Stereo; instrumental: Stereo }

export interface Stages {
  separate(audio: Stereo, aggressiveness: number, ctx: StageCtx): Promise<Stems>;
  /** RMVPE f0 (Hz per 10 ms frame) of the padded 16 kHz signal. */
  f0(padded16k: Float32Array, ctx: StageCtx): Promise<Float64Array>;
  /** ContentVec features per chunk (the chunks are views into one buffer). */
  features(chunks: Float32Array[], version: 'v1' | 'v2', ctx: StageCtx): Promise<Feats[]>;
  net(model: RvcModelInfo): Promise<RvcNet>;
}

export interface Job {
  spec: JobSpec;
  audio: Stereo; // already cropped to spec.range
  model: RvcModelInfo;
  index?: { id: string; index: IvfIndex };
  profile: ConvertParams['chunkProfile'];
}

export class JobRunner {
  private sep?: Stems & { key: string };
  private conv?: { key: string; cache: ConvertCache };

  constructor(private stages: Stages) {}

  /** Forgets cached stage results (new input). */
  clear() { this.sep = this.conv = undefined; }

  /** Stems of the last separation. */
  get stems(): Stems | undefined { return this.sep && { vocals: this.sep.vocals, instrumental: this.sep.instrumental }; }

  async run({ spec, audio, model, index, profile }: Job, on: OnProgress, cancel: CancelFlag): Promise<JobResult> {
    const dim = featDim(model.version);
    if (index && index.index.d !== dim) {
      throw new AppError('INDEX_DIM_MISMATCH', `index dimension ${index.index.d}, ${model.version} model needs ${dim}`, undefined, true);
    }
    const reused: StageId[] = [];
    const timings: Partial<Record<StageId, number>> = {};
    const rng = createPrng(spec.convert.seed);
    const ctxFor = (stage: StageId): StageCtx => (used) => ({
      backend: used, rng, throwIfCancelled: () => cancel.check(),
      progress: (done, total) => on({ stage, done, total, phase: 'run', backend: used }),
    });
    const reuse = (stage: StageId) => { reused.push(stage); on({ stage, done: 1, total: 1, reused: true }); };
    const timed = async <T>(stage: StageId, f: () => Promise<T>) => {
      on({ stage, done: 0, total: 1, phase: 'load' });
      const t0 = performance.now();
      try { return await f(); } finally { timings[stage] = performance.now() - t0; }
    };
    const rangeKey = spec.range ? `${spec.range.start}-${spec.range.end}` : 'all';

    let source = audio;
    let sepKey = 'voice';
    let stems: Stems | undefined; // only song mode returns stems (R1 H2)
    if (spec.mode === 'song') {
      sepKey = `${spec.inputId}|${rangeKey}|${spec.separate.modelId}|${spec.separate.aggressiveness}`;
      if (this.sep?.key === sepKey) reuse('separate');
      else {
        this.sep = undefined;
        const r = await timed('separate', () => this.stages.separate(audio, spec.separate.aggressiveness, ctxFor('separate')));
        this.sep = { key: sepKey, ...r };
      }
      stems = this.stems!;
      source = stems.vocals;
    }

    // 16 kHz preprocessing + chunk plan; f0 and features are cached on it
    const convKey = `${spec.inputId}|${rangeKey}|${sepKey}|${profile}`;
    if (this.conv?.key !== convKey) {
      const mono = new Float32Array(source.l.length);
      for (let i = 0; i < mono.length; i++) mono[i] = (source.l[i] + source.r[i]) / 2;
      this.conv = { key: convKey, cache: prepareConvert(resample(mono, source.sr, 16000), profile) };
    }
    const cache = this.conv.cache;

    if (model.f0) {
      if (cache.f0Hz) reuse('f0');
      else cache.f0Hz = await timed('f0', () => this.stages.f0(cache.padded, ctxFor('f0')));
    }

    const v = model.version;
    if (cache.feats[v]) reuse('features');
    else {
      const chunks = cache.chunks.map((ch) => cache.padded.subarray(ch.start, ch.end));
      cache.feats[v] = await timed('features', () => this.stages.features(chunks, v, ctxFor('features')));
    }

    const converted: Mono = await timed('convert', async () => {
      const net = await this.stages.net(model);
      try { return await synthesizeAll(cache, net, spec.convert, ctxFor('convert')(net.backend), index); }
      finally { await net.release(); }
    });
    return { vocals: stems?.vocals, instrumental: stems?.instrumental, converted, reused, timings };
  }
}
