// Port of rvc-nano src/modules/pipeline.py Pipeline.pipeline / vc, split into stages so the F2 cache can reuse
// f0 and features (ARCHITECTURE §9.2-1: all RMVPE, then all HuBERT, then all synthesis; same result).
import type { ConvertParams, Mono } from '../../shared/types';
import { filtfilt, HPF48_A, HPF48_B } from '../../dsp/filtfilt';
import { changeRms } from '../../dsp/rms';
import { quantizeF0 } from './f0';
import { blendIndex, searchIndex } from './index-search';
import type { Feats } from './features';
import type { IvfIndex } from '../../formats/faiss';
import type { RvcNet } from './net';
import type { RunCtx } from '../ctx';
import { yieldToLoop } from '../ctx';

const SR = 16000, WINDOW = 160;
export const CHUNK_PROFILES = { normal: [1, 6, 38, 41], low: [1, 5, 30, 32] } as const; // x_pad, x_query, x_center, x_max

/** numpy.pad(mode='reflect') index mapping (handles pads longer than the signal). */
function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}
export function reflectPad(x: ArrayLike<number>, pad: number): Float32Array {
  const out = new Float32Array(x.length + 2 * pad);
  for (let i = 0; i < out.length; i++) out[i] = x[reflectIndex(i - pad, x.length)];
  return out;
}

/** rvc.py infer(): peak normalize to 0.95, then pipeline.py: signal.filtfilt(bh, ah, audio). */
export function preprocess16k(mono16k: Float32Array): Float64Array {
  let max = 0;
  for (let i = 0; i < mono16k.length; i++) max = Math.max(max, Math.abs(mono16k[i]));
  const scale = max / 0.95 > 1 ? 0.95 / max : 1;
  const x = scale === 1 ? mono16k : mono16k.map((v) => v / (max / 0.95));
  return filtfilt(HPF48_B, HPF48_A, x);
}

/** pipeline.py opt_ts: split points near quiet spots (sum of |x| over 160 samples, minimum within ±t_query). */
export function chunkPoints(audio: Float64Array, profile: ConvertParams['chunkProfile']): number[] {
  const [, xQuery, xCenter, xMax] = CHUNK_PROFILES[profile];
  const tQuery = SR * xQuery, tCenter = SR * xCenter, tMax = SR * xMax;
  const n = audio.length;
  if (n + WINDOW <= tMax) return [];
  const half = WINDOW / 2;
  const sum = new Float64Array(n);
  for (let j = 0; j < n; j++) { // audio_pad (reflect 80) [j : j + 160]
    let s = 0;
    for (let i = 0; i < WINDOW; i++) s += Math.abs(audio[reflectIndex(j + i - half, n)]);
    sum[j] = s;
  }
  const opt: number[] = [];
  for (let t = tCenter; t < n; t += tCenter) {
    const a = t - tQuery, b = Math.min(n, t + tQuery);
    let m = a;
    for (let i = a; i < b; i++) if (sum[i] < sum[m]) m = i;
    opt.push(m);
  }
  return opt;
}

export interface Chunk { start: number; end: number; f0Start: number; f0End: number } // on the padded signal / f0 frames

/** The (audio_pad slice, pitch slice) pairs of the pipeline.py loop + the final tail call. */
export function chunkPlan(paddedLen: number, opt: number[], profile: ConvertParams['chunkProfile']): Chunk[] {
  const tPad2 = 2 * SR * CHUNK_PROFILES[profile][0];
  const chunks: Chunk[] = [];
  let s = 0, t: number | undefined;
  for (const o of opt) {
    t = Math.floor(o / WINDOW) * WINDOW;
    chunks.push({ start: s, end: t + tPad2 + WINDOW, f0Start: s / WINDOW, f0End: Math.floor((t + tPad2) / WINDOW) });
    s = t;
  }
  const tail = t ?? 0;
  chunks.push({ start: tail, end: paddedLen, f0Start: tail / WINDOW, f0End: Infinity });
  return chunks;
}

/** Everything that depends only on the input audio and chunk profile (F2 cache, ARCHITECTURE §6.3 ConvertCache). */
export interface ConvertCache {
  profile: ConvertParams['chunkProfile'];
  audio: Float64Array; // filtered 16 kHz, unpadded (change_rms reference)
  padded: Float32Array; // reflect-padded by t_pad
  chunks: Chunk[];
  f0Hz?: Float64Array; // RMVPE over `padded`
  feats: Partial<Record<'v1' | 'v2', Feats[]>>;
  retrieved: Map<string, Float32Array[]>; // `${indexId}:${version}` -> npy per chunk
}

export function prepareConvert(mono16k: Float32Array, profile: ConvertParams['chunkProfile']): ConvertCache {
  const audio = preprocess16k(mono16k);
  const padded = reflectPad(audio, SR * CHUNK_PROFILES[profile][0]);
  return { profile, audio, padded, chunks: chunkPlan(padded.length, chunkPoints(audio, profile), profile), feats: {}, retrieved: new Map() };
}

/** Nearest x2 upsampling along time: F.interpolate(feats.permute(0,2,1), scale_factor=2). */
function upsample2(src: Float32Array, dim: number, take: number): Float32Array {
  const out = new Float32Array(take * dim);
  for (let t = 0; t < take; t++) out.set(src.subarray((t >> 1) * dim, ((t >> 1) + 1) * dim), t * dim);
  return out;
}

/** pipeline.py vc() after feature extraction: index, x2, protect, synthesis; returns audio1 untrimmed. */
export async function synthesizeChunk(
  chunkLen: number, feats: Feats, retrieved: Float32Array | undefined, pitch: { pitch: BigInt64Array; pitchf: Float32Array } | undefined,
  net: RvcNet, p: ConvertParams, rng: RunCtx['rng'],
): Promise<Float32Array> {
  const dim = feats.dim;
  const blended = retrieved && p.indexRate !== 0 ? blendIndex(feats, retrieved, p.indexRate) : feats;
  const T = Math.min(Math.floor(chunkLen / WINDOW), 2 * feats.frames);
  const phone = upsample2(blended.data, dim, T);
  let pitchArr: BigInt64Array | undefined, pitchf: Float32Array | undefined;
  if (pitch && net.info.f0) {
    pitchArr = pitch.pitch.slice(0, T);
    pitchf = pitch.pitchf.slice(0, T);
    if (p.protect < 0.5) {
      const orig = upsample2(feats.data, dim, T);
      for (let t = 0; t < T; t++) {
        const w = pitchf[t] < 1 ? p.protect : 1; // pitchff[pitchf > 0] = 1; pitchff[pitchf < 1] = protect
        if (w === 1) continue;
        for (let j = t * dim; j < (t + 1) * dim; j++) phone[j] = phone[j] * w + orig[j] * (1 - w);
      }
    }
  }
  return net.run({ phone, frames: T, pitch: pitchArr, pitchf, sid: p.sid }, rng);
}

/** Synthesis stage over all chunks + change_rms + 0.99 peak limit (pipeline.py tail; f32 kept, no int16). */
export async function synthesizeAll(
  cache: ConvertCache, net: RvcNet, p: ConvertParams, ctx: RunCtx, index?: { id: string; index: IvfIndex },
): Promise<Mono> {
  const version = net.info.version;
  const feats = cache.feats[version];
  if (!feats) throw new Error('features not extracted');
  const tgt = net.info.sr;
  const tPadTgt = tgt * CHUNK_PROFILES[cache.profile][0];
  const q = net.info.f0 && cache.f0Hz ? quantizeF0(cache.f0Hz.subarray(0, Math.floor(cache.padded.length / WINDOW)), p.f0UpKey) : undefined;
  let retrieved: Float32Array[] | undefined;
  if (index && p.indexRate !== 0) {
    const key = `${index.id}:${version}`;
    retrieved = cache.retrieved.get(key);
    if (!retrieved) {
      retrieved = [];
      for (const f of feats) { ctx.throwIfCancelled(); retrieved.push(searchIndex(f, index.index)); await yieldToLoop(); }
      cache.retrieved.set(key, retrieved);
    }
  }
  const parts: Float32Array[] = [];
  for (let c = 0; c < cache.chunks.length; c++) {
    ctx.throwIfCancelled();
    const ch = cache.chunks[c];
    const pitch = q && { pitch: q.pitch.subarray(ch.f0Start, Math.min(ch.f0End, q.pitch.length)), pitchf: q.pitchf.subarray(ch.f0Start, Math.min(ch.f0End, q.pitchf.length)) };
    const audio1 = await synthesizeChunk(ch.end - ch.start, feats[c], retrieved?.[c], pitch, net, p, ctx.rng);
    parts.push(audio1.slice(tPadTgt, audio1.length - tPadTgt));
    ctx.progress(c + 1, cache.chunks.length);
  }
  let out: Float32Array = new Float32Array(parts.reduce((a, x) => a + x.length, 0));
  let o = 0;
  for (const x of parts) { out.set(x, o); o += x.length; }
  if (p.rmsMixRate !== 1) out = changeRms(cache.audio, SR, out, tgt, p.rmsMixRate);
  let max = 0;
  for (let i = 0; i < out.length; i++) max = Math.max(max, Math.abs(out[i]));
  if (max / 0.99 > 1) for (let i = 0; i < out.length; i++) out[i] /= max / 0.99;
  return { sr: tgt, data: out };
}
