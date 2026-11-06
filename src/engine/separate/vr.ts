// UVR VR separation (ARCHITECTURE §11.1). Line-level reference: RVC WebUI 2.2.231006 AudioPre._path_audio_,
// uvr5_pack/utils.py inference(), lib_v5/spec_utils.py (combine_spectrograms, mirroring, fft_lp/hp_filter,
// cmb_spectrogram_to_wave). Band values come from the shipped 4band_v2.json, never hard-coded.
import type { Stereo } from '../../shared/types';
import { AppError } from '../../shared/errors';
import { resample } from '../../dsp/resample';
import { istft, stft, type Spec } from '../../dsp/stft';
import { f32, runSession, type OrtSession } from '../../models/ort';
import { yieldToLoop, type RunCtx } from '../ctx';
import type { VrMeta } from '../../models/contract';

export interface VrBand {
  sr: number; hl: number; n_fft: number; crop_start: number; crop_stop: number;
  lpf_start?: number; lpf_stop?: number; hpf_start?: number; hpf_stop?: number;
}
export interface VrBandParams { bins: number; band: Record<string, VrBand>; sr: number; pre_filter_start: number; pre_filter_stop: number }
export type { VrMeta };

export function checkVrParams(p: VrBandParams): void {
  const n = Object.keys(p.band ?? {}).length;
  if (!n || p.sr !== 44100 || !p.bins) throw new AppError('CACHE_INTEGRITY', 'invalid VR band params');
}

const SEG_SEC = 60, OVERLAP_SEC = 4; // §9.2-5: segment memory independent of song length
// WebGPU batches patches; ORT WASM throws a bare C++ exception on B>1 patches here (and gains nothing on CPU).
const batchFor = (b: RunCtx['backend']) => (b === 'webgpu' ? 4 : 1);

/** Complex [2 ch][rows][frames] spectrogram. */
interface CSpec { rows: number; frames: number; re: Float32Array[]; im: Float32Array[] }

interface Analysis { X: CSpec; highEnd: CSpec; highEndH: number }

/** AudioPre band loop + combine_spectrograms for one stereo segment. */
function analyze(l: Float32Array, r: Float32Array, mp: VrBandParams): Analysis {
  const n = Object.keys(mp.band).length;
  const waves: Record<number, [Float32Array, Float32Array]> = {};
  const specs: Record<number, [Spec, Spec]> = {};
  for (let d = n; d >= 1; d--) {
    const bp = mp.band[d];
    if (d === n) waves[d] = bp.sr === mp.sr ? [l, r] : [resample(l, mp.sr, bp.sr), resample(r, mp.sr, bp.sr)];
    else {
      const up = mp.band[d + 1].sr;
      waves[d] = up === bp.sr ? waves[d + 1] : [resample(waves[d + 1][0], up, bp.sr), resample(waves[d + 1][1], up, bp.sr)];
    }
    specs[d] = [stft(waves[d][0], bp.n_fft, bp.hl), stft(waves[d][1], bp.n_fft, bp.hl)];
  }
  const frames = Math.min(...Object.values(specs).map((s) => s[0].frames));
  const rows = mp.bins + 1;
  const X: CSpec = { rows, frames, re: [new Float32Array(rows * frames), new Float32Array(rows * frames)], im: [new Float32Array(rows * frames), new Float32Array(rows * frames)] };
  let off = 0;
  for (let d = 1; d <= n; d++) {
    const bp = mp.band[d];
    for (let c = 0; c < 2; c++) {
      const s = specs[d][c];
      for (let b = bp.crop_start; b < bp.crop_stop; b++) {
        const dst = (off + b - bp.crop_start) * frames, src = b * s.frames;
        X.re[c].set(s.re.subarray(src, src + frames), dst);
        X.im[c].set(s.im.subarray(src, src + frames), dst);
      }
    }
    off += bp.crop_stop - bp.crop_start;
  }
  if (off > mp.bins) throw new AppError('CACHE_INTEGRITY', 'VR params: too many bins');
  if (mp.pre_filter_start > 0) { // multi-band pre-filter (spec_utils.combine_spectrograms)
    let gp = 1;
    for (let b = mp.pre_filter_start + 1; b < mp.pre_filter_stop; b++) {
      const g = 10 ** ((-(b - mp.pre_filter_start) * (3.5 - gp)) / 20);
      gp = g;
      for (let c = 0; c < 2; c++) for (let t = 0; t < frames; t++) { X.re[c][b * frames + t] *= g; X.im[c][b * frames + t] *= g; }
    }
  }
  // high-end bins kept aside for "mirroring"
  const top = mp.band[n];
  const highEndH = top.n_fft / 2 - top.crop_stop + (mp.pre_filter_stop - mp.pre_filter_start);
  const s0 = top.n_fft / 2 - highEndH;
  const he: CSpec = { rows: highEndH, frames, re: [], im: [] };
  for (let c = 0; c < 2; c++) {
    const s = specs[n][c];
    const re = new Float32Array(highEndH * frames), im = new Float32Array(highEndH * frames);
    for (let b = 0; b < highEndH; b++) {
      re.set(s.re.subarray((s0 + b) * s.frames, (s0 + b) * s.frames + frames), b * frames);
      im.set(s.im.subarray((s0 + b) * s.frames, (s0 + b) * s.frames + frames), b * frames);
    }
    he.re.push(re); he.im.push(im);
  }
  return { X, highEnd: he, highEndH };
}

function maxMag(X: CSpec): number {
  let m = 0;
  for (let c = 0; c < 2; c++) for (let i = 0; i < X.re[c].length; i++) m = Math.max(m, Math.hypot(X.re[c][i], X.im[c][i]));
  return m;
}

/** utils.py inference(): zero padding (left 128), 512-frame patches every 256, batched. Returns mask [2][rows*frames]. */
async function predictMask(X: CSpec, coef: number, net: OrtSession, meta: VrMeta, ctx: RunCtx, onPatch: () => void): Promise<Float32Array[]> {
  const { rows, frames } = X;
  const { window: W, offset, roi } = meta;
  const nWin = Math.ceil(frames / roi);
  const padL = offset;
  const mask = [new Float32Array(rows * frames), new Float32Array(rows * frames)];
  const outName = net.outputNames.includes('mask') ? 'mask' : net.outputNames[0];
  const BATCH = batchFor(ctx.backend);
  for (let w0 = 0; w0 < nWin; w0 += BATCH) {
    ctx.throwIfCancelled();
    const B = Math.min(BATCH, nWin - w0);
    const x = new Float32Array(B * 2 * rows * W);
    for (let b = 0; b < B; b++) {
      const start = (w0 + b) * roi - padL; // first input frame of this patch in unpadded coordinates
      for (let c = 0; c < 2; c++) for (let k = 0; k < rows; k++) {
        const dst = ((b * 2 + c) * rows + k) * W, src = k * frames;
        for (let t = Math.max(0, -start); t < W && start + t < frames; t++) x[dst + t] = Math.hypot(X.re[c][src + start + t], X.im[c][src + start + t]) / coef;
      }
    }
    const r = await runSession(net, { [net.inputNames[0]]: f32(x, [B, 2, rows, W]) });
    const m = r[outName].data as Float32Array; // [B, 2, rows, roi]
    for (let b = 0; b < B; b++) {
      const t0 = (w0 + b) * roi;
      for (let c = 0; c < 2; c++) for (let k = 0; k < rows; k++) {
        const src = ((b * 2 + c) * rows + k) * roi, dst = k * frames;
        for (let t = 0; t < roi && t0 + t < frames; t++) mask[c][dst + t0 + t] = m[src + t];
      }
    }
    for (let b = 0; b < B; b++) onPatch();
  }
  return mask;
}

/** nets_61968KB forward(): mask^(1+v/3) below split_bin, mask^(1+v) above. Outside the graph so agg can change. */
export function applyAggressiveness(mask: Float32Array[], rows: number, frames: number, agg: number, splitBin: number): void {
  const v = agg / 100;
  for (const m of mask) for (let k = 0; k < rows; k++) {
    const e = k < splitBin ? 1 + v / 3 : 1 + v;
    for (let i = k * frames; i < (k + 1) * frames; i++) m[i] = m[i] ** e;
  }
}

/** spec_utils.mirroring("mirroring", spec_m, input_high_end, mp). */
function mirroring(spec: CSpec, he: CSpec, mp: VrBandParams): CSpec {
  const { frames } = spec;
  const h = he.rows, base = mp.pre_filter_start - 10 - h;
  const out: CSpec = { rows: h, frames, re: [], im: [] };
  for (let c = 0; c < 2; c++) {
    const re = new Float32Array(h * frames), im = new Float32Array(h * frames);
    for (let b = 0; b < h; b++) {
      const srcRow = base + (h - 1 - b); // np.flip along bins
      for (let t = 0; t < frames; t++) {
        const i = b * frames + t, si = srcRow * frames + t;
        const mirror = Math.hypot(spec.re[c][si], spec.im[c][si]);
        const hr = he.re[c][i], hi = he.im[c][i], hm = Math.hypot(hr, hi);
        if (hm <= mirror) { re[i] = hr; im[i] = hi; }
        else { const ph = Math.atan2(hi, hr); re[i] = mirror * Math.cos(ph); im[i] = mirror * Math.sin(ph); }
      }
    }
    out.re.push(re); out.im.push(im);
  }
  return out;
}

/** fft_lp_filter / fft_hp_filter on a [rows][frames] band spectrogram (linear ramp gains). */
function lpFilter(s: Spec, start: number, stop: number) {
  let g = 1;
  for (let b = start; b < stop; b++) { g -= 1 / (stop - start); scaleRow(s, b, g); }
  for (let b = stop; b < s.bins; b++) scaleRow(s, b, 0);
}
function hpFilter(s: Spec, start: number, stop: number) {
  let g = 1;
  for (let b = start; b > stop; b--) { g -= 1 / (start - stop); scaleRow(s, b, g); }
  for (let b = 0; b <= stop; b++) scaleRow(s, b, 0);
}
function scaleRow(s: Spec, b: number, g: number) {
  for (let t = b * s.frames; t < (b + 1) * s.frames; t++) { s.re[t] *= g; s.im[t] *= g; }
}

/** spec_utils.cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h, extra_bins), one channel. */
function toWave(spec: CSpec, c: number, mp: VrBandParams, extra: CSpec): Float32Array {
  const n = Object.keys(mp.band).length;
  const { frames } = spec;
  let off = 0;
  let wave: Float32Array | undefined;
  for (let d = 1; d <= n; d++) {
    const bp = mp.band[d];
    const bins = bp.n_fft / 2 + 1;
    const s: Spec = { bins, frames, re: new Float32Array(bins * frames), im: new Float32Array(bins * frames) };
    const h = bp.crop_stop - bp.crop_start;
    s.re.set(spec.re[c].subarray(off * frames, (off + h) * frames), bp.crop_start * frames);
    s.im.set(spec.im[c].subarray(off * frames, (off + h) * frames), bp.crop_start * frames);
    off += h;
    if (d === n) {
      const maxBin = bp.n_fft / 2;
      s.re.set(extra.re[c], (maxBin - extra.rows) * frames);
      s.im.set(extra.im[c], (maxBin - extra.rows) * frames);
      if (bp.hpf_start! > 0) hpFilter(s, bp.hpf_start!, bp.hpf_stop! - 1);
      const w = istft(s, bp.hl);
      if (!wave) wave = w; else for (let i = 0; i < w.length; i++) wave[i] += w[i];
    } else {
      const nextSr = mp.band[d + 1].sr;
      if (d === 1) {
        lpFilter(s, bp.lpf_start!, bp.lpf_stop!);
        wave = resample(istft(s, bp.hl), bp.sr, nextSr);
      } else {
        hpFilter(s, bp.hpf_start!, bp.hpf_stop! - 1);
        lpFilter(s, bp.lpf_start!, bp.lpf_stop!);
        const w = istft(s, bp.hl);
        for (let i = 0; i < w.length; i++) wave![i] += w[i];
        wave = resample(wave!, bp.sr, nextSr);
      }
    }
  }
  return wave!;
}

/** Splits mask x X into instrumental and vocal spectrograms (mask = instrumental, §11.1-7). */
function stems(X: CSpec, mask: Float32Array[], outputIs: VrMeta['outputIs']): { inst: CSpec; voc: CSpec } {
  const mk = (): CSpec => ({ rows: X.rows, frames: X.frames, re: [new Float32Array(X.re[0].length), new Float32Array(X.re[0].length)], im: [new Float32Array(X.re[0].length), new Float32Array(X.re[0].length)] });
  const a = mk(), b = mk(); // a = masked stem, b = remainder
  for (let c = 0; c < 2; c++) for (let i = 0; i < X.re[c].length; i++) {
    const m = mask[c][i];
    a.re[c][i] = m * X.re[c][i]; a.im[c][i] = m * X.im[c][i];
    b.re[c][i] = X.re[c][i] - a.re[c][i]; b.im[c][i] = X.im[c][i] - a.im[c][i];
  }
  return outputIs === 'instrumental' ? { inst: a, voc: b } : { inst: b, voc: a };
}

/** Segment plan: [start, end) samples with OVERLAP between neighbours. */
/**
 * ~60 s segments overlapping by >= 4 s. Starts are multiples of `unit` (top-band hop x patch roi), so every segment
 * sees the same STFT frame and patch grid as whole-song processing and the result matches it outside the overlaps.
 */
export function segments(n: number, sr: number, unit: number, segSec = SEG_SEC, ovSec = OVERLAP_SEC): [number, number][] {
  const seg = segSec * sr, ov = ovSec * sr;
  if (n <= seg) return [[0, n]];
  const step = Math.max(unit, Math.floor((seg - ov) / unit) * unit);
  const out: [number, number][] = [];
  for (let s = 0; ; s += step) {
    const e = Math.min(n, s + step + ov);
    out.push([s, e]);
    if (e === n) break;
  }
  return out;
}

export async function separateVR(
  audio: Stereo, net: OrtSession, mp: VrBandParams, meta: VrMeta, opts: { aggressiveness: number }, ctx: RunCtx,
): Promise<{ vocals: Stereo; instrumental: Stereo }> {
  if (audio.sr !== mp.sr) throw new AppError('INTERNAL', `VR expects ${mp.sr} Hz input, got ${audio.sr}`);
  const n = audio.l.length;
  const top = mp.band[Object.keys(mp.band).length];
  const segs = segments(n, audio.sr, top.hl * meta.roi);
  // pass 1: global coef = max |X| over the whole song (utils.py inference uses the full spectrogram)
  let coef = 0;
  for (const [s, e] of segs) {
    ctx.throwIfCancelled();
    coef = Math.max(coef, maxMag(analyze(audio.l.subarray(s, e), audio.r.subarray(s, e), mp).X));
    await yieldToLoop();
  }
  if (coef === 0) coef = 1; // silent input
  const out = { vl: new Float32Array(n), vr: new Float32Array(n), il: new Float32Array(n), ir: new Float32Array(n) };
  const ov = OVERLAP_SEC * audio.sr;
  const totalPatches = segs.reduce((a, [s, e]) => a + Math.ceil((Math.floor((e - s) / mp.band[Object.keys(mp.band).length].hl) + 1) / meta.roi), 0);
  let done = 0;
  for (let k = 0; k < segs.length; k++) {
    const [s, e] = segs[k];
    const A = analyze(audio.l.subarray(s, e), audio.r.subarray(s, e), mp);
    const mask = await predictMask(A.X, coef, net, meta, ctx, () => ctx.progress(++done, Math.max(done, totalPatches)));
    applyAggressiveness(mask, A.X.rows, A.X.frames, opts.aggressiveness, mp.band[1].crop_stop);
    const { inst, voc } = stems(A.X, mask, meta.outputIs);
    const wav = (spec: CSpec) => { const he = mirroring(spec, A.highEnd, mp); return [toWave(spec, 0, mp, he), toWave(spec, 1, mp, he)]; };
    const [il, ir] = wav(inst), [vl, vr] = wav(voc);
    // linear crossfade over the overlap: rising here + falling in the previous segment sums to 1
    const fadeIn = k > 0 ? ov : 0;
    const fadeOut = k < segs.length - 1 ? ov : 0;
    const len = e - s;
    for (let i = 0; i < len; i++) {
      let g = 1;
      if (i < fadeIn) g = i / fadeIn;
      if (i >= len - fadeOut) g = Math.min(g, (len - i) / fadeOut);
      const take = (w: Float32Array) => (i < w.length ? w[i] : 0);
      out.il[s + i] += g * take(il); out.ir[s + i] += g * take(ir);
      out.vl[s + i] += g * take(vl); out.vr[s + i] += g * take(vr);
    }
    await yieldToLoop();
  }
  return { vocals: { sr: audio.sr, l: out.vl, r: out.vr }, instrumental: { sr: audio.sr, l: out.il, r: out.ir } };
}
