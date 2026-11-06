// RMVPE pitch (rvc-nano src/lib/rmvpe.py infer_from_audio/decode) and pipeline.py get_f0 quantization.
import { f32, runSession, type OrtSession } from '../../models/ort';
import type { RunCtx } from '../ctx';

const SR = 16000, HOP = 160;
const SEG = 30 * SR, OVERLAP = 2 * SR; // ARCHITECTURE §9.2-4: 30 s pieces with 2 s context each side
export const RMVPE_THRESHOLD = 0.03;
const CENTS_OFFSET = 1997.3794084376191;

/** rmvpe.py RMVPE.decode: argmax-local weighted average over 9 bins -> Hz (0 = unvoiced). */
export function decodeHidden(hidden: Float32Array, frames: number, thred = RMVPE_THRESHOLD, centsOffset = CENTS_OFFSET): Float64Array {
  const f0 = new Float64Array(frames);
  for (let t = 0; t < frames; t++) {
    const row = t * 360;
    let c = 0, max = -Infinity;
    for (let k = 0; k < 360; k++) if (hidden[row + k] > max) { max = hidden[row + k]; c = k; }
    let ps = 0, ws = 0;
    for (let k = c - 4; k <= c + 4; k++) {
      if (k < 0 || k >= 360) continue; // zero padding in the reference
      const s = hidden[row + k];
      ps += s * (20 * k + centsOffset);
      ws += s;
    }
    if (max <= thred) { f0[t] = 0; continue; }
    const hz = 10 * 2 ** (ps / (ws + 1e-7) / 1200);
    f0[t] = hz === 10 ? 0 : hz;
  }
  return f0;
}

/** Hz at 100 fps for the whole (padded) signal; F = floor(N/160) + 1 frames, like one RMVPE call. */
export async function extractF0(audio: Float32Array, rmvpe: OrtSession, ctx: RunCtx, meta: { threshold?: number; centsOffset?: number } = {}): Promise<Float64Array> {
  const total = Math.floor(audio.length / HOP) + 1;
  const out = new Float64Array(total);
  const pieces = Math.max(1, Math.ceil(audio.length / SEG));
  const outName = rmvpe.outputNames.includes('hidden') ? 'hidden' : rmvpe.outputNames[0];
  for (let p = 0; p < pieces; p++) {
    ctx.throwIfCancelled();
    const s = p * SEG, e = Math.min(audio.length, s + SEG);
    const s0 = Math.max(0, s - OVERLAP), e0 = Math.min(audio.length, e + OVERLAP);
    const piece = audio.slice(s0, e0);
    const r = await runSession(rmvpe, { [rmvpe.inputNames[0]]: f32(piece, [1, piece.length]) });
    const hidden = r[outName].data as Float32Array;
    const frames = Math.floor(piece.length / HOP) + 1;
    const f0 = decodeHidden(hidden, frames, meta.threshold, meta.centsOffset);
    const g0 = s / HOP, g1 = p === pieces - 1 ? total : e / HOP; // global frames this piece owns
    for (let g = g0; g < g1; g++) out[g] = f0[g - s0 / HOP];
    ctx.progress(p + 1, pieces);
  }
  return out;
}

/** numpy rint: round half to even. */
const rint = (x: number) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

/** pipeline.py get_f0 tail: key shift, mel quantization to 1..255. pitchf is the shifted Hz curve. */
export function quantizeF0(f0Hz: ArrayLike<number>, upKey: number): { pitch: BigInt64Array; pitchf: Float32Array } {
  const melMin = 1127 * Math.log(1 + 50 / 700), melMax = 1127 * Math.log(1 + 1100 / 700);
  const k = 2 ** (upKey / 12);
  const pitch = new BigInt64Array(f0Hz.length), pitchf = new Float32Array(f0Hz.length);
  for (let i = 0; i < f0Hz.length; i++) {
    const hz = f0Hz[i] * k;
    pitchf[i] = hz;
    let mel = 1127 * Math.log(1 + hz / 700);
    if (mel > 0) mel = ((mel - melMin) * 254) / (melMax - melMin) + 1;
    if (mel <= 1) mel = 1;
    if (mel > 255) mel = 255;
    pitch[i] = BigInt(rint(mel));
  }
  return { pitch, pitchf };
}
