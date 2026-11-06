// librosa.feature.rms (center=True) and rvc-nano pipeline.py change_rms(). Pad mode: see stft.ts.
import { padIndex, type PadMode } from './pad';

export function rmsFrames(y: ArrayLike<number>, frameLength: number, hop: number, mode: PadMode = 'reflect'): Float64Array {
  const pad = Math.floor(frameLength / 2);
  const n = y.length;
  const frames = Math.max(0, 1 + Math.floor((n + 2 * pad - frameLength) / hop));
  const out = new Float64Array(frames);
  for (let f = 0; f < frames; f++) {
    const s = f * hop - pad;
    let acc = 0;
    for (let i = s; i < s + frameLength; i++) { const j = padIndex(i, n, mode); if (j >= 0) acc += y[j] * y[j]; }
    out[f] = Math.sqrt(acc / frameLength);
  }
  return out;
}

/** torch F.interpolate(mode='linear', align_corners=False) of a 1-D signal to `size` points. */
export function interpLinear(x: Float64Array, size: number): Float64Array {
  const out = new Float64Array(size);
  const n = x.length;
  const scale = n / size;
  for (let i = 0; i < size; i++) {
    const src = Math.max(0, (i + 0.5) * scale - 0.5);
    const i0 = Math.min(Math.floor(src), n - 1);
    const i1 = Math.min(i0 + 1, n - 1);
    const l = src - i0;
    out[i] = x[i0] * (1 - l) + x[i1] * l;
  }
  return out;
}

/** change_rms(data1, sr1, data2, sr2, rate): scales data2 in place toward data1's loudness envelope. */
export function changeRms(data1: ArrayLike<number>, sr1: number, data2: Float32Array, sr2: number, rate: number, mode: PadMode = 'reflect'): Float32Array {
  const half1 = Math.floor(sr1 / 2), half2 = Math.floor(sr2 / 2);
  const r1 = interpLinear(rmsFrames(data1, half1 * 2, half1, mode), data2.length);
  const r2 = interpLinear(rmsFrames(data2, half2 * 2, half2, mode), data2.length);
  for (let i = 0; i < data2.length; i++) data2[i] *= r1[i] ** (1 - rate) * Math.max(r2[i], 1e-6) ** (rate - 1);
  return data2;
}
