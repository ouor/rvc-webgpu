// librosa-compatible stft / istft: hann (periodic), win_length = n_fft, center=True.
// pad_mode 'reflect' = librosa 0.9 behaviour that RVC/UVR were written against; tools/reference.py fixes the same
// mode (meta.stftPadMode), librosa >= 0.10 would default to 'constant'.
import { fftPlan } from './fft';

/** Complex spectrogram, row-major [bin][frame] like numpy's (bins, frames). */
export interface Spec { bins: number; frames: number; re: Float32Array; im: Float32Array }

const windows = new Map<number, Float64Array>();
export function hann(n: number): Float64Array {
  let w = windows.get(n);
  if (!w) {
    w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    windows.set(n, w);
  }
  return w;
}

export type PadMode = 'reflect' | 'constant';

/** numpy.pad index mapping for one reflection (pad < n), -1 for zero padding. */
export function padIndex(j: number, n: number, mode: PadMode): number {
  if (j >= 0 && j < n) return j;
  if (mode === 'constant') return -1;
  return j < 0 ? -j : 2 * (n - 1) - j;
}

export function stft(y: ArrayLike<number>, nFft: number, hop: number, mode: PadMode = 'reflect'): Spec {
  const pad = nFft >> 1;
  const frames = 1 + Math.floor((y.length + 2 * pad - nFft) / hop);
  const bins = (nFft >> 1) + 1;
  const re = new Float32Array(bins * frames), im = new Float32Array(bins * frames);
  const w = hann(nFft), plan = fftPlan(nFft);
  const buf = new Float64Array(2 * nFft), out = new Float64Array(2 * nFft);
  for (let t = 0; t < frames; t++) {
    const s = t * hop - pad;
    for (let i = 0; i < nFft; i++) {
      const j = padIndex(s + i, y.length, mode);
      buf[2 * i] = j >= 0 ? y[j] * w[i] : 0;
      buf[2 * i + 1] = 0;
    }
    plan.forward(buf, out);
    for (let k = 0; k < bins; k++) { re[k * frames + t] = out[2 * k]; im[k * frames + t] = out[2 * k + 1]; }
  }
  return { bins, frames, re, im };
}

/** Output length hop * (frames - 1), as librosa.istft(center=True, length=None). */
export function istft(spec: Spec, hop: number): Float32Array {
  const { bins, frames, re, im } = spec;
  const nFft = 2 * (bins - 1);
  const pad = nFft >> 1;
  const full = nFft + hop * (frames - 1);
  const y = new Float64Array(full), wss = new Float64Array(full);
  const w = hann(nFft), plan = fftPlan(nFft);
  const X = new Float64Array(2 * nFft), out = new Float64Array(2 * nFft);
  for (let t = 0; t < frames; t++) {
    for (let k = 0; k < bins; k++) { X[2 * k] = re[k * frames + t]; X[2 * k + 1] = im[k * frames + t]; }
    X[1] = 0; // numpy irfft ignores imag of DC and Nyquist
    X[2 * (bins - 1) + 1] = 0;
    for (let k = 1; k < bins - 1; k++) { X[2 * (nFft - k)] = X[2 * k]; X[2 * (nFft - k) + 1] = -X[2 * k + 1]; }
    plan.inverse(X, out);
    const s = t * hop;
    for (let i = 0; i < nFft; i++) { y[s + i] += out[2 * i] * w[i]; wss[s + i] += w[i] * w[i]; }
  }
  const len = hop * (frames - 1);
  const res = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const d = wss[pad + i];
    res[i] = d > 1.1754944e-38 ? y[pad + i] / d : y[pad + i];
  }
  return res;
}
