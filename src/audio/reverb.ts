// Synthetic stereo reverb impulse response: exponentially decaying noise (PRD MX-4). Deterministic.

/** IR length in seconds (tooltip text, I-17: "short room reverb, about 2.2 s"). */
export const REVERB_SECONDS = 2.2;

export function createReverbIR(ctx: BaseAudioContext, seconds = REVERB_SECONDS, decay = 3.5): AudioBuffer {
  const n = Math.floor(seconds * ctx.sampleRate);
  const ir = ctx.createBuffer(2, n, ctx.sampleRate);
  let s = 12345;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2147483648 - 1; };
  const pre = Math.floor(0.012 * ctx.sampleRate); // ~12 ms pre-delay
  for (let c = 0; c < 2; c++) {
    const d = ir.getChannelData(c);
    for (let i = pre; i < n; i++) d[i] = rand() * (1 - (i - pre) / (n - pre)) ** decay;
  }
  return ir;
}
