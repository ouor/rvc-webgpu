// Export conform (ARCHITECTURE 18.3, I-14, QA B16): every stem 44.1 kHz stereo, same length and start, and one
// clipping factor for the whole stem batch. Main thread (OfflineAudioContext); encoding is in export.worker.ts.
import type { Mono, Stereo } from '../shared/types';

export const EXPORT_SR = 44100;
export type StemKind = 'converted' | 'vocals' | 'instrumental';
export interface ConformInput { converted: Mono; vocals?: Stereo; instrumental?: Stereo }
export interface ConformOpts {
  range?: { start: number; end: number }; // seconds of the source the result covers
  sourceSeconds?: number;                  // length of the whole source (for alignToSource)
  alignToSource?: boolean;                 // pad with silence so the stems line up with the whole source at 0 s
}
export interface Conformed { sr: number; frames: number; stems: Partial<Record<StemKind, Stereo>> & { converted: Stereo }; scale: number }
export type Resampler = (x: Float32Array, from: number, to: number) => Promise<Float32Array>;

/** The browser resampler, the same one that resamples the vocal for playback and renderMix. */
export const offlineResample: Resampler = async (x, from, to) => {
  const n = Math.ceil((x.length * to) / from);
  const ctx = new OfflineAudioContext(1, n, to);
  const b = ctx.createBuffer(1, x.length, from);
  b.copyToChannel(x as Float32Array<ArrayBuffer>, 0);
  const s = ctx.createBufferSource();
  s.buffer = b;
  s.connect(ctx.destination);
  s.start(0);
  return (await ctx.startRendering()).getChannelData(0);
};

/** 1, or 0.999 / max peak when any stem peaks above 1.0 (same factor for all, so relative levels stay). */
export function stemScale(stems: Stereo[]): number {
  let peak = 0;
  for (const s of stems) for (const ch of [s.l, s.r]) for (let i = 0; i < ch.length; i++) { const a = Math.abs(ch[i]); if (a > peak) peak = a; }
  return peak > 1 ? 0.999 / peak : 1;
}

export async function conformStems(r: ConformInput, o: ConformOpts = {}, resample: Resampler = offlineResample): Promise<Conformed> {
  const sr = EXPORT_SR;
  const conv = r.converted.sr === sr ? r.converted.data : await resample(r.converted.data, r.converted.sr, sr);
  const len = r.instrumental?.l.length ?? r.vocals?.l.length ?? conv.length; // the input range (B16: converted was 746 frames short)
  const head = o.alignToSource && o.range ? Math.round(o.range.start * sr) : 0;
  const total = o.alignToSource && o.sourceSeconds ? Math.max(head + len, Math.round(o.sourceSeconds * sr)) : head + len;
  const place = (x: Float32Array) => {
    if (!head && total === len && x.length === len) return x; // already conformed: no copy
    const y = new Float32Array(total);
    y.set(x.subarray(0, Math.min(len, x.length)), head);
    return y;
  };
  const stereo = (s: Stereo): Stereo => ({ sr, l: place(s.l), r: place(s.r) });
  const mono = place(conv);
  const stems: Conformed['stems'] = { converted: { sr, l: mono, r: mono } }; // mono copied to both channels (I-14)
  if (r.vocals) stems.vocals = stereo(r.vocals);
  if (r.instrumental) stems.instrumental = stereo(r.instrumental);
  return { sr, frames: total, stems, scale: stemScale(Object.values(stems)) };
}
