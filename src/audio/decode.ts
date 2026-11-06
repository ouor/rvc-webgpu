// File -> 44.1 kHz stereo (ARCHITECTURE §1-2: decodeAudioData resamples to the context rate).
import { AppError } from '../shared/errors';
import type { Stereo } from '../shared/types';

export const PROCESS_SR = 44100;
export const WARN_SECONDS = 10 * 60; // IN-3
export const MAX_SECONDS = 20 * 60;

export interface Decoded {
  audio: Stereo; // copy owned by the caller; hand it to engine.setInput (it is transferred)
  buffer: AudioBuffer; // for preview playback
  duration: number;
  originalChannels: number;
  tooLongWarning: boolean;
}

export async function decodeFile(file: Blob): Promise<Decoded> {
  const ctx = new OfflineAudioContext(2, 1, PROCESS_SR);
  let buffer: AudioBuffer;
  try {
    buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch (e) {
    throw new AppError('AUDIO_DECODE', 'decodeAudioData failed', String(e));
  }
  if (buffer.duration > MAX_SECONDS) throw new AppError('AUDIO_TOO_LONG', `audio is ${Math.round(buffer.duration)} s`);
  const l = buffer.getChannelData(0).slice();
  const r = buffer.numberOfChannels > 1 ? buffer.getChannelData(1).slice() : l.slice();
  return { audio: { sr: buffer.sampleRate, l, r }, buffer, duration: buffer.duration, originalChannels: buffer.numberOfChannels, tooLongWarning: buffer.duration > WARN_SECONDS };
}

/** Min/max pairs per column for waveform drawing (DESIGN: ~1,000 points, wavesurfer `peaks`). */
export function peaks(data: Float32Array, points = 1000): Float32Array {
  const out = new Float32Array(points * 2);
  const step = data.length / points;
  for (let p = 0; p < points; p++) {
    let lo = 0, hi = 0;
    for (let i = Math.floor(p * step); i < Math.min(data.length, Math.floor((p + 1) * step)); i++) { const x = data[i]; if (x < lo) lo = x; if (x > hi) hi = x; }
    out[2 * p] = lo; out[2 * p + 1] = hi;
  }
  return out;
}

export const toAudioBuffer = (a: { sr: number; data?: Float32Array; l?: Float32Array; r?: Float32Array }): AudioBuffer => {
  const ch = a.data ? [a.data] : [a.l!, a.r!];
  const b = new AudioBuffer({ numberOfChannels: ch.length, length: ch[0].length, sampleRate: a.sr });
  ch.forEach((c, i) => b.copyToChannel(c as Float32Array<ArrayBuffer>, i));
  return b;
};
