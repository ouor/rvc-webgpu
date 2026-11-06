// WAV (PCM16 / float32) and MP3 encoding of mixes and stems (PRD MX-5).
import { AppError } from '../shared/errors';
import type { Mono, Stereo } from '../shared/types';

export type PcmSource = AudioBuffer | Mono | Stereo;
export type AudioFormat = 'pcm16' | 'f32' | 'mp3';

function channels(src: PcmSource): { sr: number; ch: Float32Array[] } {
  if ('getChannelData' in src) return { sr: src.sampleRate, ch: Array.from({ length: src.numberOfChannels }, (_, i) => src.getChannelData(i)) };
  if ('data' in src) return { sr: src.sr, ch: [src.data] };
  return { sr: src.sr, ch: [src.l, src.r] };
}

/** `scale` (pcm16 only): stem batch factor against clipping (I-14); f32 keeps the samples as they are. */
export function encodeWav(src: PcmSource, fmt: 'pcm16' | 'f32' = 'pcm16', scale = 1): Blob {
  const { sr, ch } = channels(src);
  const n = ch[0].length, nc = ch.length, bps = fmt === 'pcm16' ? 2 : 4;
  const data = n * nc * bps;
  const buf = new ArrayBuffer(44 + data);
  const v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + data, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, fmt === 'pcm16' ? 1 : 3, true); v.setUint16(22, nc, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * nc * bps, true); v.setUint16(32, nc * bps, true); v.setUint16(34, bps * 8, true);
  str(36, 'data'); v.setUint32(40, data, true);
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++, o += bps) {
    const x = ch[c][i];
    if (fmt === 'f32') v.setFloat32(o, x, true);
    else v.setInt16(o, Math.max(-32768, Math.min(32767, Math.round(x * scale * 32767))), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/** MP3 via mediabunny + its LAME WASM encoder, loaded only when first used (code-split). */
export async function encodeMp3(src: PcmSource, bitrate = 320_000, scale = 1): Promise<Blob> { // MX-5: 320 kbps CBR
  const { sr, ch } = channels(src);
  const mb = await import('mediabunny');
  if (!(await mb.canEncodeAudio('mp3'))) (await import('@mediabunny/mp3-encoder')).registerMp3Encoder();
  const output = new mb.Output({ format: new mb.Mp3OutputFormat(), target: new mb.BufferTarget() });
  const source = new mb.AudioSampleSource({ codec: 'mp3', bitrate });
  output.addAudioTrack(source);
  await output.start();
  const block = sr; // one second per sample keeps memory flat
  for (let s = 0; s < ch[0].length; s += block) {
    const len = Math.min(block, ch[0].length - s);
    const inter = new Float32Array(len * ch.length);
    for (let i = 0; i < len; i++) for (let c = 0; c < ch.length; c++) inter[i * ch.length + c] = ch[c][s + i] * scale;
    const sample = new mb.AudioSample({ data: inter, format: 'f32', numberOfChannels: ch.length, sampleRate: sr, timestamp: s / sr });
    await source.add(sample);
    sample.close();
  }
  await output.finalize();
  const out = (output.target as InstanceType<typeof mb.BufferTarget>).buffer;
  if (!out) throw new AppError('INTERNAL', 'mp3 encoder produced no data');
  return new Blob([out], { type: 'audio/mpeg' });
}

export const encodeAudio = (src: PcmSource, fmt: AudioFormat, scale = 1) =>
  (fmt === 'mp3' ? encodeMp3(src, undefined, scale) : Promise.resolve(encodeWav(src, fmt, scale)));
