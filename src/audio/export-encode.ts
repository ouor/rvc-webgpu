// What the export worker does, as plain functions (testable without a worker).
import { Zip, ZipPassThrough } from 'fflate';
import type { Mono, Stereo } from '../shared/types';
import { encodeAudio, type AudioFormat } from './wav';

export type PlainAudio = Mono | Stereo;
export interface ZipFile { name: string; audio?: PlainAudio; scale?: number; text?: string }

/** `onProgress(done, total)` after each file (files, not bytes). */
async function zip(files: ZipFile[], fmt: AudioFormat, onProgress?: (done: number, total: number) => void): Promise<Blob> {
  const parts: Uint8Array[] = [];
  let fail: Error | undefined;
  const z = new Zip((err, chunk) => { if (err) fail = err; else parts.push(chunk); });
  for (const f of files) {
    // one file at a time: encode, store, drop (18.3 memory rule)
    const bytes = f.audio ? new Uint8Array(await (await encodeAudio(f.audio, fmt, f.scale ?? 1)).arrayBuffer()) : new TextEncoder().encode(f.text ?? '');
    const e = new ZipPassThrough(f.name);
    z.add(e);
    e.push(bytes, true);
    onProgress?.(files.indexOf(f) + 1, files.length);
  }
  z.end();
  if (fail) throw fail;
  return new Blob(parts as BlobPart[], { type: 'application/zip' });
}

export const exportApi = {
  encode: (audio: PlainAudio, fmt: AudioFormat, scale = 1) => encodeAudio(audio, fmt, scale),
  zip,
};
export type ExportApi = typeof exportApi;
