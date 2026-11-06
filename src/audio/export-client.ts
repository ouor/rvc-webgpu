// Main-thread handle to the export worker, created on first use (ENGINE_API section 5).
import * as Comlink from 'comlink';
import { installErrorTransfer } from '../shared/rpc';
import type { AudioFormat, PcmSource } from './wav';
import type { ExportApi, PlainAudio } from './export-encode';

/** AudioBuffer cannot be posted; copy its channels to plain arrays. */
const plain = (a: PcmSource): PlainAudio =>
  'getChannelData' in a
    ? (a.numberOfChannels === 1 ? { sr: a.sampleRate, data: a.getChannelData(0).slice() } : { sr: a.sampleRate, l: a.getChannelData(0).slice(), r: a.getChannelData(1).slice() })
    : a;

export interface ExportFile { name: string; audio?: PcmSource; scale?: number; text?: string }

export function createExporter() {
  let worker: Worker | undefined, api: Comlink.Remote<ExportApi> | undefined;
  const get = () => {
    if (!api) {
      installErrorTransfer();
      worker = new Worker(new URL('./export.worker.ts', import.meta.url), { type: 'module', name: 'export' });
      api = Comlink.wrap<ExportApi>(worker);
    }
    return api;
  };
  return {
    /** `scale` applies to 'pcm16' and 'mp3' (stems: Conformed.scale); 'f32' ignores it. */
    encode: (audio: PcmSource, fmt: AudioFormat, scale = 1): Promise<Blob> => get().encode(plain(audio), fmt, scale),
    /** STORE zip; each audio file is byte-identical to encode() with the same arguments. */
    zip: (files: ExportFile[], fmt: AudioFormat): Promise<Blob> => get().zip(files.map((f) => ({ ...f, audio: f.audio && plain(f.audio) })), fmt),
    terminate() { worker?.terminate(); worker = api = undefined; },
  };
}
export type Exporter = ReturnType<typeof createExporter>;
