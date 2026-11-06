// Pure formatting and estimate helpers for the UI.
import type { Backend, StageId } from '@/shared/types';

export const fmtTime = (s: number) => {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

/** "m:ss" or "m:ss.s" or plain seconds -> seconds, NaN when unreadable. */
export const parseTime = (v: string) => {
  const m = /^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$/.exec(v);
  return m ? Number(m[1] ?? 0) * 60 + Number(m[2]) : NaN;
};

export const fmtBytes = (b: number) => {
  if (b < 1e6) return b < 1e3 ? `${b}B` : `${Math.max(1, Math.round(b / 1e3))}KB`;
  if (b < 1e9) return `${Math.round(b / 1e6)}MB`;
  return `${(b / 1e9).toFixed(1)}GB`;
};

export const fmtHz = (sr: number) => `${+(sr / 1000).toFixed(1)}kHz`;
export const fmtSigned = (n: number) => (n > 0 ? `+${n}` : n < 0 ? `−${-n}` : '0');

/** Seconds of processing per second of audio, from ENGINE_NOTES measurements (217 s song, WebGPU / 20 s WASM). */
const RATE: Record<Backend, Record<Exclude<StageId, 'download'>, number>> = {
  webgpu: { separate: 0.52, f0: 0.02, features: 0.02, convert: 0.15 },
  wasm: { separate: 3.7, f0: 0.4, features: 0.4, convert: 1.3 },
};

export type RunStage = Exclude<StageId, 'download'>;
export const estimateStage = (stage: RunStage, seconds: number, backend: Backend) => RATE[backend][stage] * seconds + 3;

export function estimateRun(seconds: number, backend: Backend, stages: RunStage[]) {
  return stages.reduce((a, s) => a + estimateStage(s, seconds, backend), 0);
}

export const stripExt = (name: string) => name.replace(/\.[^.]+$/, '');
export const extOf = (name: string) => (/\.([^.]+)$/.exec(name)?.[1] ?? '').toLowerCase();

export const AUDIO_EXT = ['mp3', 'wav', 'flac', 'ogg', 'oga', 'opus', 'm4a', 'aac', 'mp4', 'webm'];
export const MODEL_EXT = ['pth', 'onnx', 'zip'];
export const INDEX_EXT = ['index'];
const accept = (ext: string[]) => ext.map((e) => `.${e}`).join(',');
export const AUDIO_ACCEPT = `audio/*,${accept(AUDIO_EXT)}`;
export const MODEL_ACCEPT = accept([...MODEL_EXT, ...INDEX_EXT]);
export const INDEX_ACCEPT = accept(INDEX_EXT);

export function download(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

/** Filesystem-safe file name part. */
export const safe = (s: string) => s.replace(/[\\/:*?"<>|]+/g, '_').trim();
