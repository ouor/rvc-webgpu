// Pure formatting and estimate helpers for the UI.
import type { Backend, StageId } from '@/shared/types';
import type { T } from '../i18n';
import type { Options } from '../store';
import type { MixParams } from '@/audio/mixer';

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

/** DESIGN 9.5 size rule, the same on every screen (QA B17): "about 108MB (model 55MB + index 53MB)", or "about 55MB" without an index. */
export const fmtModelSize = (t: T, model: number, index?: number) =>
  index ? t('model.size', { total: fmtBytes(model + index), a: fmtBytes(model), b: fmtBytes(index) }) : t('model.sizeOnly', { a: fmtBytes(model) });

/** Bytes of an index: the file size when known, else the IVF-Flat vectors (fp32), which is what dominates the file. */
export const indexBytes = (x?: { bytes?: number; info?: { nVectors: number; dim: number } }) => x?.bytes ?? (x?.info ? x.info.nVectors * x.info.dim * 4 : undefined);

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

/** Filesystem-safe file name part (I-15: forbidden characters become "-", Korean and spaces stay). */
export const safe = (s: string) => s.replace(/[\u0000-\u001f\\/:*?"<>|]+/g, '-').trim() || 'untitled'; // eslint-disable-line no-control-regex -- control characters are not allowed in file names

export type StemKind = 'mix' | 'converted' | 'vocals' | 'inst';
const KIND: Record<StemKind, string> = { mix: 'cover', converted: 'vocals-converted', vocals: 'vocals-original', inst: 'instrumental' };
const fromTag = (sec: number) => `from-${String(Math.floor(sec / 60)).padStart(2, '0')}m${String(Math.floor(sec % 60)).padStart(2, '0')}s`;
/** I-15 / MX-6: {song}_{kind}_{model}[_from-01m05s].{ext}; the start tag marks a range result that is not padded to the original timing. */
export const exportName = (song: string, model: string, kind: StemKind, ext: string, rangeStart?: number) =>
  `${safe(song)}_${KIND[kind]}_${safe(model)}${rangeStart !== undefined ? `_${fromTag(rangeStart)}` : ''}.${ext}`;
export const zipName = (song: string, model: string) => `${safe(song)}_${safe(model)}.zip`;

export const fmtEta = (t: T, s: number) => (s < 60 ? t('eta.underMin') : t('eta.min', { n: Math.round(s / 60) }));
export const fmtLeft = (t: T, s: number) => (s < 60 ? t('eta.sec', { n: Math.max(1, Math.round(s / 5) * 5) }) : t('eta.minSec', { m: Math.floor(s / 60), s: Math.round(s % 60) }));

export const hostOf = (url: string) => { try { return new URL(url, location.href).host; } catch { return '?'; } };

/** Rerun: option fields that differ, in display order. */
const CHANGE_KEYS = ['key', 'indexRate', 'protect', 'rmsMixRate', 'aggressiveness', 'sid'] as const;
export const changedKeys = (a: Options, b: Options) => CHANGE_KEYS.filter((k) => a[k] !== b[k]);
const CHANGE_LABEL = { key: 'opt.keyShort', indexRate: 'opt.indexShort', protect: 'opt.protectShort', rmsMixRate: 'opt.rmsShort', aggressiveness: 'opt.aggShort', sid: 'opt.sidShort' } as const;
/** A/B label (DESIGN 9.12): the first changed option with this side's value, "+n more" for the rest; pitch when nothing differs. */
export function abLabel(t: T, mine: Options, other: Options) {
  const keys = changedKeys(other, mine);
  const k = keys[0] ?? 'key';
  const d = `${t(CHANGE_LABEL[k])} ${k === 'key' ? fmtSigned(mine[k]) : mine[k]}`;
  return keys.length > 1 ? t('res.abMore', { d, n: keys.length - 1 }) : d;
}
export const formatChanges = (t: T, a: Options, b: Options) =>
  changedKeys(a, b).map((k) => `${t(CHANGE_LABEL[k])} ${k === 'key' ? fmtSigned(a[k]) : a[k]} → ${k === 'key' ? fmtSigned(b[k]) : b[k]}`).join(' · ');

/** I-17: mute (also in the saved mix) and solo (listening only) as gains; -Infinity dB is silence in the mix graph. */
export const effectiveMix = (m: MixParams, mute: { vocal: boolean; inst: boolean }, solo?: 'vocal' | 'inst'): MixParams => ({
  ...m,
  vocalGainDb: mute.vocal || solo === 'inst' ? -Infinity : m.vocalGainDb,
  instGainDb: mute.inst || solo === 'vocal' ? -Infinity : m.instGainDb,
});
