// UI state (zustand). This file only holds data and selectors; every write goes through a named action in controller.ts
// (eslint forbids set/get in components). Non-serialisable fields (AudioBuffer, Confirm.resolve) are kept out of persist by partialize.
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AppError } from '@/shared/errors';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type BackendPref, type CacheEntry, type DownloadItem, type EnvReport, type IndexInfo, type RvcModelInfo, type SavedModel, type RecentSummary } from '@/shared/types';
import type { AudioFormat } from '@/audio/wav';
import type { MixParams } from '@/audio/mixer';
import { DEFAULT_MIX } from '@/audio/mixer';
import type { RunStage } from './lib/format';

export type Lang = 'ko' | 'en';
export type Theme = 'system' | 'light' | 'dark';
export type SettingsTab = 'general' | 'storage' | 'models' | 'privacy' | 'about';

export const DEFAULT_BASE_URL: string = import.meta.env.VITE_MODEL_BASE_URL ?? '/models/';

export interface Settings {
  lang: Lang;
  theme: Theme;
  backend: BackendPref;
  modelBaseUrl?: string; // undefined = the deployment default (A3: a redeploy can move it)
  remember: boolean;
  noticeAck: boolean;
  advancedOpen: boolean;
  saveRecent: boolean; // I-2: keep finished results in recent work
}

export interface Options {
  mode: 'song' | 'voice' | 'stems';
  key: number;
  indexRate: number;
  protect: number;
  rmsMixRate: number;
  sid: number;
  aggressiveness: number;
  range?: { start: number; end: number };
}
const { f0UpKey, indexRate, protect, rmsMixRate, sid } = DEFAULT_CONVERT;
export const DEFAULT_OPTIONS: Options = { mode: 'song', key: f0UpKey, indexRate, protect, rmsMixRate, sid, aggressiveness: DEFAULT_SEPARATE.aggressiveness };

export interface Song {
  status: 'parsing' | 'ready' | 'error';
  name: string;
  size: number;
  lastModified?: number;
  duration?: number;
  channels?: number;
  sourceRate?: number;
  peaks?: Float32Array;
  buffer?: AudioBuffer;
  inputId?: string;
  error?: AppError;
}

export interface Model {
  status: 'parsing' | 'ready' | 'error';
  fileName: string;
  fromZip?: boolean;
  fromMine?: boolean;
  info?: RvcModelInfo;
  error?: AppError;
  saved?: boolean;
  saveState?: 'saving' | 'just' | 'quota' | 'failed'; // DESIGN 9.5 caption under the switch
}

export interface IndexState {
  status: 'parsing' | 'ready' | 'error';
  origin: 'user' | 'bundled'; // bundled = came with the zip / saved model and goes away with it
  fileName: string;
  bytes?: number; // file size of a user-added index
  id?: string; // set for a loaded index (also from a zip or a saved model)
  info?: IndexInfo;
  error?: AppError;
}

export type StepState = 'wait' | 'load' | 'run' | 'done' | 'reused' | 'skipped' | 'error' | 'cancelled';
export type Step = RunStage | 'mix';
export const STEPS: Step[] = ['separate', 'f0', 'features', 'convert', 'mix'];
export interface StepInfo { state: StepState; done: number; total: number; startedAt?: number; ms?: number }

export interface Job {
  status: 'running' | 'cancelling' | 'error';
  rerun: boolean;
  mode: Options['mode'];
  seconds: number; // processed audio length
  startedAt: number;
  steps: Record<Step, StepInfo>;
  current?: Step;
  fallback?: boolean; // a stage fell back from WebGPU to WASM
  error?: AppError;
  change?: { from: Options; to: Options }; // rerun: options before and after (title suffix, "key only" check)
}

export interface Download {
  status: 'confirm' | 'loading' | 'failed' | 'storage' | 'offline';
  open: boolean;
  items: DownloadItem[];
  done: number;
  total: number;
  got?: Record<string, number>; // per DownloadItem bytes so far (only items that started)
  cur?: string; // item being fetched
  startedAt?: number;
  error?: AppError;
}

export interface Result {
  songName: string;
  modelName: string;
  mode: Options['mode'];
  opts: Options;
  hasIndex: boolean;
  converted: AudioBuffer; // model SR
  vocals?: AudioBuffer;
  inst?: AudioBuffer;
  ms: number;
  seq: number;
  savedOut?: boolean; // mix or a file was downloaded/shared (I-1)
  recentId?: string; // I-2: stored in recent work
  recentState?: 'saving' | 'saved' | 'off' | 'quota' | 'failed'; // DESIGN 9.9 save line
  restored?: { createdAt: number; modelId: string }; // opened from recent work (no source, no rerun)
}

/** `alt`: an optional second action; resolve gets 'alt' for it, false for cancel. */
export interface Confirm { title: string; body: string; action: string; alt?: string; destructive?: boolean; resolve: (ok: boolean | 'alt') => void }

export interface State {
  settings: Settings;
  opts: Options;
  env?: EnvReport;
  envError?: AppError;
  engineReady: boolean;
  online: boolean;
  crashed: boolean;
  dismissed: string[];
  song?: Song;
  /** I-20 stems mode: the backing file; `song` is then the vocal file. */
  backing?: Song;
  recent?: RecentSummary[];
  model?: Model;
  index?: IndexState;
  plan?: DownloadItem[];
  dl?: Download;
  job?: Job;
  stems?: { vocals: AudioBuffer; inst: AudioBuffer };
  result?: Result;
  prev?: Result;
  ab: 'A' | 'B';
  mix: MixParams;
  mute: { vocal: boolean; inst: boolean }; // I-17: also applies to the saved mix
  solo?: 'vocal' | 'inst'; // listening only
  format: AudioFormat;
  view: 'work' | 'result';
  settingsTab?: SettingsTab;
  confirm?: Confirm;
  myModels?: SavedModel[]; // most recently used first
  cache?: { entries: CacheEntry[]; usage: number; quota: number; persisted: boolean };
  resolvedTheme: 'light' | 'dark'; // what the .dark class currently says (theme may be 'system')
}

export const useApp = create<State>()(
  persist<State, [], [], Partial<State>>(
    (): State => ({
      settings: { lang: 'ko', theme: 'system', backend: 'auto', remember: true, noticeAck: false, advancedOpen: false, saveRecent: true },
      opts: DEFAULT_OPTIONS,
      engineReady: false,
      online: navigator.onLine,
      crashed: false,
      dismissed: [],
      ab: 'B',
      mix: DEFAULT_MIX,
      mute: { vocal: false, inst: false },
      format: 'mp3',
      view: 'work',
      resolvedTheme: 'light',
    }),
    {
      name: 'web-rvc:ui',
      // Bump on any change to the persisted shape and handle the old shape in migrate (A5).
      version: 3,
      migrate: (p, v) => {
        const s = (p ?? {}) as Partial<State>;
        // v1 stored the deployment default URL; drop it so the current default applies (A3).
        if (v < 2 && s.settings?.modelBaseUrl === DEFAULT_BASE_URL) s.settings = { ...s.settings, modelBaseUrl: undefined };
        // v2 always stored the old default (WAV 16-bit); MP3 is the new default (MX-5), a choice made from now on is kept.
        if (v < 3) delete s.format;
        return s;
      },
      // ET-3: options are remembered only when the user keeps "remember" on; the range belongs to one song, never kept.
      partialize: (s) => ({ settings: s.settings, format: s.format, ...(s.settings.remember ? { opts: { ...s.opts, range: undefined } } : {}) }),
      merge: (p, cur) => {
        const saved = (p ?? {}) as Partial<State>;
        return { ...cur, ...saved, settings: { ...cur.settings, ...saved.settings }, opts: { ...cur.opts, ...saved.opts } };
      },
    },
  ),
);

export const set = useApp.setState;
export const get = useApp.getState;

export function confirmAsync(c: Omit<Confirm, 'resolve'>): Promise<boolean | 'alt'> {
  get().confirm?.resolve(false); // a newer dialog replaces an open one; its caller must not wait forever
  return new Promise((resolve) => set({ confirm: { ...c, resolve } }));
}

export const selBusy = (s: State) => !!s.job && s.job.status !== 'error';
/** The engine runs one job or download at a time (BUSY); also blocks cache/backend changes under a download. */
export const selEngineBusy = (s: State) => selBusy(s) || s.dl?.status === 'loading';
export const selModelBaseUrl = (s: State) => s.settings.modelBaseUrl ?? DEFAULT_BASE_URL;
/** Result shown/exported now: A = previous run while comparing. */
export const selCurrentResult = (s: State) => (s.ab === 'A' && s.prev) || s.result;
/** Bytes still to download for the current job plan. */
export const selMissingBytes = (s: State) => s.plan?.filter((d) => !d.cached).reduce((a, d) => a + d.bytes, 0) ?? 0;
/** Seconds of audio the next run processes. */
export const selSeconds = (s: State) => (s.opts.range ? s.opts.range.end - s.opts.range.start : s.song?.duration ?? 0);
export const isBusy = () => selBusy(get());
