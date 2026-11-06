// UI state (zustand). Engine calls live in controller.ts; this file only holds data.
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AppError } from '@/shared/errors';
import type { BackendPref, DownloadItem, EnvReport, IndexInfo, RvcModelInfo } from '@/shared/types';
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
  modelBaseUrl: string;
  remember: boolean;
  noticeAck: boolean;
  advancedOpen: boolean;
}

export interface Options {
  mode: 'song' | 'voice';
  key: number;
  indexRate: number;
  protect: number;
  rmsMixRate: number;
  sid: number;
  aggressiveness: number;
  range?: { start: number; end: number };
}
export const DEFAULT_OPTIONS: Options = { mode: 'song', key: 0, indexRate: 0.75, protect: 0.33, rmsMixRate: 0.25, sid: 0, aggressiveness: 10 };

export interface Song {
  status: 'parsing' | 'ready' | 'error';
  name: string;
  size: number;
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
}

export interface IndexState {
  status: 'parsing' | 'ready' | 'error';
  fileName: string;
  id?: string; // set for a loaded index (also from a zip or a saved model)
  info?: IndexInfo; // undefined when the index came inside a zip / my-model (engine returns only the id)
  error?: AppError;
}

export type StepState = 'wait' | 'load' | 'run' | 'done' | 'reused' | 'skipped' | 'error' | 'cancelled';
export type Step = RunStage | 'mix';
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
  changes?: string; // rerun title suffix, e.g. "키 0 → +12"
}

export interface Download {
  status: 'confirm' | 'loading' | 'failed' | 'storage' | 'offline';
  open: boolean;
  items: DownloadItem[];
  done: number;
  total: number;
  startedAt?: number;
  error?: AppError;
}

export interface Result {
  songName: string;
  modelName: string;
  mode: Options['mode'];
  opts: Options;
  converted: AudioBuffer; // model SR
  vocals?: AudioBuffer;
  inst?: AudioBuffer;
  ms: number;
  seq: number;
}

export interface Confirm { title: string; body: string; action: string; destructive?: boolean; resolve: (ok: boolean) => void }

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
  format: AudioFormat;
  view: 'work' | 'result';
  settingsTab?: SettingsTab;
  confirm?: Confirm;
  myModelsVersion: number;
}

export const useApp = create<State>()(
  persist(
    (): State => ({
      settings: { lang: 'ko', theme: 'system', backend: 'auto', modelBaseUrl: DEFAULT_BASE_URL, remember: true, noticeAck: false, advancedOpen: false },
      opts: DEFAULT_OPTIONS,
      engineReady: false,
      online: navigator.onLine,
      crashed: false,
      dismissed: [],
      ab: 'B',
      mix: DEFAULT_MIX,
      format: 'pcm16',
      view: 'work',
      myModelsVersion: 0,
    }),
    {
      name: 'web-rvc:ui',
      version: 1,
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

export function confirmAsync(c: Omit<Confirm, 'resolve'>): Promise<boolean> {
  return new Promise((resolve) => set({ confirm: { ...c, resolve } }));
}

export const selBusy = (s: State) => !!s.job && s.job.status !== 'error';
export const isBusy = () => selBusy(get());
