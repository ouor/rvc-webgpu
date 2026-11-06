// Types that cross the worker boundary. Structured-clone friendly only (ARCHITECTURE §6.1).

export type Backend = 'webgpu' | 'wasm';
export type BackendPref = 'auto' | Backend;

export interface Mono { sr: number; data: Float32Array }
export interface Stereo { sr: number; l: Float32Array; r: Float32Array }

export type StageId = 'download' | 'separate' | 'f0' | 'features' | 'convert';
export interface Progress {
  stage: StageId;
  done: number; // units processed (chunks, patches, bytes)
  total: number;
  /** load = model is being prepared (UI: "RMVPE 불러오는 중"), run = processing, verify = SHA-256 check. */
  phase?: 'load' | 'run' | 'verify';
  /** download stage only: registry file being fetched. */
  file?: string;
  /** download stage only: DownloadItem id the file belongs to, and that item's bytes (done/total are overall). */
  item?: string;
  itemDone?: number;
  itemTotal?: number;
  reused?: boolean; // skipped thanks to the F2 cache
  skipped?: boolean; // separate stage in 'stems' mode: the user gave the stems (I-20)
  backend?: Backend;
}

export interface EnvReport {
  backend: Backend; // EP that will be tried first
  webgpu: boolean;
  shaderF16: boolean;
  crossOriginIsolated: boolean;
  wasmThreads: number;
  gpuAdapter?: string; // WebGPU adapter description/vendor, when available
  modelRevision: string; // registry revision the public model cache belongs to
  storage: { usage: number; quota: number; persisted: boolean };
}

export interface RvcModelInfo {
  id: string; // session handle (first 16 hex chars of the file hash)
  name: string;
  source: 'pth' | 'onnx';
  version: 'v1' | 'v2';
  f0: boolean;
  sr: 32000 | 40000 | 48000;
  nSpk: number; // as stored in the checkpoint config (WebUI writes 109 even for single-speaker models)
  templateId?: string;
  bytes: number;
  /** F4: sr was not in the file; UI must ask and call setOnnxModelSr(). */
  srUnknown?: boolean;
  /** Set when a .zip or a saved model brought an index, which was loaded alongside the model. */
  indexId?: string;
  index?: IndexInfo; // details of that index (vector count etc.)
  meta?: { epoch?: number; vocoder?: string; embedder?: string; author?: string };
}

export interface IndexInfo {
  id: string;
  nVectors: number;
  dim: 256 | 768;
  nlist: number;
  nprobe: number;
  downsampled?: number; // vectors kept after the memory cap
}

export interface ConvertParams {
  f0UpKey: number; // -24..24
  indexRate: number; // 0..1, ignored without index
  protect: number; // 0..0.5
  rmsMixRate: number; // 0..1
  sid: number;
  chunkProfile: 'normal' | 'low'; // 1/6/38/41 | 1/5/30/32 seconds (x_pad/x_query/x_center/x_max)
  seed: number;
}

/** Ids of the fixed public models (DownloadItem.id, CacheEntry.id). RVC templates are public items too; their ids come from the registry. */
export const PUBLIC_IDS = { contentvec: 'contentvec', rmvpe: 'rmvpe', separator: 'uvr-hp-vocal' } as const;
export type PublicId = (typeof PUBLIC_IDS)[keyof typeof PUBLIC_IDS];

export interface SeparateParams { modelId: typeof PUBLIC_IDS.separator; aggressiveness: number /* 0..20, default 10 */ }

export interface JobSpec {
  inputId: string;
  range?: { start: number; end: number }; // seconds
  /** song = separate then convert; voice = convert the input as is; stems = input from setStemsInput, no separation (I-20). */
  mode: 'song' | 'voice' | 'stems';
  separate: SeparateParams;
  modelId: string;
  indexId?: string;
  convert: ConvertParams;
}

export interface JobResult {
  vocals?: Stereo;
  instrumental?: Stereo;
  converted: Mono; // model SR
  reused: StageId[];
  timings: Partial<Record<StageId, number>>; // ms
}

export interface DownloadItem { id: string; label: string; bytes: number; cached: boolean }

/**
 * Cache item. status from list(): 'verified' = all files present (each was SHA-256 checked when it entered the cache
 * and is checked again on every load), 'incomplete' = a file is missing or partially downloaded.
 * cache.verify(id) re-hashes now and can also return 'corrupt' (bad files are deleted).
 */
export interface CacheEntry { id: string; name: string; bytes: number; kind: 'public' | 'my-model' | 'recent'; status: 'verified' | 'incomplete' | 'corrupt' }

/** Mix settings kept with a recent job (same fields as audio/mixer MixParams, without the playback-only solo). */
export interface RecentMix { vocalGainDb: number; instGainDb: number; reverb: number; muteVocal?: boolean; muteInst?: boolean }
export type RecentStem = 'converted' | 'vocals' | 'inst';
/** recent/<id>/job.json, version 1 (ARCHITECTURE 18.2). */
export interface RecentJob {
  v: 1;
  id: string;
  createdAt: number;
  appVersion: string;
  mode: JobSpec['mode'];
  song: { name: string; durationSec: number; range?: { start: number; end: number } };
  model: { id: string; name: string; version: 'v1' | 'v2'; sr: number; f0: boolean; indexId?: string };
  convert: ConvertParams;
  separate?: SeparateParams; // song mode only
  mix: RecentMix;
  /** scale: the file holds samples x scale (0.999 / peak when the peak was above 1.0); open() divides it out. */
  files: Partial<Record<RecentStem, { file: string; sr: number; channels: 1 | 2; frames: number; scale: number; bytes: number }>>;
}
export interface RecentSummary {
  id: string; createdAt: number; songName: string; modelName: string; mode: JobSpec['mode'];
  f0UpKey: number; range?: { start: number; end: number }; bytes: number;
}

/** myModels.list() entry: the model info plus when it was saved and last opened (ms since epoch). */
export interface SavedModel extends RvcModelInfo { savedAt: number; lastUsed: number }

/** cache.importFiles() result per given file. */
export interface ImportResult { name: string; path?: string; status: 'imported' | 'unknown' | 'mismatch' }

export const DEFAULT_CONVERT: ConvertParams = {
  f0UpKey: 0, indexRate: 0.75, protect: 0.33, rmsMixRate: 0.25, sid: 0, chunkProfile: 'normal', seed: 0,
};
export const DEFAULT_SEPARATE: SeparateParams = { modelId: PUBLIC_IDS.separator, aggressiveness: 10 };

// ---- Engine API exposed by the worker (ARCHITECTURE §6.2); implemented in engine/engine.ts ----

export type OnProgress = (p: Progress) => void;

export interface Engine {
  init(opts: { backend: BackendPref; modelBaseUrl: string }): Promise<EnvReport>;
  setInput(audio: Stereo, meta: { name: string; size: number; lastModified: number }): Promise<string>;
  /** I-20: already separated vocal + instrumental (44.1 kHz). The shorter one is padded with silence. For mode 'stems'. */
  setStemsInput(vocals: Stereo, instrumental: Stereo, meta: { name: string; size: number; lastModified: number }): Promise<string>;
  loadRvcModel(file: File): Promise<RvcModelInfo>;
  setOnnxModelSr(modelId: string, sr: 32000 | 40000 | 48000): Promise<void>;
  loadIndex(file: File): Promise<IndexInfo>;
  plan(job: JobSpec): Promise<DownloadItem[]>;
  download(ids: string[], onProgress: OnProgress): Promise<void>;
  run(job: JobSpec, onProgress: OnProgress): Promise<JobResult>;
  /** Stems of the last separation (available as soon as the separate stage finishes, also during run). */
  getStems(): Promise<{ vocals: Stereo; instrumental: Stereo } | null>;
  cancel(): void;
  /** I-2: results of finished runs kept on this device (at most 3). */
  recent: {
    /** Saves the last run() result. The UI calls it only while its "save recent jobs" switch is on. */
    save(meta: { songName: string; songSeconds: number; appVersion: string; mix: RecentMix }): Promise<{ id: string } | { skipped: 'quota' }>;
    list(): Promise<RecentSummary[]>;
    open(id: string): Promise<{ job: RecentJob; converted: Mono; vocals?: Stereo; instrumental?: Stereo }>;
    updateMix(id: string, mix: RecentMix): Promise<void>;
    remove(id: string): Promise<void>;
    clear(): Promise<void>;
  };
  cache: {
    list(): Promise<CacheEntry[]>;
    /** Full SHA-256 check of one item now (public or my-model id). */
    verify(id: string): Promise<CacheEntry['status']>;
    /** PRD R10: fill the public cache from local copies of the model files (matched by name, SHA-256 checked). */
    importFiles(files: File[]): Promise<ImportResult[]>;
    remove(id: string): Promise<void>;
    clear(): Promise<void>;
  };
  myModels: {
    save(modelId: string, indexId?: string): Promise<void>;
    list(): Promise<SavedModel[]>; // most recently used first
    open(id: string): Promise<RvcModelInfo>;
    rename(id: string, name: string): Promise<void>;
    remove(id: string): Promise<void>;
  };
}
