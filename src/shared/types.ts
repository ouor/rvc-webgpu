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

export interface SeparateParams { modelId: 'uvr-hp-vocal'; aggressiveness: number /* 0..20, default 10 */ }

export interface JobSpec {
  inputId: string;
  range?: { start: number; end: number }; // seconds
  mode: 'song' | 'voice';
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
export interface CacheEntry { id: string; name: string; bytes: number; kind: 'public' | 'my-model'; status: 'verified' | 'incomplete' | 'corrupt' }

/** myModels.list() entry: the model info plus when it was saved and last opened (ms since epoch). */
export interface SavedModel extends RvcModelInfo { savedAt: number; lastUsed: number }

/** cache.importFiles() result per given file. */
export interface ImportResult { name: string; path?: string; status: 'imported' | 'unknown' | 'mismatch' }

export const DEFAULT_CONVERT: ConvertParams = {
  f0UpKey: 0, indexRate: 0.75, protect: 0.33, rmsMixRate: 0.25, sid: 0, chunkProfile: 'normal', seed: 0,
};
export const DEFAULT_SEPARATE: SeparateParams = { modelId: 'uvr-hp-vocal', aggressiveness: 10 };

// ---- Engine API exposed by the worker (ARCHITECTURE §6.2); implemented in engine/engine.ts ----

export type OnProgress = (p: Progress) => void;

export interface Engine {
  init(opts: { backend: BackendPref; modelBaseUrl: string }): Promise<EnvReport>;
  setInput(audio: Stereo, meta: { name: string; size: number; lastModified: number }): Promise<string>;
  loadRvcModel(file: File): Promise<RvcModelInfo>;
  setOnnxModelSr(modelId: string, sr: 32000 | 40000 | 48000): Promise<void>;
  loadIndex(file: File): Promise<IndexInfo>;
  plan(job: JobSpec): Promise<DownloadItem[]>;
  download(ids: string[], onProgress: OnProgress): Promise<void>;
  run(job: JobSpec, onProgress: OnProgress): Promise<JobResult>;
  /** Stems of the last separation (available as soon as the separate stage finishes, also during run). */
  getStems(): Promise<{ vocals: Stereo; instrumental: Stereo } | null>;
  cancel(): void;
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
