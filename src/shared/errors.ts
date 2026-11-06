// Error model (ARCHITECTURE §10). `message` is developer English; UI text comes from i18n keyed by `code`.

export type ErrorCode =
  | 'AUDIO_DECODE' | 'AUDIO_TOO_LONG'
  | 'PTH_NOT_ZIP' | 'PTH_UNSAFE' | 'PTH_CORRUPT' | 'PTH_TRAINING_CKPT'
  | 'PTH_UNSUPPORTED_ARCH' | 'PTH_UNSUPPORTED_VOCODER' | 'PTH_UNSUPPORTED_EMBEDDER'
  | 'ONNX_UNRECOGNIZED'
  | 'INDEX_EMPTY' | 'INDEX_UNSUPPORTED' | 'INDEX_DIM_MISMATCH'
  | 'ZIP_NO_MODEL'
  | 'NET_DOWNLOAD' | 'CACHE_INTEGRITY' | 'STORAGE_QUOTA'
  | 'BACKEND_INIT' | 'OOM' | 'GPU_LOST'
  | 'MODELS_MISSING' // run() before the public models it needs were downloaded (AC-18: no silent downloads)
  | 'CANCELLED' | 'BUSY' | 'NOT_FOUND' | 'INTERNAL';

export class AppError extends Error {
  override name = 'AppError';
  constructor(public code: ErrorCode, message: string, public detail?: string, public recoverable = false) {
    super(message);
  }
}

export const isAppError = (e: unknown): e is AppError => e instanceof AppError;

/** Wraps unknown throwables so callers always see an AppError. */
export function toAppError(e: unknown, fallback: ErrorCode = 'INTERNAL'): AppError {
  if (e instanceof AppError) return e;
  const msg = e instanceof Error ? e.message : String(e);
  return new AppError(fallback, msg, e instanceof Error ? e.stack : undefined);
}
