// onnxruntime-web: EP selection, session creation with WebGPU -> WASM fallback, error classification (§10).
import * as ort from 'onnxruntime-web/webgpu';
import { AppError } from '../shared/errors';
import type { Backend, BackendPref } from '../shared/types';

export type OrtSession = ort.InferenceSession;
export { ort };

export interface GpuProbe { webgpu: boolean; shaderF16: boolean }

/** navigator.gpu lookup (the one DOM-ish dependency of the engine, injectable for tests). */
export async function probeGpu(): Promise<GpuProbe> {
  const gpu = (globalThis.navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } })?.gpu;
  if (!gpu) return { webgpu: false, shaderF16: false };
  try {
    const a = await gpu.requestAdapter();
    return { webgpu: !!a, shaderF16: !!a?.features.has('shader-f16') };
  } catch {
    return { webgpu: false, shaderF16: false };
  }
}

export function configureOrt(): number {
  const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  const cores = globalThis.navigator?.hardwareConcurrency ?? 4;
  ort.env.wasm.numThreads = isolated ? Math.min(8, cores) : 1;
  ort.env.logLevel = 'error';
  return ort.env.wasm.numThreads;
}

export function pickBackend(pref: BackendPref, gpu: GpuProbe): Backend {
  if (pref === 'wasm') return 'wasm';
  return gpu.webgpu ? 'webgpu' : 'wasm';
}

/** Maps ORT / allocation failures to OOM / GPU_LOST, everything else to `fallback`. */
export function classifyOrtError(e: unknown, fallback: 'BACKEND_INIT' | 'INTERNAL' = 'INTERNAL'): AppError {
  if (e instanceof AppError) return e;
  const msg = e instanceof Error ? e.message : String(e);
  if (e instanceof RangeError || /out of memory|\boom\b|bad_alloc|Array buffer allocation failed|memory access out of bounds/i.test(msg))
    return new AppError('OOM', 'out of memory', msg, true);
  if (/device (was )?lost|GPUDevice.*lost|DeviceLost/i.test(msg)) return new AppError('GPU_LOST', 'WebGPU device lost', msg, true);
  return new AppError(fallback, msg, e instanceof Error ? e.stack : undefined);
}

/** Creates a session on `backend`; a WebGPU failure retries once on WASM (EX-1). */
export async function createSession(
  graph: Uint8Array, weights: Uint8Array | undefined, backend: Backend,
): Promise<{ session: OrtSession; backend: Backend }> {
  const opts = (ep: Backend): ort.InferenceSession.SessionOptions => ({
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    ...(weights ? { externalData: [{ path: 'weights.bin', data: weights }] } : {}),
  });
  try {
    return { session: await ort.InferenceSession.create(graph, opts(backend)), backend };
  } catch (e) {
    const c = classifyOrtError(e, 'BACKEND_INIT');
    if (backend === 'wasm' || c.code === 'OOM') throw c;
    try {
      return { session: await ort.InferenceSession.create(graph, opts('wasm')), backend: 'wasm' };
    } catch (e2) {
      throw classifyOrtError(e2, 'BACKEND_INIT');
    }
  }
}

/** session.run with error classification. */
export async function runSession(s: OrtSession, feeds: Record<string, ort.Tensor>): Promise<ort.InferenceSession.OnnxValueMapType> {
  try {
    return await s.run(feeds);
  } catch (e) {
    throw classifyOrtError(e);
  }
}

export async function releaseSession(s: OrtSession | undefined): Promise<void> {
  try { await s?.release(); } catch { /* already gone */ }
}

export const f32 = (data: Float32Array, dims: number[]) => new ort.Tensor('float32', data, dims);
export const i64 = (data: BigInt64Array, dims: number[]) => new ort.Tensor('int64', data, dims);
