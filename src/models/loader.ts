// (weight-less graph, state_dict, manifest) -> fp32 weights.bin -> ORT session (ARCHITECTURE §7.3, §17.2).
import { AppError, type ErrorCode } from '../shared/errors';
import type { Backend } from '../shared/types';
import { numel, toFloat32, type Pth, type TensorRef } from '../formats/pth';
import { createSession, type OrtSession } from './ort';
import { MANIFEST_FORMAT, type Manifest } from './contract';

export type { IoSpec, Manifest } from './contract';

/** Validates the manifest shape rules tools promise (§17.2 rules 1-2). */
export function checkManifest(m: Manifest, code: ErrorCode = 'CACHE_INTEGRITY'): void {
  if (m.format !== MANIFEST_FORMAT) throw new AppError(code, `manifest ${m.id}: format ${m.format}`);
  let end = 0;
  for (const t of m.tensors) {
    if (t.offset !== end) throw new AppError(code, `manifest ${m.id}: tensor ${t.key} offset ${t.offset}, expected ${end}`);
    end += numel(t.shape) * 4;
  }
  if (end !== m.bytes) throw new AppError(code, `manifest ${m.id}: bytes ${m.bytes}, tensors end at ${end}`);
}

/**
 * Builds weights.bin from `tensors` (a state_dict map) following the manifest, then creates the session.
 * `mismatch` is the error raised when a key is missing or its shape differs: PTH_UNSUPPORTED_ARCH for user models,
 * CACHE_INTEGRITY for public ones (§17.2 rule 4).
 */
export async function createSessionFromParts(
  parts: { graph: Uint8Array; manifest: Manifest; pth: Pth; tensors: Map<string, TensorRef> },
  backend: Backend,
  mismatch: ErrorCode,
): Promise<{ session: OrtSession; backend: Backend }> {
  const { manifest: m, pth, tensors } = parts;
  checkManifest(m, mismatch === 'PTH_UNSUPPORTED_ARCH' ? 'CACHE_INTEGRITY' : mismatch);
  let bin: Uint8Array;
  try {
    bin = new Uint8Array(m.bytes);
  } catch (e) {
    throw new AppError('OOM', `cannot allocate ${m.bytes} bytes for ${m.id} weights`, String(e), true);
  }
  for (const spec of m.tensors) {
    const t = tensors.get(spec.key);
    if (!t) throw new AppError(mismatch, `tensor ${spec.key} missing for ${m.id}`, spec.key);
    if (t.shape.join() !== spec.shape.join())
      throw new AppError(mismatch, `tensor ${spec.key} shape ${t.shape} != ${spec.shape}`, spec.key);
    toFloat32(t.dtype, await pth.read(t), new Float32Array(bin.buffer, spec.offset, numel(spec.shape)));
  }
  return createSession(parts.graph, bin, backend); // bin is dropped on return: ORT copied it to the GPU / WASM heap
}

/** Flat state_dict (public models) or the `weight` dict (user RVC models) as key -> tensor. */
export function tensorMap(root: unknown, sub?: string): Map<string, TensorRef> {
  const d = sub ? (root as Map<string, unknown>)?.get?.(sub) : root;
  if (!(d instanceof Map)) throw new AppError('PTH_CORRUPT', 'state_dict is not a dict');
  return d as Map<string, TensorRef>;
}
