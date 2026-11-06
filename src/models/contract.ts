// The tools <-> app contract (ARCHITECTURE §17) in one place: what tools/publish.py writes and the app reads.
// tests/contract.test.ts checks weights/dist against it with the app's own readers; tools check the same rules.
import { AppError, type ErrorCode } from '../shared/errors';
import { PUBLIC_IDS, type PublicId } from '../shared/types';

export { PUBLIC_IDS, type PublicId };

export const MANIFEST_FORMAT = 1;
export const REGISTRY_FORMAT = 1;

// ---- manifest (§17.2): one per ONNX graph, weights come from the .pth in manifest order ----

export interface IoSpec { name: string; dtype: 'float32' | 'int64'; shape: (number | string)[] }
export interface Manifest {
  format: typeof MANIFEST_FORMAT;
  id: string;
  externalData: 'weights.bin';
  bytes: number;
  tensors: { key: string; shape: number[]; offset: number }[];
  inputs: IoSpec[];
  outputs: IoSpec[];
  meta: Record<string, unknown>;
}

// ---- manifest.meta per model kind (§17.3) ----

export interface ContentvecMeta { sr: number; hop: number }
export interface RmvpeMeta { sr: number; hop: number; threshold: number; centsOffset: number }
export interface VrMeta { window: number; offset: number; roi: number; bins: number; outputIs: 'instrumental' | 'vocal' }
/** RVC template; `phase` (a description) is present when the graph takes the NSF phase input. */
export interface TemplateMeta { version: 'v1' | 'v2'; f0: boolean; sr: 32000 | 40000 | 48000; upp: number; signature: string; phase?: string }

export type MetaKind = PublicId | 'template';
export interface MetaOf { contentvec: ContentvecMeta; rmvpe: RmvpeMeta; 'uvr-hp-vocal': VrMeta; template: TemplateMeta }

const num = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
const RULES: { [K in MetaKind]: Record<string, (v: unknown) => boolean> } = {
  contentvec: { sr: num, hop: num },
  rmvpe: { sr: num, hop: num, threshold: num, centsOffset: num },
  'uvr-hp-vocal': { window: num, offset: num, roi: num, bins: num, outputIs: (v) => v === 'instrumental' || v === 'vocal' },
  template: {
    version: (v) => v === 'v1' || v === 'v2', f0: (v) => typeof v === 'boolean', sr: (v) => v === 32000 || v === 40000 || v === 48000,
    upp: num, signature: (v) => typeof v === 'string',
  },
};

/** Throws when a required meta key is missing or has the wrong type (no silent fallbacks, R3 A15). */
export function checkMeta<K extends MetaKind>(kind: K, meta: Record<string, unknown>, code: ErrorCode = 'CACHE_INTEGRITY'): MetaOf[K] {
  for (const [k, ok] of Object.entries(RULES[kind])) {
    if (!ok(meta[k])) throw new AppError(code, `${kind} manifest meta.${k} is ${JSON.stringify(meta[k])}`);
  }
  return meta as unknown as MetaOf[K];
}

// ---- registry (§17.5) ----

export interface PublicModel { graph: string; weights: string; manifest: string }
export interface Registry {
  format: typeof REGISTRY_FORMAT;
  revision: string;
  files: Record<string, { bytes: number; sha256: string }>;
  models: { contentvec: PublicModel; rmvpe: PublicModel; 'uvr-hp-vocal': PublicModel & { params: string } };
  /** Keyed by templateSignature() of the .pth config (formats/rvc-checkpoint.ts). */
  rvcTemplates: Record<string, { id: string; graph: string; manifest: string }>;
}

/** Format and required keys; every referenced path must have a files entry. */
export function checkRegistry(r: unknown): Registry {
  const bad = (why: string): never => { throw new AppError('NET_DOWNLOAD', 'model registry does not match this app', why, false); };
  const g = r as Partial<Registry> | null;
  if (!g || typeof g !== 'object') bad('not an object');
  if (g!.format !== REGISTRY_FORMAT) bad(`format ${String(g!.format)}`);
  if (typeof g!.revision !== 'string' || !g!.revision) bad('revision');
  if (!g!.files || typeof g!.files !== 'object') bad('files');
  if (!g!.models || typeof g!.models !== 'object') bad('models');
  if (!g!.rvcTemplates || typeof g!.rvcTemplates !== 'object') bad('rvcTemplates');
  const reg = g as Registry;
  const paths: string[] = [];
  for (const id of Object.values(PUBLIC_IDS)) {
    const m = reg.models[id] as PublicModel & { params?: string };
    if (!m) bad(`models.${id}`);
    paths.push(m.graph, m.weights, m.manifest, ...(m.params ? [m.params] : []));
  }
  for (const t of Object.values(reg.rvcTemplates)) paths.push(t.graph, t.manifest);
  for (const p of paths) if (!reg.files[p]) bad(`no files entry for ${p}`);
  return reg;
}
