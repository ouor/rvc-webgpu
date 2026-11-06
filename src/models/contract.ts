// The tools <-> app contract (ARCHITECTURE §17). The values live in contract.json, which tools read too;
// this module only adds types and the checks. tests/contract.test.ts checks weights/dist against both.
import { AppError, type ErrorCode } from '../shared/errors';
import { PUBLIC_IDS, type PublicId } from '../shared/types';
import C from './contract.json';

export { PUBLIC_IDS, type PublicId };
export const CONTRACT = C;

export const MANIFEST_FORMAT = C.manifestFormat as 1;
export const REGISTRY_FORMAT = C.registryFormat as 1;

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

/** contract.json meta rule: a typeof name (numbers must be finite) or the list of allowed values. */
type Rule = string | readonly (string | number)[];
const ok = (rule: Rule, v: unknown) =>
  Array.isArray(rule) ? rule.includes(v as never) : typeof v === rule && (rule !== 'number' || Number.isFinite(v));
const metaRules = (kind: MetaKind): Record<string, Rule> => (kind === 'template' ? C.template.meta : C.public[kind].meta);

/** Throws when a required meta key is missing or has the wrong type (no silent fallbacks, R3 A15). */
export function checkMeta<K extends MetaKind>(kind: K, meta: Record<string, unknown>, code: ErrorCode = 'CACHE_INTEGRITY'): MetaOf[K] {
  for (const [k, rule] of Object.entries(metaRules(kind))) {
    if (!ok(rule, meta[k])) throw new AppError(code, `${kind} manifest meta.${k} is ${JSON.stringify(meta[k])}`);
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
  const g = r as Record<string, unknown> | null;
  if (!g || typeof g !== 'object') bad('not an object');
  for (const k of C.registryKeys) if (!g![k]) bad(k);
  if (g!.format !== REGISTRY_FORMAT) bad(`format ${String(g!.format)}`);
  if (typeof g!.revision !== 'string') bad('revision');
  const reg = g as unknown as Registry;
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
