// Fixture loader for ARCHITECTURE §17.6 cases. Fixture root: $WEB_RVC_FIXTURES or tests/fixtures.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export const FIXTURES = process.env.WEB_RVC_FIXTURES ?? join(import.meta.dirname, 'fixtures');
export const WEIGHTS = process.env.WEB_RVC_WEIGHTS; // e.g. external/weight; heavy tests skip without it
export const EXTERNAL = join(import.meta.dirname, '..', 'external');

type Arr = Float32Array | Float64Array | BigInt64Array;
interface Entry { file: string; dtype: 'float32' | 'float64' | 'int64'; shape: number[] }
export interface Fixture { meta: Record<string, unknown>; inputs: Record<string, Arr>; outputs: Record<string, Arr>; shapes: Record<string, number[]> }

export const hasFixture = (name: string) => existsSync(join(FIXTURES, name, 'case.json'));

export function loadFixture(name: string, root = FIXTURES): Fixture {
  const dir = join(root, name);
  const c = JSON.parse(readFileSync(join(dir, 'case.json'), 'utf8'));
  const shapes: Record<string, number[]> = {};
  const load = (group: Record<string, Entry>) => Object.fromEntries(Object.entries(group).map(([k, e]) => {
    const b = readFileSync(join(dir, e.file));
    const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    shapes[k] = e.shape;
    return [k, e.dtype === 'int64' ? new BigInt64Array(ab) : e.dtype === 'float64' ? new Float64Array(ab) : new Float32Array(ab)];
  }));
  return { meta: c.meta ?? {}, inputs: load(c.inputs), outputs: load(c.outputs), shapes };
}

/** max |a-b| / max |b| */
export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0, den = 0;
  for (let i = 0; i < b.length; i++) { num = Math.max(num, Math.abs(a[i] - b[i])); den = Math.max(den, Math.abs(b[i])); }
  return num / (den || 1);
}

export function snrDb(a: ArrayLike<number>, ref: ArrayLike<number>): number {
  let n = 0, d = 0;
  for (let i = 0; i < ref.length; i++) { n += (a[i] - ref[i]) ** 2; d += ref[i] ** 2; }
  return 10 * Math.log10(d / (n || 1e-30));
}

// ---- synthetic inputs (no model files needed) ----

/** The §17.4 example config (v2, 40 kHz); `nSpk` is cfg[15]. */
export const CFG_V2_40K = [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000];
export const SIG_V2_F0_40K = '["v2",1,192,192,768,2,6,3,"1",[3,7,11],[[1,3,5],[1,3,5],[1,3,5]],[10,10,2,2],512,[16,16,4,4],256,40000]';

interface FakeTensor { tensor: string; shape: number[] } // fp16 storage named `tensor`

/** Minimal pickle (protocol 2) encoder for dicts, lists, ints, strings, bools and fp16 tensors. */
function pickle(v: unknown): number[] {
  const enc = (s: string) => [...new TextEncoder().encode(s)];
  const str = (s: string) => { const b = enc(s); return b.length < 256 ? [0x8c, b.length, ...b] : [0x58, ...le32(b.length), ...b]; };
  const le32 = (n: number) => [n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255];
  const go = (x: unknown): number[] => {
    if (typeof x === 'string') return str(x);
    if (typeof x === 'boolean') return [x ? 0x88 : 0x89];
    if (typeof x === 'number') return [0x4a, ...le32(x)];
    if (Array.isArray(x)) return x.length ? [0x5d, 0x28, ...x.flatMap(go), 0x65] : [0x5d];
    if (x && typeof x === 'object' && 'tensor' in x) {
      const t = x as FakeTensor;
      const n = t.shape.reduce((a, b) => a * b, 1);
      const stride = t.shape.map((_, i) => t.shape.slice(i + 1).reduce((a, b) => a * b, 1));
      const tuple = (xs: number[]) => [0x28, ...xs.flatMap(go), 0x74];
      return [0x63, ...enc('torch._utils\n_rebuild_tensor_v2\n'), 0x28,
        0x28, ...str('storage'), 0x63, ...enc('torch\nHalfStorage\n'), ...str(t.tensor), ...str('cpu'), ...go(n), 0x74, 0x51,
        ...go(0), ...tuple(t.shape), ...tuple(stride), 0x89, 0x74, 0x52];
    }
    const entries = Object.entries(x as object);
    return [0x7d, ...(entries.length ? [0x28, ...entries.flatMap(([k, val]) => [...str(k), ...go(val)]), 0x75] : [])];
  };
  return [0x80, 2, ...go(v), 0x2e];
}

/** A tiny RVC inference .pth: config, version, f0 and an fp16 emb_g [nSpk, 256] (all zeros). */
export async function fakeRvcPth(o: { version?: 'v1' | 'v2'; nSpk?: number; salt?: string; embShape?: number[] } = {}): Promise<Blob> {
  const { zipSync } = await import('fflate');
  const nSpk = o.nSpk ?? 1;
  const cfg = [...CFG_V2_40K];
  cfg[15] = nSpk;
  if (o.version === 'v1') cfg[4] = 768; // v1 shares the config layout; the version string selects 256-dim features
  const shape = o.embShape ?? [nSpk, 256];
  const root = { weight: { 'emb_g.weight': { tensor: '0', shape } }, config: cfg, version: o.version ?? 'v2', f0: 1, info: o.salt ?? '1epoch' };
  return new Blob([zipSync({ 'archive/data.pkl': new Uint8Array(pickle(root)), 'archive/data/0': new Uint8Array(nSpk * 256 * 2) }, { level: 0 })]);
}

/** faiss IndexFlatL2 ("IxF2") with n vectors of dimension d; `seed` changes the content (and so the id). */
export function fakeFlatIndex(d: number, n: number, seed = 1): Blob {
  const head = new DataView(new ArrayBuffer(4 + 4 + 8 + 16 + 1 + 4 + 8));
  let o = 0;
  [..."IxF2"].forEach((c) => head.setUint8(o++, c.charCodeAt(0)));
  head.setInt32(o, d, true); o += 4;
  head.setBigUint64(o, BigInt(n), true); o += 8 + 16;
  head.setUint8(o, 1); o += 1;
  head.setInt32(o, 1, true); o += 4; // METRIC_L2
  head.setBigUint64(o, BigInt(n * d), true);
  const data = Float32Array.from({ length: n * d }, (_, i) => Math.sin(i * seed));
  return new Blob([head.buffer, data.buffer]);
}
