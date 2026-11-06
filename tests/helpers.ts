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
