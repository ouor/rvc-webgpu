// PyTorch zip checkpoint (.pth) reader that never executes pickle (ARCHITECTURE §7.1).
// Port of docs/dev/architecture/spikes/pth-reader.mjs, reading from a Blob so tensor bytes are sliced lazily.
import { AppError } from '../shared/errors';

export type DType = 'float16' | 'bfloat16' | 'float32' | 'int64' | 'int32' | 'bool' | 'uint8';
export interface TensorRef { tensor: true; dtype: DType; shape: number[]; storage: string; offset: number }

const STORAGE: Record<string, DType> = {
  HalfStorage: 'float16', FloatStorage: 'float32', BFloat16Storage: 'bfloat16',
  LongStorage: 'int64', IntStorage: 'int32', BoolStorage: 'bool', ByteStorage: 'uint8',
};
const GLOBALS = new Set([
  'collections OrderedDict', 'torch._utils _rebuild_tensor_v2', 'torch._utils _rebuild_parameter',
  ...Object.keys(STORAGE).map((s) => 'torch ' + s),
]);
export const DTYPE_BYTES: Record<DType, number> = { float16: 2, bfloat16: 2, float32: 4, int64: 8, int32: 4, bool: 1, uint8: 1 };

const corrupt = (msg: string) => new AppError('PTH_CORRUPT', msg);
const bytesOf = async (b: Blob) => new Uint8Array(await b.arrayBuffer());

// ---- zip (central directory only; entries are read on demand) ----

interface ZipEntry { method: number; size: number; local: number }

export interface ZipIndex {
  names: string[];
  /** Bytes of a STORED entry as a Blob slice (no read yet). */
  entry(name: string): Promise<Blob | null>;
}

export async function readZipIndex(blob: Blob): Promise<ZipIndex> {
  const tailLen = Math.min(blob.size, 65557 + 20);
  const tail = await bytesOf(blob.slice(blob.size - tailLen));
  const tdv = new DataView(tail.buffer);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) if (tdv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new AppError('PTH_NOT_ZIP', 'not a zip-format checkpoint (legacy torch.save or not a model)');
  let total = tdv.getUint16(eocd + 10, true);
  let cdSize = tdv.getUint32(eocd + 12, true);
  let cdOff = tdv.getUint32(eocd + 16, true);
  if (total === 0xffff || cdOff === 0xffffffff || cdSize === 0xffffffff) {
    const loc = eocd - 20; // zip64 EOCD locator sits right before the EOCD
    if (loc < 0 || tdv.getUint32(loc, true) !== 0x07064b50) throw corrupt('zip64 locator missing');
    const e64 = Number(tdv.getBigUint64(loc + 8, true));
    const rec = new DataView((await bytesOf(blob.slice(e64, e64 + 56))).buffer);
    total = Number(rec.getBigUint64(32, true));
    cdSize = Number(rec.getBigUint64(40, true));
    cdOff = Number(rec.getBigUint64(48, true));
  }
  if (cdOff + cdSize > blob.size) throw corrupt('central directory out of range');
  const cd = await bytesOf(blob.slice(cdOff, cdOff + cdSize));
  const dv = new DataView(cd.buffer);
  const td = new TextDecoder();
  const entries = new Map<string, ZipEntry>();
  let p = 0;
  for (let i = 0; i < total; i++) {
    if (p + 46 > cd.length || dv.getUint32(p, true) !== 0x02014b50) throw corrupt('bad central directory');
    const method = dv.getUint16(p + 10, true);
    let size = dv.getUint32(p + 24, true);
    const csize = dv.getUint32(p + 20, true);
    const nameLen = dv.getUint16(p + 28, true), extraLen = dv.getUint16(p + 30, true), cmtLen = dv.getUint16(p + 32, true);
    let local = dv.getUint32(p + 42, true);
    const name = td.decode(cd.subarray(p + 46, p + 46 + nameLen));
    if (size === 0xffffffff || local === 0xffffffff || csize === 0xffffffff) {
      let q = p + 46 + nameLen;
      const end = q + extraLen;
      while (q + 4 <= end) {
        const id = dv.getUint16(q, true), len = dv.getUint16(q + 2, true);
        if (id === 1) { // zip64 extra: only the saturated fields, in order usize, csize, offset
          let r = q + 4;
          if (size === 0xffffffff) { size = Number(dv.getBigUint64(r, true)); r += 8; }
          if (csize === 0xffffffff) r += 8;
          if (local === 0xffffffff) local = Number(dv.getBigUint64(r, true));
        }
        q += 4 + len;
      }
    }
    entries.set(name, { method, size, local });
    p += 46 + nameLen + extraLen + cmtLen;
  }
  return {
    names: [...entries.keys()],
    async entry(name) {
      const e = entries.get(name);
      if (!e) return null;
      if (e.method !== 0) throw corrupt(`compressed zip entry ${name}`);
      const h = new DataView((await bytesOf(blob.slice(e.local, e.local + 30))).buffer);
      if (h.byteLength < 30 || h.getUint32(0, true) !== 0x04034b50) throw corrupt('bad local header');
      const start = e.local + 30 + h.getUint16(26, true) + h.getUint16(28, true);
      if (start + e.size > blob.size) throw corrupt('zip entry out of range');
      return blob.slice(start, start + e.size);
    },
  };
}

// ---- restricted unpickler ----

const MARK = Symbol('mark');
type Global = { global: string };
interface StorageRef { storageType: DType; key: string }

/** Interprets only an allowlisted subset of pickle. Dicts become Map, tuples/lists arrays, tensors TensorRef. */
export function unpickle(bytes: Uint8Array): unknown {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const td = new TextDecoder();
  const stack: unknown[] = [];
  const memo: unknown[] = [];
  let i = 0;
  const need = (n: number) => { if (i + n > bytes.length) throw corrupt('truncated pickle'); };
  const popMark = () => {
    const k = stack.lastIndexOf(MARK);
    if (k < 0) throw corrupt('pickle MARK not found');
    return stack.splice(k).slice(1);
  };
  const top = () => stack[stack.length - 1];
  const asGlobal = (mod: string, name: string): Global => {
    const g = mod + ' ' + name;
    if (!GLOBALS.has(g)) throw new AppError('PTH_UNSAFE', 'refused pickle global ' + g, g);
    return { global: g };
  };
  const setItems = (d: unknown, kv: unknown[]) => {
    if (!(d instanceof Map)) throw corrupt('SETITEM on non-dict');
    for (let j = 0; j < kv.length; j += 2) d.set(kv[j], kv[j + 1]);
  };
  const call = (fn: unknown, args: unknown): unknown => {
    const g = (fn as Global)?.global;
    const a = args as unknown[];
    switch (g) {
      case 'collections OrderedDict': return new Map();
      case 'torch._utils _rebuild_tensor_v2': {
        const [st, offset, shape, stride] = a as [StorageRef, number, number[], number[]];
        const nat = (x: unknown) => Number.isSafeInteger(x) && (x as number) >= 0;
        if (!st?.key || !nat(offset) || !Array.isArray(shape) || !Array.isArray(stride) || !shape.every(nat) || !stride.every(nat))
          throw corrupt('bad tensor record');
        let s = 1; // only contiguous tensors (what torch.save writes for state_dicts)
        for (let d = shape.length - 1; d >= 0; d--) {
          if (shape[d] !== 1 && stride[d] !== s) throw new AppError('PTH_UNSAFE', 'non-contiguous tensor');
          s *= shape[d];
        }
        return { tensor: true, dtype: st.storageType, storage: st.key, offset, shape: [...shape] } satisfies TensorRef;
      }
      case 'torch._utils _rebuild_parameter': return a[0];
      default: throw new AppError('PTH_UNSAFE', 'refused call of ' + String(g));
    }
  };
  for (;;) {
    need(1);
    const op = bytes[i++];
    switch (op) {
      case 0x80: need(1); i++; break; // PROTO
      case 0x95: need(8); i += 8; break; // FRAME
      case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
      case 0x5d: case 0x29: stack.push([]); break; // EMPTY_LIST, EMPTY_TUPLE
      case 0x28: stack.push(MARK); break; // MARK
      case 0x71: need(1); memo[bytes[i++]] = top(); break; // BINPUT
      case 0x72: need(4); memo[dv.getUint32(i, true)] = top(); i += 4; break; // LONG_BINPUT
      case 0x94: memo.push(top()); break; // MEMOIZE
      case 0x68: need(1); stack.push(memo[bytes[i++]]); break; // BINGET
      case 0x6a: need(4); stack.push(memo[dv.getUint32(i, true)]); i += 4; break; // LONG_BINGET
      case 0x58: { need(4); const n = dv.getUint32(i, true); i += 4; need(n); stack.push(td.decode(bytes.subarray(i, i + n))); i += n; break; } // BINUNICODE
      case 0x8c: { need(1); const n = bytes[i++]; need(n); stack.push(td.decode(bytes.subarray(i, i + n))); i += n; break; } // SHORT_BINUNICODE
      case 0x4a: need(4); stack.push(dv.getInt32(i, true)); i += 4; break; // BININT
      case 0x4b: need(1); stack.push(bytes[i++]); break; // BININT1
      case 0x4d: need(2); stack.push(dv.getUint16(i, true)); i += 2; break; // BININT2
      case 0x8a: { // LONG1
        need(1); const n = bytes[i++]; need(n);
        let v = 0n;
        for (let j = n - 1; j >= 0; j--) v = (v << 8n) | BigInt(bytes[i + j]);
        if (n && bytes[i + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
        i += n; stack.push(Number(v)); break;
      }
      case 0x47: need(8); stack.push(dv.getFloat64(i, false)); i += 8; break; // BINFLOAT
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x4e: stack.push(null); break; // NONE
      case 0x74: stack.push(popMark()); break; // TUPLE
      case 0x85: stack.push([stack.pop()]); break; // TUPLE1
      case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x63: { // GLOBAL "module\nname\n"
        let j = bytes.indexOf(10, i); if (j < 0) throw corrupt('bad GLOBAL');
        const mod = td.decode(bytes.subarray(i, j)); i = j + 1;
        j = bytes.indexOf(10, i); if (j < 0) throw corrupt('bad GLOBAL');
        const name = td.decode(bytes.subarray(i, j)); i = j + 1;
        stack.push(asGlobal(mod, name)); break;
      }
      case 0x93: { const name = stack.pop(), mod = stack.pop(); stack.push(asGlobal(String(mod), String(name))); break; } // STACK_GLOBAL
      case 0x52: { const args = stack.pop(), fn = stack.pop(); stack.push(call(fn, args)); break; } // REDUCE
      case 0x51: { // BINPERSID ('storage', StorageType, key, location, numel)
        const pid = stack.pop() as unknown[];
        const t = Array.isArray(pid) && pid[0] === 'storage' ? STORAGE[((pid[1] as Global)?.global ?? '').split(' ')[1]] : undefined;
        if (!t) throw corrupt('bad persistent id');
        stack.push({ storageType: t, key: String(pid[2]) } satisfies StorageRef); break;
      }
      case 0x73: { const v = stack.pop(), k = stack.pop(); setItems(top(), [k, v]); break; } // SETITEM
      case 0x75: { const kv = popMark(); setItems(top(), kv); break; } // SETITEMS
      case 0x61: { const v = stack.pop(); const l = top(); if (!Array.isArray(l)) throw corrupt('APPEND on non-list'); l.push(v); break; } // APPEND
      case 0x65: { const vs = popMark(); const l = top(); if (!Array.isArray(l)) throw corrupt('APPENDS on non-list'); l.push(...vs); break; } // APPENDS
      case 0x2e: return stack.pop(); // STOP
      default: throw new AppError('PTH_UNSAFE', `unsupported pickle opcode 0x${op.toString(16)} at ${i - 1}`);
    }
  }
}

export const isTensor = (v: unknown): v is TensorRef => typeof v === 'object' && v !== null && (v as TensorRef).tensor === true;
export const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

export interface Pth {
  root: unknown;
  /** Raw little-endian bytes of a tensor (copy). */
  read(t: TensorRef): Promise<Uint8Array>;
}

export async function readPth(blob: Blob): Promise<Pth> {
  const zip = await readZipIndex(blob);
  const pkl = zip.names.find((n) => n.endsWith('/data.pkl') && n.split('/').length === 2);
  if (!pkl) throw corrupt('data.pkl not found');
  const prefix = pkl.slice(0, -'data.pkl'.length);
  const root = unpickle(await bytesOf((await zip.entry(pkl))!));
  return {
    root,
    async read(t) {
      const st = await zip.entry(prefix + 'data/' + t.storage);
      if (!st) throw corrupt('missing storage ' + t.storage);
      const w = DTYPE_BYTES[t.dtype];
      const start = t.offset * w, n = numel(t.shape) * w;
      if (start + n > st.size) throw corrupt('storage too small for tensor');
      return bytesOf(st.slice(start, start + n));
    },
  };
}

/** fp16 (little-endian bytes) -> fp32, written into dst. */
export function halfToFloat(src: Uint8Array, dst: Float32Array): void {
  const n = src.byteLength >> 1;
  const F16 = (globalThis as { Float16Array?: new (b: ArrayBufferLike, o: number, n: number) => ArrayLike<number> }).Float16Array;
  if (F16 && src.byteOffset % 2 === 0) { dst.set(new F16(src.buffer, src.byteOffset, n)); return; }
  for (let i = 0; i < n; i++) {
    const x = src[2 * i] | (src[2 * i + 1] << 8);
    const s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 31, m = x & 1023;
    dst[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
}

/** Converts tensor bytes of any float dtype to fp32 into dst. */
export function toFloat32(dtype: DType, src: Uint8Array, dst: Float32Array): void {
  if (dtype === 'float16') return halfToFloat(src, dst);
  if (dtype === 'float32') { dst.set(new Float32Array(src.buffer, src.byteOffset, src.byteLength >> 2)); return; }
  if (dtype === 'bfloat16') {
    const u = new Uint32Array(dst.buffer, dst.byteOffset, dst.length);
    for (let i = 0; i < dst.length; i++) u[i] = (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
    return;
  }
  throw corrupt(`tensor dtype ${dtype} is not a float type`);
}
