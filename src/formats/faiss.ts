// faiss index reader for what RVC/Applio write: IndexIVFFlat ("IwFl") and IndexFlatL2 ("IxF2").
// Format: external/faiss/faiss/impl/index_read.cpp (read_index_header, read_ivf_header, read_direct_map, "ilar").
// Reads the Blob in slices, so a 350MB index never needs a second full copy in memory.
import { AppError } from '../shared/errors';

export interface IvfIndex {
  d: number;
  nlist: number; // 1 for a flat index
  nprobe: number;
  centroids: Float32Array; // nlist * d (empty for a flat index)
  lists: Float32Array[]; // vectors per list, row-major n * d
  ids: Int32Array[]; // original row (faiss label) of each vector, per list
  nVectors: number; // vectors kept
  ntotal: number; // vectors in the file
}

export const MAX_VECTORS = 200_000; // ARCHITECTURE §9.3 memory cap (~600MB at 768 dims)

const unsupported = (m: string) => new AppError('INDEX_UNSUPPORTED', m);
const fourcc = (s: string) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24);

class Cursor {
  pos = 0;
  constructor(private blob: Blob) {}
  async take(n: number): Promise<ArrayBuffer> {
    if (n < 0 || this.pos + n > this.blob.size) throw new AppError('INDEX_UNSUPPORTED', 'truncated index file');
    const b = await this.blob.slice(this.pos, this.pos + n).arrayBuffer();
    this.pos += n;
    return b;
  }
  async u8() { return new Uint8Array(await this.take(1))[0]; }
  async i32() { return new DataView(await this.take(4)).getInt32(0, true); }
  async u64() { return Number(new DataView(await this.take(8)).getBigUint64(0, true)); }
  skip(n: number) { this.pos += n; }
}

async function readHeader(c: Cursor) {
  const d = await c.i32();
  const ntotal = await c.u64();
  c.skip(16); // two dummy idx_t
  await c.u8(); // is_trained
  const metric = await c.i32();
  if (metric !== 1) throw unsupported(`metric ${metric} (only L2 is supported)`);
  // RVC indexes hold v1 (256) or v2 (768) features; checked before any vector is read (R1 M9)
  if (d !== 256 && d !== 768) throw new AppError('INDEX_DIM_MISMATCH', `index dimension ${d}`);
  return { d, ntotal };
}

/** IxF2 body after the fourcc: header + float vector (size stored as float count). */
async function readFlat(c: Cursor) {
  const { d, ntotal } = await readHeader(c);
  const n = await c.u64();
  const floats = n === ntotal * d ? n : n / 4; // older writers stored the byte count
  if (floats !== ntotal * d) throw unsupported('flat index size mismatch');
  return { d, ntotal, data: new Float32Array(await c.take(floats * 4)) };
}

/** Deterministic keep-mask for down-sampling (xorshift32). */
function keeper(keep: number, total: number) {
  if (total <= keep) return () => true;
  let s = 0x9e3779b9;
  const p = keep / total;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296 < p; };
}

function sample(vecs: Float32Array, ids: Int32Array, d: number, take: () => boolean): [Float32Array, Int32Array] {
  const n = vecs.length / d;
  const keep = new Uint8Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) if ((keep[i] = take() ? 1 : 0)) m++;
  const out = new Float32Array(m * d), oid = new Int32Array(m); // exact size: no second full copy
  for (let i = 0, k = 0; i < n; i++) if (keep[i]) { out.set(vecs.subarray(i * d, i * d + d), k * d); oid[k++] = ids[i]; }
  return [out, oid];
}

const iota = (n: number) => Int32Array.from({ length: n }, (_, i) => i);

export async function readFaissIndex(blob: Blob, cap = MAX_VECTORS): Promise<IvfIndex> {
  const c = new Cursor(blob);
  if (blob.size < 4) throw unsupported('not a faiss index');
  const h = await c.i32();
  if (h === fourcc('IxF2')) {
    const f = await readFlat(c);
    if (f.ntotal === 0) throw new AppError('INDEX_EMPTY', 'index has no vectors (trained_*.index?)');
    const [vecs, ids] = f.ntotal > cap ? sample(f.data, iota(f.ntotal), f.d, keeper(cap, f.ntotal)) : [f.data, iota(f.ntotal)];
    return { d: f.d, nlist: 1, nprobe: 1, centroids: new Float32Array(0), lists: [vecs], ids: [ids], nVectors: ids.length, ntotal: f.ntotal };
  }
  if (h !== fourcc('IwFl')) throw unsupported('unsupported faiss index type 0x' + (h >>> 0).toString(16));

  const { d, ntotal } = await readHeader(c);
  const nlist = await c.u64();
  const nprobe = await c.u64();
  if (ntotal === 0) throw new AppError('INDEX_EMPTY', 'index has no vectors (trained_*.index?)');
  if (nlist < 1 || nlist > ntotal) throw unsupported(`nlist ${nlist} for ${ntotal} vectors`);
  if (await c.i32() !== fourcc('IxF2')) throw unsupported('IVF quantizer is not IndexFlatL2');
  const q = await readFlat(c);
  if (q.ntotal !== nlist || q.d !== d) throw unsupported('quantizer shape mismatch');
  const dmType = await c.u8(); // direct map: type, array, [hashtable]
  c.skip((await c.u64()) * 8);
  if (dmType === 2) c.skip((await c.u64()) * 16);

  const ilh = await c.i32();
  if (ilh !== fourcc('ilar')) throw unsupported('inverted lists are not "ilar"');
  const nl = await c.u64();
  const codeSize = await c.u64();
  if (nl !== nlist || codeSize !== d * 4) throw unsupported('inverted list shape mismatch');
  const sizes = new Array<number>(nlist).fill(0);
  const lt = await c.i32();
  const sv = await c.u64();
  const raw = new BigUint64Array(await c.take(sv * 8));
  if (lt === fourcc('full')) {
    if (sv !== nlist) throw unsupported('bad list sizes');
    raw.forEach((v, i) => (sizes[i] = Number(v)));
  } else if (lt === fourcc('sprs')) {
    for (let j = 0; j + 1 < raw.length; j += 2) {
      const l = Number(raw[j]);
      if (l >= nlist) throw unsupported('list number out of range');
      sizes[l] = Number(raw[j + 1]);
    }
  } else throw unsupported('unknown list size encoding');
  if (sizes.reduce((a, b) => a + b, 0) !== ntotal) throw unsupported('list sizes do not add up to ntotal');

  const take = keeper(cap, ntotal);
  const lists: Float32Array[] = [], idLists: Int32Array[] = [];
  let kept = 0;
  for (let i = 0; i < nlist; i++) {
    const n = sizes[i];
    if (n === 0) { lists.push(new Float32Array(0)); idLists.push(new Int32Array(0)); continue; }
    const v = new Float32Array(await c.take(n * codeSize));
    const id64 = new BigInt64Array(await c.take(n * 8)); // labels = rows of big_npy (reconstruct_n order)
    const ids = Int32Array.from(id64, Number);
    const [l, li] = ntotal > cap ? sample(v, ids, d, take) : [v, ids];
    kept += li.length;
    lists.push(l);
    idLists.push(li);
  }
  return { d, nlist, nprobe: Math.max(1, Math.min(nprobe, nlist)), centroids: q.data, lists, ids: idLists, nVectors: kept, ntotal };
}
