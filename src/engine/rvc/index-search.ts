// faiss IVF nprobe search + rvc-nano weighting (pipeline.py vc(): index.search k=8, weight = (1/score)^2).
import type { IvfIndex } from '../../formats/faiss';
import type { Feats } from './features';

const K = 8;

/** Squared L2 between q (offset qo) and v (offset vo), length d. */
function l2(q: Float32Array, qo: number, v: Float32Array, vo: number, d: number): number {
  let s = 0;
  for (let j = 0; j < d; j++) { const x = q[qo + j] - v[vo + j]; s += x * x; }
  return s;
}

/** Top-k (k=8) for query row t: fills dist (squared L2, ascending) and list/row (-1 = no neighbour). */
function knn(index: IvfIndex, q: Float32Array, qo: number, cd: Float64Array, probe: Int32Array,
  dist: Float64Array, list: Int32Array, row: Int32Array): void {
  const { d, nlist, nprobe, centroids, lists } = index;
  if (nlist === 1) probe[0] = 0;
  else { // coarse: nprobe closest centroids
    for (let c = 0; c < nlist; c++) cd[c] = l2(q, qo, centroids, c * d, d);
    for (let p = 0; p < nprobe; p++) {
      let m = 0;
      for (let c = 1; c < nlist; c++) if (cd[c] < cd[m]) m = c;
      probe[p] = m;
      cd[m] = Infinity;
    }
  }
  dist.fill(Infinity);
  list.fill(-1);
  for (let p = 0; p < nprobe; p++) {
    const v = lists[probe[p]];
    const n = v.length / d;
    for (let r = 0; r < n; r++) {
      const x = l2(q, qo, v, r * d, d);
      if (x >= dist[K - 1]) continue;
      let i = K - 1;
      while (i > 0 && dist[i - 1] > x) { dist[i] = dist[i - 1]; list[i] = list[i - 1]; row[i] = row[i - 1]; i--; }
      dist[i] = x; list[i] = probe[p]; row[i] = r;
    }
  }
}

function scratch(index: IvfIndex) {
  return { cd: new Float64Array(index.nlist), probe: new Int32Array(index.nprobe), dist: new Float64Array(K), list: new Int32Array(K), row: new Int32Array(K) };
}

/** faiss index.search(q, k=8): distances and labels, for tests against python. */
export function searchLabels(index: IvfIndex, q: Float32Array, n: number): { distances: Float32Array; labels: Int32Array } {
  const s = scratch(index);
  const distances = new Float32Array(n * K), labels = new Int32Array(n * K).fill(-1);
  for (let t = 0; t < n; t++) {
    knn(index, q, t * index.d, s.cd, s.probe, s.dist, s.list, s.row);
    for (let i = 0; i < K; i++) {
      distances[t * K + i] = s.dist[i];
      if (s.list[i] >= 0) labels[t * K + i] = index.ids[s.list[i]][s.row[i]];
    }
  }
  return { distances, labels };
}

/** For every frame: weighted average of its k=8 nearest index vectors (the `npy` of pipeline.py). */
export function searchIndex(feats: Feats, index: IvfIndex): Float32Array {
  const d = index.d, q = feats.data;
  const out = new Float32Array(q.length);
  const { cd, probe, dist, list, row } = scratch(index);
  for (let t = 0; t < feats.frames; t++) {
    const qo = t * d;
    knn(index, q, qo, cd, probe, dist, list, row);
    let wsum = 0;
    for (let i = 0; i < K; i++) if (list[i] >= 0) wsum += 1 / Math.max(dist[i], 1e-30) ** 2;
    if (!(wsum > 0) || !Number.isFinite(wsum)) { out.set(q.subarray(qo, qo + d), qo); continue; } // no neighbours / exact hit
    for (let i = 0; i < K; i++) {
      if (list[i] < 0) continue;
      const w = 1 / Math.max(dist[i], 1e-30) ** 2 / wsum;
      const v = index.lists[list[i]], vo = row[i] * d;
      for (let j = 0; j < d; j++) out[qo + j] += v[vo + j] * w;
    }
  }
  return out;
}

/** feats = npy * rate + (1 - rate) * feats */
export function blendIndex(feats: Feats, retrieved: Float32Array, rate: number): Feats {
  const data = new Float32Array(feats.data.length);
  for (let i = 0; i < data.length; i++) data[i] = retrieved[i] * rate + (1 - rate) * feats.data[i];
  return { ...feats, data };
}
