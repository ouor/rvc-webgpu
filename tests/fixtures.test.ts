// Tests against core dev A's fixtures (tests/fixtures, ARCHITECTURE §17.6). Model cases run on ORT WASM in Node.
import { existsSync, openAsBlob, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { FIXTURES, EXTERNAL, hasFixture, loadFixture, relErr, snrDb } from './helpers';
import { decodeHidden, extractF0 } from '../src/engine/rvc/f0';
import { searchLabels, searchIndex } from '../src/engine/rvc/index-search';
import { extractFeatures } from '../src/engine/rvc/features';
import { prepareConvert, synthesizeAll } from '../src/engine/rvc/pipeline';
import { createTemplateNet } from '../src/engine/rvc/net';
import { segments, separateVR, type VrBandParams, type VrMeta } from '../src/engine/separate/vr';
import { readFaissIndex } from '../src/formats/faiss';
import { readPth } from '../src/formats/pth';
import { readRvcCheckpoint } from '../src/formats/rvc-checkpoint';
import { createSessionFromParts, tensorMap, type Manifest } from '../src/models/loader';
import { ort, releaseSession, type OrtSession } from '../src/models/ort';
import type { RunCtx } from '../src/engine/ctx';
import type { Prng } from '../src/dsp/prng';
import { DEFAULT_CONVERT, type RvcModelInfo } from '../src/shared/types';
import { resample } from '../src/dsp/resample';

const DIST = join(import.meta.dirname, '..', 'weights', 'dist');
const KYUBEY = join(EXTERNAL, 'weight', 'rvc-kyubey.pth');
const KYUBEY_INDEX = join(EXTERNAL, 'weight', 'rvc-kyubey.index');
const fx = (name: string) => (hasFixture(name) ? it : it.skip);
// Skips (not passes) when the fixture, weights/dist or a listed model file is missing (R1 L1).
const fxModel = (name: string, ...files: string[]) =>
  (hasFixture(name) && existsSync(join(DIST, 'registry.json')) && files.every((f) => existsSync(f)) ? it : it.skip);
const manifestOf = (p: string) => JSON.parse(readFileSync(join(DIST, p + '.manifest.json'), 'utf8')) as Manifest;

async function open(p: string): Promise<OrtSession> {
  const pth = await readPth(await openAsBlob(join(DIST, p + '.pth')));
  const graph = new Uint8Array(readFileSync(join(DIST, p + '.onnx')));
  return (await createSessionFromParts({ graph, manifest: manifestOf(p), pth, tensors: tensorMap(pth.root) }, 'wasm', 'CACHE_INTEGRITY')).session;
}

const ctx = (rng?: Prng): RunCtx => ({ backend: 'wasm', progress: () => {}, throwIfCancelled: () => {}, rng: rng ?? { next: () => 0, gaussian: (n) => new Float32Array(n) } });

/** Feeds fixture rnd [192, ΣT] / noise [ΣT·upp] chunk by chunk, in the order the template net asks (rnd, noise). */
function fixedNoise(rnd: Float32Array, noise: Float32Array, upp: number): Prng {
  const totalT = rnd.length / 192;
  let col = 0, pending: Float32Array | undefined;
  return {
    next: () => 0,
    gaussian(n) {
      if (pending) { const x = pending; pending = undefined; expect(x.length).toBe(n); return x; }
      const T = n / 192;
      const r = new Float32Array(n);
      for (let k = 0; k < 192; k++) r.set(rnd.subarray(k * totalT + col, k * totalT + col + T), k * T);
      pending = noise.slice(col * upp, (col + T) * upp);
      col += T;
      return r;
    },
  };
}

/** Pearson correlation of log f0 over frames voiced in both (b shifted by `off` frames). */
function logF0Corr(a: ArrayLike<number>, b: ArrayLike<number>, off: number): number {
  let n = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length && i + off < b.length; i++) {
    if (!(a[i] > 0 && b[i + off] > 0)) continue;
    const x = Math.log(a[i]), y = Math.log(b[i + off]);
    n++; sa += x; sb += y; sab += x * y; saa += x * x; sbb += y * y;
  }
  return (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
}

beforeAll(() => { ort.env.wasm.numThreads = 8; });

describe('fixtures: pure JS', () => {
  it('VR segments start on the patch grid and overlap by 4 s', () => {
    const unit = 480 * 256, n = 217 * 44100;
    const segs = segments(n, 44100, unit);
    expect(segs[0][0]).toBe(0);
    expect(segs.at(-1)![1]).toBe(n);
    for (let k = 1; k < segs.length; k++) {
      expect(segs[k][0] % unit).toBe(0);
      expect(segs[k - 1][1] - segs[k][0]).toBe(4 * 44100);
    }
  });
  fx('rmvpe-decode')('rmvpe decode', () => {
    const f = loadFixture('rmvpe-decode');
    const h = f.inputs.hidden as Float32Array;
    expect(relErr(decodeHidden(h, h.length / 360), f.outputs.f0 as Float32Array)).toBeLessThan(1e-5);
  });
  for (const c of ['faiss-ivf', 'faiss-flat']) fx(c)(`${c}: labels and distances match python search`, async () => {
    const f = loadFixture(c);
    const index = await readFaissIndex(await openAsBlob(join(FIXTURES, c, f.meta.indexFile as string)));
    const q = f.inputs.queries as Float32Array;
    const r = searchLabels(index, q, q.length / index.d);
    expect(Array.from(r.labels)).toEqual(Array.from(f.outputs.labels as BigInt64Array, Number));
    expect(relErr(r.distances, f.outputs.distances as Float32Array)).toBeLessThan(1e-4);
  });
});

describe('fixtures: ONNX (WASM)', () => {
  fxModel('uvr-patch')('uvr patch mask', async () => {
    const f = loadFixture('uvr-patch');
    const s = await open('uvr/uvr-hp-vocal');
    try {
      const m = (await s.run({ x: new ort.Tensor('float32', f.inputs.x as Float32Array, f.shapes.x) })).mask.data as Float32Array;
      expect(snrDb(m, f.outputs.mask as Float32Array)).toBeGreaterThan(40);
    } finally { await releaseSession(s); }
  });

  for (const c of ['rvc-template-v2-f0-40k', 'rvc-template-v2-f0-40k-f0zero']) fxModel(c, KYUBEY)(c, async () => {
    const f = loadFixture(c);
    const ckpt = await readRvcCheckpoint(await openAsBlob(KYUBEY));
    const info: RvcModelInfo = { id: 'k', name: 'k', source: 'pth', version: 'v2', f0: true, sr: 40000, nSpk: ckpt.nSpk, bytes: 0 };
    const net = await createTemplateNet(info, ckpt, { graph: new Uint8Array(readFileSync(join(DIST, 'rvc/v2-f0-40k.onnx'))), manifest: manifestOf('rvc/v2-f0-40k') }, 'wasm');
    try {
      const queue = [f.inputs.rnd as Float32Array, f.inputs.noise as Float32Array];
      const out = await net.run({ phone: f.inputs.phone as Float32Array, frames: 200, pitch: f.inputs.pitch as BigInt64Array, pitchf: f.inputs.pitchf as Float32Array, sid: 0 },
        { next: () => 0, gaussian: () => queue.shift()! });
      expect(snrDb(out, f.outputs.audio as Float32Array)).toBeGreaterThan(c.endsWith('f0zero') ? 60 : 30);
    } finally { await net.release(); }
  });

  fxModel('contentvec-10s')('contentvec 10 s', async () => {
    const f = loadFixture('contentvec-10s');
    const s = await open('contentvec/contentvec');
    try {
      for (const v of ['v1', 'v2'] as const) {
        const r = await extractFeatures(f.inputs.source as Float32Array, s, v);
        const ref = f.outputs[v === 'v1' ? 'feats256' : 'feats768'] as Float32Array;
        expect(snrDb(r.data, ref)).toBeGreaterThan(40);
      }
    } finally { await releaseSession(s); }
  });

  fxModel('rmvpe-10s')('rmvpe 10 s through extractF0', async () => {
    const f = loadFixture('rmvpe-10s');
    const s = await open('rmvpe/rmvpe');
    try {
      const f0 = await extractF0(f.inputs.audio as Float32Array, s, ctx());
      const ref = f.outputs.f0 as Float32Array;
      let agree = 0;
      for (let i = 0; i < ref.length; i++) if ((f0[i] > 0) === (ref[i] > 0)) agree++;
      expect(agree / ref.length).toBeGreaterThan(0.99);
      expect(relErr(f0, ref)).toBeLessThan(1e-3);
    } finally { await releaseSession(s); }
  });

  fxModel('uvr-song-30s')('separateVR 30 s vs AudioPre reference (SNR > 30 dB)', async () => {
    const f = loadFixture('uvr-song-30s');
    const n = f.shapes.input44k[1];
    const x = f.inputs.input44k as Float32Array;
    const s = await open('uvr/uvr-hp-vocal');
    try {
      const mp = JSON.parse(readFileSync(join(DIST, 'uvr/4band_v2.json'), 'utf8')) as VrBandParams;
      const meta = manifestOf('uvr/uvr-hp-vocal').meta as unknown as VrMeta;
      const r = await separateVR({ sr: 44100, l: x.subarray(0, n), r: x.subarray(n) }, s, mp, meta, { aggressiveness: 10 }, ctx());
      const m = f.shapes.vocals44k[1];
      const voc = f.outputs.vocals44k as Float32Array, inst = f.outputs.inst44k as Float32Array;
      expect(snrDb(r.vocals.l.subarray(0, m), voc.subarray(0, m))).toBeGreaterThan(30);
      expect(snrDb(r.vocals.r.subarray(0, m), voc.subarray(m))).toBeGreaterThan(30);
      expect(snrDb(r.instrumental.l.subarray(0, m), inst.subarray(0, m))).toBeGreaterThan(30);
    } finally { await releaseSession(s); }
  });

  fxModel('pipeline-voice-20s', KYUBEY, KYUBEY_INDEX)('voice pipeline 20 s (rmvpe + contentvec + index + synth)', async () => {
    const f = loadFixture('pipeline-voice-20s');
    const cache = prepareConvert(f.inputs.vocal16k as Float32Array, 'normal');
    const rm = await open('rmvpe/rmvpe');
    try {
    cache.f0Hz = await extractF0(cache.padded, rm, ctx());
    expect(relErr(cache.f0Hz.subarray(0, f.outputs.f0Hz.length), f.outputs.f0Hz as Float32Array)).toBeLessThan(1e-3);
    const hu = await open('contentvec/contentvec');
    try { cache.feats.v2 = [await extractFeatures(cache.padded.slice(cache.chunks[0].start, cache.chunks[0].end), hu, 'v2')]; } finally { await releaseSession(hu); }
    expect(snrDb(cache.feats.v2[0].data, f.outputs.feats as Float32Array)).toBeGreaterThan(40);
    const index = await readFaissIndex(await openAsBlob(KYUBEY_INDEX));
    expect(snrDb(searchIndex(cache.feats.v2[0], index).map((v, i) => v * 0.75 + 0.25 * cache.feats.v2![0].data[i]), f.outputs.featsIr75 as Float32Array)).toBeGreaterThan(40);
    const ckpt = await readRvcCheckpoint(await openAsBlob(KYUBEY));
    const info: RvcModelInfo = { id: 'k', name: 'k', source: 'pth', version: 'v2', f0: true, sr: 40000, nSpk: ckpt.nSpk, bytes: 0 };
    const net = await createTemplateNet(info, ckpt, { graph: new Uint8Array(readFileSync(join(DIST, 'rvc/v2-f0-40k.onnx'))), manifest: manifestOf('rvc/v2-f0-40k') }, 'wasm');
    try {
      const out = await synthesizeAll(cache, net, { ...DEFAULT_CONVERT }, ctx(fixedNoise(f.inputs.rnd as Float32Array, f.inputs.noise as Float32Array, 400)), { id: 'k', index });
      expect(out.data.length).toBe(f.outputs.converted.length);
      expect(snrDb(out.data, f.outputs.converted as Float32Array)).toBeGreaterThan(30); // measured 37 dB with the float64 phase
      // AC-2: log-f0 of the output vs the input (+t_pad frame offset). The sine phase now comes from JS float64 (sinePhase).
      const fo = await extractF0(resample(out.data, 40000, 16000), rm, ctx());
      expect(logF0Corr(fo, cache.f0Hz, 100)).toBeGreaterThan(0.95); // measured 0.996
    } finally { await net.release(); }
    } finally { await releaseSession(rm); }
  });
});
