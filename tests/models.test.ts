// ONNX stage equivalence on onnxruntime-web WASM (Node) against A's reference run, one chunk each.
// Needs weights/dist (A's artifacts) and weights/reference; slow (~1 min), skipped otherwise.
import { existsSync, openAsBlob, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { ort, releaseSession, type OrtSession } from '../src/models/ort';
import { createSessionFromParts, tensorMap, type Manifest } from '../src/models/loader';
import { readPth } from '../src/formats/pth';
import { readRvcCheckpoint } from '../src/formats/rvc-checkpoint';
import { extractFeatures } from '../src/engine/rvc/features';
import { decodeHidden } from '../src/engine/rvc/f0';
import { createTemplateNet } from '../src/engine/rvc/net';
import { synthesizeChunk } from '../src/engine/rvc/pipeline';
import { EXTERNAL, loadFixture, snrDb } from './helpers';
import type { Prng } from '../src/dsp/prng';
import type { RvcModelInfo } from '../src/shared/types';

const ROOT = join(import.meta.dirname, '..', 'weights');
const DIST = join(ROOT, 'dist'), REF = join(ROOT, 'reference');
const ok = existsSync(join(DIST, 'registry.json')) && existsSync(join(REF, 'bunnygirl-rvc-kyubey', 'case.json'));

async function open(dir: string, name: string): Promise<OrtSession> {
  const graph = new Uint8Array(readFileSync(join(DIST, dir, name + '.onnx')));
  const manifest = JSON.parse(readFileSync(join(DIST, dir, name + '.manifest.json'), 'utf8')) as Manifest;
  const pth = await readPth(await openAsBlob(join(DIST, dir, name + '.pth')));
  return (await createSessionFromParts({ graph, manifest, pth, tensors: tensorMap(pth.root) }, 'wasm', 'CACHE_INTEGRITY')).session;
}

function cosine(a: Float32Array, b: Float32Array, dim: number) {
  let worst = 1;
  for (let t = 0; t < a.length / dim; t++) {
    let ab = 0, aa = 0, bb = 0;
    for (let j = t * dim; j < (t + 1) * dim; j++) { ab += a[j] * b[j]; aa += a[j] ** 2; bb += b[j] ** 2; }
    worst = Math.min(worst, ab / Math.sqrt(aa * bb));
  }
  return worst;
}

(ok ? describe : describe.skip)('ONNX stages vs reference (WASM)', () => {
  const f = ok ? loadFixture('bunnygirl-rvc-kyubey', REF) : undefined!;
  const chunks = f?.meta.chunks as { audio: [number, number]; f0Frames: [number, number]; featFrames: number; T: number }[];
  const last = chunks?.length - 1;
  const featRow0 = chunks?.slice(0, last).reduce((a, c) => a + c.featFrames, 0);
  const pad = f?.outputs.audioPad16k as Float32Array;
  beforeAll(() => { ort.env.wasm.numThreads = 8; });

  it('contentvec feats768 (last chunk) cosine > 0.999', async () => {
    const s = await open('contentvec', 'contentvec');
    try {
      const [a, b] = chunks[last].audio;
      const feats = await extractFeatures(pad.slice(a, b), s, 'v2');
      expect(feats.frames).toBe(chunks[last].featFrames);
      const ref = (f.outputs.feats as Float32Array).subarray(featRow0 * 768, (featRow0 + feats.frames) * 768);
      expect(cosine(feats.data, ref, 768)).toBeGreaterThan(0.999);
    } finally { await releaseSession(s); }
  });

  it('rmvpe hidden (first 32 s piece) -> f0 within 5 cents on voiced frames', async () => {
    const s = await open('rmvpe', 'rmvpe');
    try {
      const piece = pad.slice(0, 32 * 16000);
      const r = await s.run({ audio: new ort.Tensor('float32', piece, [1, piece.length]) });
      const frames = Math.floor(piece.length / 160) + 1;
      expect(r.hidden.dims).toEqual([1, frames, 360]);
      const f0 = decodeHidden(r.hidden.data as Float32Array, frames);
      const ref = f.outputs.f0Hz as Float32Array;
      let n = 0, cents = 0, agree = 0;
      const own = 30 * 100; // frames before the 2 s context tail
      for (let t = 0; t < own; t++) {
        if ((f0[t] > 0) === (ref[t] > 0)) agree++;
        if (f0[t] > 0 && ref[t] > 0) { n++; cents += Math.abs(1200 * Math.log2(f0[t] / ref[t])); }
      }
      expect(agree / own).toBeGreaterThan(0.99);
      expect(cents / n).toBeLessThan(5);
    } finally { await releaseSession(s); }
  });

  it.skipIf(!existsSync(join(EXTERNAL, 'weight', 'rvc-kyubey.pth')))('template synthesis (last chunk, reference rnd/noise) matches raw output', async () => {
    const blob = await openAsBlob(join(EXTERNAL, 'weight', 'rvc-kyubey.pth'));
    const ckpt = await readRvcCheckpoint(blob);
    const info: RvcModelInfo = { id: 'k', name: 'kyubey', source: 'pth', version: 'v2', f0: true, sr: 40000, nSpk: ckpt.nSpk, bytes: blob.size };
    const graph = new Uint8Array(readFileSync(join(DIST, 'rvc', 'v2-f0-40k.onnx')));
    const manifest = JSON.parse(readFileSync(join(DIST, 'rvc', 'v2-f0-40k.manifest.json'), 'utf8')) as Manifest;
    const net = await createTemplateNet(info, ckpt, { graph, manifest }, 'wasm');
    try {
      const c = chunks[last];
      const T = c.T, upp = 400;
      const col0 = chunks.slice(0, last).reduce((a, x) => a + x.T, 0);
      const rndAll = f.inputs.rnd as Float32Array, totalT = rndAll.length / 192;
      const rnd = new Float32Array(192 * T);
      for (let k = 0; k < 192; k++) rnd.set(rndAll.subarray(k * totalT + col0, k * totalT + col0 + T), k * T);
      const noise = (f.inputs.noise as Float32Array).subarray(col0 * upp, (col0 + T) * upp);
      const queue = [rnd, noise];
      const rng: Prng = { next: () => 0, gaussian: (n) => { const x = queue.shift()!; expect(x.length).toBe(n); return x; } };
      const feats = { data: (f.outputs.feats as Float32Array).slice(featRow0 * 768, (featRow0 + c.featFrames) * 768), frames: c.featFrames, dim: 768 as const };
      const pitch = (f.outputs.pitch as BigInt64Array).subarray(c.f0Frames[0]);
      const pitchf = (f.outputs.pitchf as Float32Array).subarray(c.f0Frames[0]);
      const p = { f0UpKey: 0, indexRate: 0, protect: 0.33, rmsMixRate: 0.25, sid: 0, chunkProfile: 'normal' as const, seed: 0 };
      const audio1 = await synthesizeChunk(c.audio[1] - c.audio[0], feats, undefined, { pitch, pitchf }, net, p, rng);
      const trimmed = audio1.slice(40000, audio1.length - 40000);
      const raw = f.outputs.rawIr0 as Float32Array;
      const ref = raw.subarray(raw.length - trimmed.length);
      expect(snrDb(trimmed, ref)).toBeGreaterThan(25); // NSF phase drift (ARCHITECTURE §12): waveform SNR is only a sanity floor
    } finally { await net.release(); }
  });
});
