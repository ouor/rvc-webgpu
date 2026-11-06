// Stage-by-stage comparison with core dev A's reference run (weights/reference/bunnygirl-rvc-kyubey, tools/reference.py).
// Pure-JS stages run on the whole song; ONNX stages run one chunk on onnxruntime-web WASM (Node).
import { existsSync, openAsBlob } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { chunkPlan, chunkPoints, preprocess16k, reflectPad } from '../src/engine/rvc/pipeline';
import { decodeHidden, quantizeF0 } from '../src/engine/rvc/f0';
import { blendIndex, searchIndex } from '../src/engine/rvc/index-search';
import { changeRms } from '../src/dsp/rms';
import { readFaissIndex } from '../src/formats/faiss';
import { EXTERNAL, loadFixture, relErr, snrDb } from './helpers';

const REF = join(import.meta.dirname, '..', 'weights', 'reference');
const CASE = 'bunnygirl-rvc-kyubey';
const run = existsSync(join(REF, CASE, 'case.json')) ? describe : describe.skip;

run('reference: bunnygirl x kyubey (JS stages)', () => {
  const f = loadFixture(CASE, REF);
  const meta = f.meta as { optTs: number[]; chunks: { audio: [number, number]; f0Frames: [number, number]; featFrames: number; T: number }[] };
  const audio = preprocess16k(f.inputs.vocal16k as Float32Array);

  it('preprocess + reflect pad matches audioPad16k', () => {
    const padded = reflectPad(audio, 16000);
    expect(padded.length).toBe(f.outputs.audioPad16k.length);
    expect(relErr(padded, f.outputs.audioPad16k as Float32Array)).toBeLessThan(1e-5);
  });

  it('opt_ts and chunk plan match', () => {
    const opt = chunkPoints(audio, 'normal');
    expect(opt).toEqual(meta.optTs);
    const plan = chunkPlan(audio.length + 32000, opt, 'normal');
    expect(plan.map((c) => [c.start, c.end])).toEqual(meta.chunks.map((c) => c.audio));
    expect(plan.map((c) => c.f0Start)).toEqual(meta.chunks.map((c) => c.f0Frames[0]));
  });

  it('RMVPE decode and f0 quantization match', () => {
    const hidden = f.outputs.rmvpeHidden as Float32Array;
    const f0 = decodeHidden(hidden, hidden.length / 360);
    expect(relErr(f0, f.outputs.f0Hz as Float32Array)).toBeLessThan(1e-5);
    const n = f.outputs.pitch.length;
    const q = quantizeF0(f.outputs.f0Hz as Float32Array, 0);
    const pitch = f.outputs.pitch as BigInt64Array;
    let diff = 0;
    for (let i = 0; i < n; i++) if (q.pitch[i] !== pitch[i]) diff++;
    expect(diff).toBeLessThanOrEqual(2); // float32 storage of f0Hz can flip a rounding boundary
    expect(relErr(q.pitchf.subarray(0, n), f.outputs.pitchf as Float32Array)).toBeLessThan(1e-6);
  });

  it('change_rms + 0.99 peak limit matches', () => {
    const y = changeRms(audio, 16000, (f.outputs.rawIr75 as Float32Array).slice(), 40000, 0.25);
    let max = 0;
    for (const v of y) max = Math.max(max, Math.abs(v));
    if (max / 0.99 > 1) for (let i = 0; i < y.length; i++) y[i] /= max / 0.99;
    expect(snrDb(y, f.outputs.convertedIr75 as Float32Array)).toBeGreaterThan(80);
  });

  const indexPath = join(EXTERNAL, 'weight', 'rvc-kyubey.index');
  (existsSync(indexPath) ? it : it.skip)('IVF search + blend matches faiss (index_rate 0.75)', async () => {
    const index = await readFaissIndex(await openAsBlob(indexPath));
    expect(index.nVectors).toBe(index.ntotal);
    const feats = { data: f.outputs.feats as Float32Array, frames: f.outputs.feats.length / 768, dim: 768 as const };
    const npy = searchIndex(feats, index);
    const blended = blendIndex(feats, npy, 0.75);
    expect(snrDb(blended.data, f.outputs.featsIr75 as Float32Array)).toBeGreaterThan(60);
  });
});
