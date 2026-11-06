// Trust boundaries (R1 M1, M2, M7-M10): every failure on user files is an AppError with an expected code,
// and the worker boundary always delivers AppError.
import * as Comlink from 'comlink';
import { zipSync } from 'fflate';
import { describe, expect, it } from 'vitest';
import { readModelArchive } from '../src/formats/archive';
import { readFaissIndex } from '../src/formats/faiss';
import { readRvcCheckpoint } from '../src/formats/rvc-checkpoint';
import { readOnnxInfo, type OnnxInfo } from '../src/formats/onnx-meta';
import { describeOnnx } from '../src/engine/rvc/net';
import { classifyOrtError } from '../src/models/ort';
import { AppError, toAppError } from '../src/shared/errors';
import { installErrorTransfer } from '../src/shared/rpc';
import { fakeFlatIndex, fakeRvcPth } from './helpers';

const codeOf = async (f: () => Promise<unknown>) => {
  try { await f(); return 'OK'; } catch (e) {
    if (!(e instanceof AppError)) throw new Error(`not an AppError: ${String(e)}`);
    return e.code;
  }
};

/** Truncations and bit flips of `good`, deterministic. */
function* mutants(good: Uint8Array, n: number) {
  let s = 12345;
  const rnd = (m: number) => { s = (s * 1103515245 + 12345) >>> 0; return s % m; };
  for (let k = 0; k < n; k++) {
    const b = good.slice(0, k % 3 === 0 ? rnd(good.length) : good.length);
    for (let f = 0; f < 1 + rnd(4) && b.length; f++) b[rnd(b.length)] ^= 1 << rnd(8);
    yield b;
  }
}

describe('corrupt inputs give AppError (M10)', () => {
  it('.pth', async () => {
    const good = new Uint8Array(await (await fakeRvcPth()).arrayBuffer());
    for (const b of mutants(good, 300)) {
      expect(['OK', 'PTH_NOT_ZIP', 'PTH_CORRUPT', 'PTH_UNSAFE', 'PTH_TRAINING_CKPT', 'PTH_UNSUPPORTED_ARCH', 'PTH_UNSUPPORTED_VOCODER', 'PTH_UNSUPPORTED_EMBEDDER'])
        .toContain(await codeOf(() => readRvcCheckpoint(new Blob([b]))));
    }
  });

  it('faiss index', async () => {
    const good = new Uint8Array(await fakeFlatIndex(256, 8).arrayBuffer());
    for (const b of mutants(good, 300)) {
      expect(['OK', 'INDEX_UNSUPPORTED', 'INDEX_EMPTY', 'INDEX_DIM_MISMATCH']).toContain(await codeOf(() => readFaissIndex(new Blob([b]))));
    }
  });

  it('model zip', async () => {
    const good = zipSync({ 'm/model.pth': new Uint8Array(await (await fakeRvcPth()).arrayBuffer()), 'm/added.index': new Uint8Array(64) });
    for (const b of mutants(good, 200)) {
      expect(['OK', 'ZIP_NO_MODEL']).toContain(await codeOf(() => readModelArchive(new Blob([b]))));
    }
  });

  it('pth tensor records and emb_g shape are validated (M8)', async () => {
    expect(await codeOf(async () => readRvcCheckpoint(await fakeRvcPth({ embShape: [-1, 256] })))).toBe('PTH_CORRUPT');
    expect(await codeOf(async () => readRvcCheckpoint(await fakeRvcPth({ embShape: [256] })))).toBe('PTH_CORRUPT');
    expect(await codeOf(async () => readRvcCheckpoint(await fakeRvcPth()))).toBe('OK');
  });

  it('faiss rejects other dimensions from the header (M9)', async () => {
    expect(await codeOf(() => readFaissIndex(fakeFlatIndex(100, 4)))).toBe('INDEX_DIM_MISMATCH');
    expect(await codeOf(() => readFaissIndex(fakeFlatIndex(768, 4)))).toBe('OK');
  });

  it('zip entries with a zip-bomb ratio are refused before extraction (M7)', async () => {
    const bomb = zipSync({ 'model.pth': new Uint8Array(4 << 20) }, { level: 9 });
    expect(await codeOf(() => readModelArchive(new Blob([bomb])))).toBe('ZIP_NO_MODEL');
  });

  it('readOnnxInfo on junk throws (callers map it to ONNX_UNRECOGNIZED)', () => {
    for (const b of mutants(new Uint8Array([8, 7, 18, 4, 116, 101, 115, 116]), 50)) {
      try { readOnnxInfo(b); } catch { /* any throw is fine; it must not hang */ }
    }
  });
});

describe('describeOnnx (M10)', () => {
  const io = (name: string, dims: (number | string)[] = [1]) => ({ name, elemType: 1, dims });
  const onnx = (names: string[], c = 768, meta: Record<string, string> = {}): OnnxInfo =>
    ({ metadata: meta, inputs: [io('phone', [1, 'T', c]), io('phone_lengths'), ...names.map((n) => io(n))], outputs: [] });
  it.each([
    [['ds', 'rnd', 'pitch', 'pitchf'], 768, { schema: 'webui', f0: true, version: 'v2' }],
    [['ds', 'rnd'], 256, { schema: 'webui', f0: false, version: 'v1' }],
    [['sid', 'rate', 'pitch', 'nsff0'], 768, { schema: 'applio', f0: true, version: 'v2' }],
    [['sid', 'rate'], 768, { schema: 'applio', f0: false, version: 'v2' }],
  ] as const)('%j', (names, c, want) => {
    expect(describeOnnx(onnx([...names], c))).toMatchObject(want);
  });
  it('reads sr from metadata, rejects unknown schemas', () => {
    expect(describeOnnx(onnx(['ds', 'rnd'], 768, { model_info: '{"sr":48000}' })).sr).toBe(48000);
    expect(() => describeOnnx(onnx(['x']))).toThrow(AppError);
  });
});

describe('error classification and the worker boundary (M1, M2)', () => {
  it('allocation failures are OOM wherever they come from; WASM traps are not', () => {
    expect(toAppError(new RangeError('Array buffer allocation failed')).code).toBe('OOM');
    expect(toAppError(new Error('WebAssembly.Memory.grow(): Maximum memory size exceeded')).code).toBe('OOM');
    expect(toAppError(new RangeError('Offset is outside the bounds of the DataView')).code).toBe('INTERNAL');
    expect(classifyOrtError(new Error('RuntimeError: memory access out of bounds')).code).toBe('INTERNAL');
    expect(classifyOrtError(new Error('GPUDevice was lost')).code).toBe('GPU_LOST');
  });

  it('serialize/deserialize keeps AppError fields and turns any other throwable into an AppError', () => {
    installErrorTransfer();
    const h = Comlink.transferHandlers.get('throw')!;
    const roundTrip = (value: unknown) => {
      const [wire] = h.serialize({ value });
      try { h.deserialize(wire); } catch (e) { return e as AppError; }
      throw new Error('deserialize did not throw');
    };
    const a = roundTrip(new AppError('CACHE_INTEGRITY', 'bad', 'file.pth', true));
    expect(a).toBeInstanceOf(AppError);
    expect(a).toMatchObject({ code: 'CACHE_INTEGRITY', detail: 'file.pth', recoverable: true });
    expect(roundTrip(new TypeError('x is undefined'))).toMatchObject({ code: 'INTERNAL', message: 'x is undefined' });
    expect(roundTrip('plain string')).toMatchObject({ code: 'INTERNAL' });
  });
});
