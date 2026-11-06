import { existsSync, openAsBlob } from 'node:fs';
import { join } from 'node:path';
import { zipSync, strToU8 } from 'fflate';
import { describe, expect, it } from 'vitest';
import { readPth, unpickle } from '../src/formats/pth';
import { readRvcCheckpoint, templateSignature } from '../src/formats/rvc-checkpoint';
import { pickArchiveEntries } from '../src/formats/archive';
import { readFaissIndex } from '../src/formats/faiss';
import { AppError } from '../src/shared/errors';
import { EXTERNAL } from './helpers';

const W = join(EXTERNAL, 'weight');
const has = (f: string) => existsSync(join(W, f));
const code = async (p: Promise<unknown> | (() => unknown)) => {
  try { await (typeof p === 'function' ? p() : p); } catch (e) { return (e as AppError).code; }
  return 'OK';
};

// tiny pickle builder: protocol 2 opcodes only
const u = (s: string) => [0x8c, s.length, ...strToU8(s)];
const pkl = (...ops: number[]) => new Uint8Array([0x80, 2, ...ops, 0x2e]);
const ptZip = (data: Uint8Array) => new Blob([zipSync({ 'archive/data.pkl': data }, { level: 0 })]);

describe('pickle allowlist (AC-23)', () => {
  it('refuses os.system via GLOBAL', async () => {
    const evil = pkl(0x63, ...strToU8('os\nsystem\n'), ...u('echo pwned'), 0x85, 0x52);
    expect(await code(() => unpickle(evil))).toBe('PTH_UNSAFE');
    expect(await code(readPth(ptZip(evil)))).toBe('PTH_UNSAFE');
  });
  it('refuses builtins via STACK_GLOBAL', async () => {
    expect(await code(() => unpickle(pkl(...u('builtins'), ...u('eval'), 0x93)))).toBe('PTH_UNSAFE');
  });
  it('refuses unknown opcodes (INST)', async () => {
    expect(await code(() => unpickle(pkl(0x69, ...strToU8('os\nsystem\n'))))).toBe('PTH_UNSAFE');
  });
  it('classifies a training checkpoint', async () => {
    const train = pkl(0x7d, 0x28, ...u('model'), 0x7d, ...u('iteration'), 0x4b, 1, 0x75);
    expect(await code(readRvcCheckpoint(ptZip(train)))).toBe('PTH_TRAINING_CKPT');
  });
  it('tells a legacy pickle from a cut-off or non-model file (QA B10)', async () => {
    expect(await code(readPth(new Blob([pkl(0x7d)])))).toBe('PTH_NOT_ZIP'); // legacy torch.save starts with PROTO
    const zip = new Uint8Array(await ptZip(pkl(0x7d)).arrayBuffer());
    expect(await code(readPth(new Blob([zip.slice(0, zip.length - 30)])))).toBe('PTH_CORRUPT'); // truncated download
    expect(await code(readPth(new Blob([strToU8('just some text, renamed to .pth')])))).toBe('PTH_CORRUPT');
    expect(await code(readPth(new Blob([new Uint8Array(1000)])))).toBe('PTH_CORRUPT');
  });
});

describe('template signature (§17.4)', () => {
  it('matches the contract example', () => {
    const cfg = [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000];
    expect(templateSignature('v2', 1, cfg)).toBe('["v2",1,192,192,768,2,6,3,"1",[3,7,11],[[1,3,5],[1,3,5],[1,3,5]],[10,10,2,2],512,[16,16,4,4],256,40000]');
  });
});

describe.each(['rvc-kyubey.pth', 'rvc-nemu.pth'])('real model %s', (f) => {
  (has(f) ? it : it.skip)('parses as v2 / f0 / 40k with the standard signature', async () => {
    const c = await readRvcCheckpoint(await openAsBlob(join(W, f)));
    expect([c.version, c.f0, c.sr]).toEqual(['v2', true, 40000]);
    expect(c.signature).toBe('["v2",1,192,192,768,2,6,3,"1",[3,7,11],[[1,3,5],[1,3,5],[1,3,5]],[10,10,2,2],512,[16,16,4,4],256,40000]');
    expect(c.nSpk).toBeGreaterThan(0);
    const t = c.weights.get('enc_p.emb_phone.weight')!;
    expect((await c.pth.read(t)).length).toBe(192 * 768 * 2);
  });
});

(has('hubert_base.pt') ? it : it.skip)('refuses fairseq hubert_base.pt (non-allowlisted globals)', async () => {
  expect(await code(readPth(await openAsBlob(join(W, 'hubert_base.pt'))))).toBe('PTH_UNSAFE');
});

describe('zip picker (MD-5)', () => {
  it('prefers added_ index and ignores G_/D_ checkpoints', () => {
    expect(pickArchiveEntries([
      { name: 'm/G_2333.pth', size: 500 }, { name: 'm/model.pth', size: 50 }, { name: 'm/trained_IVF.index', size: 1 },
      { name: 'm/added_IVF.index', size: 9 }, { name: '__MACOSX/m/._model.pth', size: 4 },
    ])).toEqual({ model: 'm/model.pth', index: 'm/added_IVF.index' });
  });
});

describe.each([['rvc-kyubey.index', 768], ['rvc-nemu.index', 768]])('faiss %s', (f, d) => {
  (has(f) ? it : it.skip)('reads IVF-Flat from Blob slices', async () => {
    const idx = await readFaissIndex(await openAsBlob(join(W, f)));
    expect(idx.d).toBe(d);
    expect(idx.nVectors).toBe(idx.ntotal);
    expect(idx.lists.reduce((a, l) => a + l.length / d, 0)).toBe(idx.ntotal);
    expect(idx.centroids.length).toBe(idx.nlist * d);
  });
});

it.skipIf(!has('rvc-kyubey.index'))('faiss: down-samples above the cap', async () => {
  const idx = await readFaissIndex(await openAsBlob(join(W, 'rvc-kyubey.index')), 5000);
  expect(idx.nVectors).toBeLessThan(idx.ntotal);
  expect(Math.abs(idx.nVectors - 5000)).toBeLessThan(500);
});

it('faiss: rejects junk', async () => {
  expect(await code(readFaissIndex(new Blob([strToU8('IwPQxxxxxxxx')])))).toBe('INDEX_UNSUPPORTED');
});
