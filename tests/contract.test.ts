// tools <-> app contract (R3 A7): weights/dist as tools/publish.py writes it, read with the app's own code.
// The weights/dist part skips (not passes) when the folder is absent.
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, openAsBlob, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { checkMeta, checkRegistry, CONTRACT as C, PUBLIC_IDS, type Manifest, type Registry } from '../src/models/contract';
import { checkManifest, tensorMap } from '../src/models/loader';
import { readPth } from '../src/formats/pth';
import { checkVrParams } from '../src/engine/separate/vr';
import bundled from '../src/models/registry.gen.json';

const DIST = join(import.meta.dirname, '..', 'weights', 'dist');
const HAS_DIST = existsSync(join(DIST, 'registry.json'));
const json = (p: string) => JSON.parse(readFileSync(join(DIST, p), 'utf8'));
/** contract.json: header, keys, and graph I/O names and dtypes in order. */
function checkAgainstJson(man: Manifest, io: { inputs: { name: string; dtype: string }[]; outputs: { name: string; dtype: string }[] }) {
  expect(Object.keys(man).sort()).toEqual([...C.manifestKeys].sort());
  expect([man.format, man.externalData]).toEqual([C.manifestFormat, C.externalData]);
  const nd = (xs: { name: string; dtype: string }[]) => xs.map((x) => ({ name: x.name, dtype: x.dtype }));
  expect(nd(man.inputs), `${man.id} inputs`).toEqual(io.inputs);
  expect(nd(man.outputs), `${man.id} outputs`).toEqual(io.outputs);
}
const sha256 = (p: string) => new Promise<string>((ok, no) => {
  const h = createHash('sha256');
  createReadStream(join(DIST, p)).on('data', (d) => h.update(d)).on('end', () => ok(h.digest('hex'))).on('error', no);
});

describe('contract checks', () => {
  it('contract.json lists exactly the public ids of shared/types', () => {
    expect(Object.keys(C.public).sort()).toEqual(Object.values(PUBLIC_IDS).sort());
  });

  it('the bundled registry satisfies the contract', () => {
    expect(checkRegistry(bundled).format).toBe(1);
  });

  it('checkRegistry rejects other formats and missing keys', () => {
    const r = structuredClone(bundled) as Registry;
    expect(() => checkRegistry({ ...r, format: 2 })).toThrow(/does not match/);
    expect(() => checkRegistry({ ...r, rvcTemplates: undefined })).toThrow();
    const noFile = structuredClone(r);
    delete noFile.files[noFile.models.rmvpe.graph];
    expect(() => checkRegistry(noFile)).toThrow();
  });

  it('checkMeta requires the keys the app reads', () => {
    expect(() => checkMeta('rmvpe', { sr: 16000, hop: 160, threshold: 0.03 })).toThrow(/centsOffset/);
    expect(() => checkMeta('template', { version: 'v2', f0: true, sr: 44100, upp: 400, signature: '' })).toThrow(/sr/);
  });
});

describe.skipIf(!HAS_DIST)('weights/dist', () => {
  const reg = HAS_DIST ? checkRegistry(json('registry.json')) : (undefined as never);

  it('every registry file exists with the listed size and sha256', async () => {
    for (const [p, f] of Object.entries(reg.files)) {
      expect((await openAsBlob(join(DIST, p))).size, p).toBe(f.bytes);
      expect(await sha256(p), p).toBe(f.sha256);
    }
  });

  it('public models: manifest rules, meta keys, weights match the manifest', async () => {
    for (const id of Object.values(PUBLIC_IDS)) {
      const m = reg.models[id];
      const man = json(m.manifest) as Manifest;
      const c = C.public[id];
      expect(man.id).toBe(id);
      expect([m.graph, m.weights, m.manifest]).toEqual(['.onnx', '.pth', '.manifest.json'].map((x) => c.files + x));
      checkAgainstJson(man, c);
      checkManifest(man);
      checkMeta(id, man.meta);
      const pth = await readPth(await openAsBlob(join(DIST, m.weights)));
      const tensors = tensorMap(pth.root);
      for (const t of man.tensors) expect(tensors.get(t.key)?.shape, `${id} ${t.key}`).toEqual(t.shape);
    }
    expect(reg.models['uvr-hp-vocal'].params).toBe(C.public['uvr-hp-vocal'].params);
    checkVrParams(json(reg.models['uvr-hp-vocal'].params));
  });

  it('templates: manifest rules, meta keys, registry key = meta.signature', () => {
    for (const [sig, t] of Object.entries(reg.rvcTemplates)) {
      const man = json(t.manifest) as Manifest;
      expect(man.id).toBe(t.id);
      const base = C.template.files.replace('{id}', t.id);
      expect([t.graph, t.manifest]).toEqual([base + '.onnx', base + '.manifest.json']);
      checkManifest(man);
      const meta = checkMeta('template', man.meta);
      expect(meta.signature).toBe(sig);
      checkAgainstJson(man, { inputs: C.template.inputs[meta.f0 ? 'f0' : 'nof0'], outputs: C.template.outputs });
    }
  });
});
