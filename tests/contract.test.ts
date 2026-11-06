// tools <-> app contract (R3 A7): weights/dist as tools/publish.py writes it, read with the app's own code.
// The weights/dist part skips (not passes) when the folder is absent.
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, openAsBlob, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { checkMeta, checkRegistry, PUBLIC_IDS, type Manifest, type Registry } from '../src/models/contract';
import { checkManifest, tensorMap } from '../src/models/loader';
import { readPth } from '../src/formats/pth';
import { checkVrParams } from '../src/engine/separate/vr';
import bundled from '../src/models/registry.gen.json';

const DIST = join(import.meta.dirname, '..', 'weights', 'dist');
const HAS_DIST = existsSync(join(DIST, 'registry.json'));
const json = (p: string) => JSON.parse(readFileSync(join(DIST, p), 'utf8'));
const sha256 = (p: string) => new Promise<string>((ok, no) => {
  const h = createHash('sha256');
  createReadStream(join(DIST, p)).on('data', (d) => h.update(d)).on('end', () => ok(h.digest('hex'))).on('error', no);
});

describe('contract checks', () => {
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
      expect(man.id).toBe(id);
      checkManifest(man);
      checkMeta(id, man.meta);
      const pth = await readPth(await openAsBlob(join(DIST, m.weights)));
      const tensors = tensorMap(pth.root);
      for (const t of man.tensors) expect(tensors.get(t.key)?.shape, `${id} ${t.key}`).toEqual(t.shape);
    }
    checkVrParams(json(reg.models['uvr-hp-vocal'].params));
  });

  it('templates: manifest rules, meta keys, registry key = meta.signature', () => {
    for (const [sig, t] of Object.entries(reg.rvcTemplates)) {
      const man = json(t.manifest) as Manifest;
      expect(man.id).toBe(t.id);
      checkManifest(man);
      const meta = checkMeta('template', man.meta);
      expect(meta.signature).toBe(sig);
      expect(man.inputs.some((x) => x.name === 'phase')).toBe(meta.f0 && meta.phase !== undefined);
    }
  });
});
