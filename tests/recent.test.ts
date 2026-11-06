import { describe, expect, it } from 'vitest';
import { RecentJobStore } from '../src/models/recent-jobs';
import { MemoryStore } from '../src/models/store';
import { AppError } from '../src/shared/errors';
import { DEFAULT_CONVERT, type RecentJob, type Stereo } from '../src/shared/types';

const meta = (name = 'song'): Omit<RecentJob, 'v' | 'id' | 'createdAt' | 'files'> => ({
  appVersion: 't', mode: 'song', song: { name, durationSec: 1 }, convert: DEFAULT_CONVERT,
  model: { id: 'm', name: 'kyubey', version: 'v2', sr: 40000, f0: true }, mix: { vocalGainDb: 0, instGainDb: -3, reverb: 0.2 },
});
const st = (n: number, v: number): Stereo => ({ sr: 44100, l: new Float32Array(n).fill(v), r: new Float32Array(n).fill(-v / 2) });
const data = () => ({ converted: { sr: 40000, data: Float32Array.from({ length: 4000 }, (_, i) => Math.sin(i / 9) * 0.5) }, vocals: st(4410, 0.3), instrumental: st(4410, 1.125) });
const tick = () => new Promise((r) => setTimeout(r, 2)); // distinct createdAt

describe('recent jobs (I-2)', () => {
  it('save/open round trip at original level, also above 1.0 (scale), and updateMix', async () => {
    const s = new MemoryStore(), r = new RecentJobStore(() => s, async () => undefined);
    const saved = await r.save(meta(), data());
    if (!('id' in saved)) throw new Error('skipped');
    const o = await r.open(saved.id);
    expect(o.job.files.inst!.scale).toBeCloseTo(0.999 / 1.125);
    expect(o.instrumental!.l[10]).toBeCloseTo(1.125, 3);
    expect(o.instrumental!.r[10]).toBeCloseTo(-0.5625, 3);
    expect(o.converted.sr).toBe(40000);
    expect(Math.max(...o.converted.data.map((x, i) => Math.abs(x - data().converted.data[i])))).toBeLessThan(1 / 32767);
    await r.updateMix(saved.id, { vocalGainDb: -20, instGainDb: 0, reverb: 0.5, muteInst: true });
    expect((await r.open(saved.id)).job.mix.muteInst).toBe(true);
    expect((await r.list())[0]).toMatchObject({ songName: 'song', modelName: 'kyubey', bytes: 44 + 8000 + 2 * (44 + 17640) });
  });

  it('keeps 3, newest first; drops unfinished folders; remove/clear', async () => {
    const s = new MemoryStore(), r = new RecentJobStore(() => s, async () => undefined);
    for (const n of ['a', 'b', 'c', 'd']) { await r.save(meta(n), data()); await tick(); }
    expect((await r.list()).map((j) => j.songName)).toEqual(['d', 'c', 'b']);
    await s.write('recent/zzzzzzz-0000/converted.wav', [new Uint8Array(10)]); // tab closed mid-save
    expect(await r.list()).toHaveLength(3);
    expect(await s.has('recent/zzzzzzz-0000/converted.wav')).toBe(false);
    await r.remove((await r.list())[0].id);
    expect(await r.list()).toHaveLength(2);
    await r.clear();
    expect(await r.list()).toEqual([]);
    await expect(r.open('../x')).rejects.toMatchObject({ code: 'NOT_FOUND' });
  });

  it('quota: estimate too small deletes the oldest once, else skips; quota error mid-write leaves nothing', async () => {
    const s = new MemoryStore();
    let free = Infinity;
    const r = new RecentJobStore(() => s, async () => ({ quota: free, usage: 0 }));
    await r.save(meta('old'), data());
    free = 100;
    expect(await r.save(meta('new'), data())).toEqual({ skipped: 'quota' });
    expect(await r.list()).toEqual([]); // the oldest was given up to make room

    free = Infinity;
    const full = Object.assign(Object.create(s) as MemoryStore, {
      async write(p: string, c: Iterable<Uint8Array>) {
        if (p.endsWith('inst.wav')) throw new AppError('STORAGE_QUOTA', 'full', undefined, true);
        return s.write(p, c);
      },
    });
    const r2 = new RecentJobStore(() => full, async () => undefined);
    expect(await r2.save(meta(), data())).toEqual({ skipped: 'quota' });
    expect(await s.list('recent/')).toEqual([]);
  });
});
