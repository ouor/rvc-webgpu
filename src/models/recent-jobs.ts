// I-2 recent jobs (ARCHITECTURE 18.2): finished results as 16-bit WAV + job.json in OPFS, at most MAX_RECENT.
// WAVs are written first and job.json last, so a folder without job.json is an unfinished save.
import { AppError } from '../shared/errors';
import type { Mono, RecentJob, RecentMix, RecentStem, RecentSummary, Stereo } from '../shared/types';
import { readWav16, wavBytes, wavChunks } from '../formats/wav';
import { isRecentId, RECENT_ROOT, recentDir, recentFile } from './layout';
import type { BlobStore } from './store';

export const MAX_RECENT = 3;
export type RecentData = { converted: Mono; vocals?: Stereo; instrumental?: Stereo };
type Estimate = () => Promise<{ quota?: number; usage?: number } | undefined>;

const peakScale = (ch: Float32Array[]) => {
  let p = 0;
  for (const c of ch) for (let i = 0; i < c.length; i++) { const a = Math.abs(c[i]); if (a > p) p = a; }
  return p > 1 ? 0.999 / p : 1; // lossless level for 16 bit, divided out on open
};

function migrate(raw: RecentJob): RecentJob {
  if (raw.v === 1) return raw;
  throw new AppError('CACHE_INTEGRITY', `recent job version ${(raw as { v: unknown }).v} is newer than this app`);
}

export class RecentJobStore {
  private saving = new Set<string>();
  constructor(private store: () => BlobStore, private estimate: Estimate = async () => navigator.storage?.estimate?.()) {}

  private check(id: string) { if (!isRecentId(id)) throw new AppError('NOT_FOUND', `unknown recent job ${id}`); }

  private async read(id: string): Promise<RecentJob> {
    this.check(id);
    const p = recentFile(id, 'job');
    if (!(await this.store().has(p))) throw new AppError('NOT_FOUND', `unknown recent job ${id}`);
    try { return migrate(JSON.parse(await (await this.store().read(p)).text())); }
    catch (e) { throw e instanceof AppError ? e : new AppError('CACHE_INTEGRITY', `broken job.json for recent job ${id}`, String(e), true); }
  }

  private write(job: RecentJob) { return this.store().write(recentFile(job.id, 'job'), [new TextEncoder().encode(JSON.stringify(job))]); }

  /** Complete jobs, newest first; unfinished folders (no job.json, not being saved now) are removed. */
  private async jobs(): Promise<RecentJob[]> {
    const ids = new Set((await this.store().list(RECENT_ROOT + '/')).map((f) => f.path.split('/')[1]));
    const out: RecentJob[] = [];
    for (const id of ids) {
      if (this.saving.has(id)) continue;
      const job = isRecentId(id) ? await this.read(id).catch(() => undefined) : undefined;
      if (job) out.push(job); else await this.store().remove(recentDir(id));
    }
    return out.sort((a, b) => b.createdAt - a.createdAt);
  }

  async save(meta: Omit<RecentJob, 'v' | 'id' | 'createdAt' | 'files'>, data: RecentData): Promise<{ id: string } | { skipped: 'quota' }> {
    const parts: [RecentStem, Float32Array[], number][] = [['converted', [data.converted.data], data.converted.sr]];
    if (data.vocals) parts.push(['vocals', [data.vocals.l, data.vocals.r], data.vocals.sr]);
    if (data.instrumental) parts.push(['inst', [data.instrumental.l, data.instrumental.r], data.instrumental.sr]);
    const need = parts.reduce((a, [, ch]) => a + wavBytes(ch.length, ch[0].length), 0);

    // space: drop the oldest job once if the estimate says it does not fit (18.2 policy 1)
    const free = async () => { const e = await this.estimate().catch(() => undefined); return e?.quota ? e.quota - (e.usage ?? 0) : Infinity; };
    if ((await free()) < need * 1.1) {
      const old = (await this.jobs()).at(-1);
      if (old) await this.remove(old.id);
      if ((await free()) < need * 1.1) return { skipped: 'quota' };
    }

    const createdAt = Date.now();
    const id = `${createdAt.toString(36)}-${Math.floor(Math.random() * 36 ** 4).toString(36).padStart(4, '0')}`;
    const job: RecentJob = { ...meta, v: 1, id, createdAt, files: {} };
    this.saving.add(id);
    try {
      for (const [kind, ch, sr] of parts) {
        const scale = peakScale(ch);
        await this.store().write(recentFile(id, kind), wavChunks(ch, sr, scale));
        job.files[kind] = { file: `${kind}.wav`, sr, channels: ch.length as 1 | 2, frames: ch[0].length, scale, bytes: wavBytes(ch.length, ch[0].length) };
      }
      await this.write(job);
    } catch (e) {
      await this.store().remove(recentDir(id)).catch(() => {});
      if (e instanceof AppError && e.code === 'STORAGE_QUOTA') return { skipped: 'quota' }; // policy 2
      throw e;
    } finally { this.saving.delete(id); }
    for (const old of (await this.jobs()).slice(MAX_RECENT)) await this.remove(old.id); // policy 3
    return { id };
  }

  async list(): Promise<RecentSummary[]> {
    return (await this.jobs()).map((j) => ({
      id: j.id, createdAt: j.createdAt, songName: j.song.name, modelName: j.model.name, mode: j.mode, f0UpKey: j.convert.f0UpKey,
      range: j.song.range, bytes: Object.values(j.files).reduce((a, f) => a + (f?.bytes ?? 0), 0),
    }));
  }

  async open(id: string): Promise<{ job: RecentJob } & RecentData> {
    const job = await this.read(id);
    const load = async (kind: RecentStem) => {
      const f = job.files[kind];
      if (!f) return undefined;
      try { return await readWav16(await this.store().read(recentFile(id, kind)), f.scale); }
      catch (e) { throw e instanceof AppError ? e : new AppError('CACHE_INTEGRITY', `broken ${kind}.wav in recent job ${id}`, String(e), true); }
    };
    const stereo = (w?: { sr: number; channels: Float32Array[] }): Stereo | undefined => w && { sr: w.sr, l: w.channels[0], r: w.channels[1] };
    const c = await load('converted');
    if (!c) throw new AppError('CACHE_INTEGRITY', `recent job ${id} has no converted vocal`, undefined, true);
    return { job, converted: { sr: c.sr, data: c.channels[0] }, vocals: stereo(await load('vocals')), instrumental: stereo(await load('inst')) };
  }

  async updateMix(id: string, mix: RecentMix) { await this.write({ ...(await this.read(id)), mix }); }
  async remove(id: string) { this.check(id); await this.store().remove(recentDir(id)); }
  async clear() { await this.store().remove(RECENT_ROOT); }
  async bytes() { return (await this.store().list(RECENT_ROOT + '/')).reduce((a, f) => a + f.bytes, 0); }
}
