// F3 "my models": the user's model (and index) kept in OPFS (ARCHITECTURE §8.2).
// Files are content-addressed: the id is the first 16 hex chars of the model file's SHA-256, so a stored file is
// never rewritten. That matters because the Blob being saved may be the stored OPFS file itself (a model that was
// opened from here), and OPFS truncates a file before writing it.
import { AppError } from '../shared/errors';
import type { IndexInfo, RvcModelInfo, SavedModel } from '../shared/types';
import { isMyModelId, MY_MODELS_ROOT, myModelDir, myModelFile, myModelIdOf } from './layout';
import { blobChunks, sha256Hex, type BlobStore } from './store';

/** info.json, version 1. `info` is a display cache refreshed on every open; the model file is the truth. */
interface SavedFileV1 { v: 1; savedAt: number; lastUsed: number; info: RvcModelInfo }
/** Before versioning (2026-11-07): the flat SavedModel / RvcModelInfo object. */
type SavedFileV0 = RvcModelInfo & { v?: undefined; savedAt?: number; lastUsed?: number };

function migrate(raw: SavedFileV1 | SavedFileV0): SavedFileV1 {
  if (raw.v === 1) return raw;
  if (raw.v === undefined) {
    const { savedAt, lastUsed, ...info } = raw;
    return { v: 1, savedAt: savedAt ?? 0, lastUsed: lastUsed ?? savedAt ?? 0, info };
  }
  throw new AppError('CACHE_INTEGRITY', `my-model info.json version ${(raw as { v: unknown }).v} is newer than this app`);
}

const toSaved = (f: SavedFileV1): SavedModel => ({ ...f.info, savedAt: f.savedAt, lastUsed: f.lastUsed });

export const modelIdOf = async (b: Blob) => (await sha256Hex(b)).slice(0, 16);

export class MyModelStore {
  constructor(private store: () => BlobStore) {}

  private check(id: string) { if (!isMyModelId(id)) throw new AppError('NOT_FOUND', `unknown my-model ${id}`); }

  private async readFile(id: string): Promise<SavedFileV1> {
    this.check(id);
    const p = myModelFile(id, 'info');
    if (!(await this.store().has(p))) throw new AppError('NOT_FOUND', `unknown my-model ${id}`);
    try { return migrate(JSON.parse(await (await this.store().read(p)).text())); }
    catch (e) { throw e instanceof AppError ? e : new AppError('CACHE_INTEGRITY', `broken info.json for my-model ${id}`, String(e), true); }
  }

  private writeFile(f: SavedFileV1) {
    return this.store().write(myModelFile(f.info.id, 'info'), [new TextEncoder().encode(JSON.stringify(f))]);
  }

  /** Writes `src` to `path` unless the same content is already there (see the file comment). */
  private async put(path: string, src: Blob, same: boolean) {
    const s = this.store();
    if (same && (await s.has(path)) && (await s.size(path)) === src.size) return;
    await s.write(path, blobChunks(src));
  }

  async save(model: { info: RvcModelInfo; file: Blob }, index?: { info: IndexInfo; file: Blob }): Promise<void> {
    const id = model.info.id;
    this.check(id);
    const prev = await this.readFile(id).catch(() => undefined);
    await this.put(myModelFile(id, model.info.source), model.file, true); // same id = same bytes
    if (index) await this.put(myModelFile(id, 'index'), index.file, prev?.info.index?.id === index.info.id);
    else await this.store().remove(myModelFile(id, 'index'));
    const now = Date.now();
    await this.writeFile({ v: 1, savedAt: prev?.savedAt ?? now, lastUsed: now, info: { ...model.info, indexId: index?.info.id, index: index?.info } });
  }

  /** Most recently used first; broken entries are skipped. */
  async list(): Promise<SavedModel[]> {
    const out: SavedModel[] = [];
    for (const f of await this.store().list(MY_MODELS_ROOT + '/')) {
      if (!f.path.endsWith('/info.json')) continue;
      try { out.push(toSaved(await this.readFile(myModelIdOf(f.path)!))); } catch { /* skip */ }
    }
    return out.sort((a, b) => b.lastUsed - a.lastUsed);
  }

  async get(id: string): Promise<SavedModel> { return toSaved(await this.readFile(id)); }

  /** The stored files of `id` (OPFS Files: they become unreadable if the stored file changes). */
  async files(id: string): Promise<{ saved: SavedModel; model: Blob; index?: Blob }> {
    const f = await this.readFile(id);
    const s = this.store();
    const mp = myModelFile(id, f.info.source);
    if (!(await s.has(mp))) throw new AppError('CACHE_INTEGRITY', `my-model ${id} has no model file`, mp, true);
    const ip = myModelFile(id, 'index');
    return { saved: toSaved(f), model: await s.read(mp), index: (await s.has(ip)) ? await s.read(ip) : undefined };
  }

  /** Updates the display cache and last use after an open. */
  async touch(id: string, info: RvcModelInfo): Promise<void> {
    const f = await this.readFile(id);
    await this.writeFile({ ...f, lastUsed: Date.now(), info: { ...info, name: f.info.name } });
  }

  async rename(id: string, name: string): Promise<string> {
    const f = await this.readFile(id);
    const n = name.trim() || f.info.name;
    await this.writeFile({ ...f, info: { ...f.info, name: n } });
    return n;
  }

  async remove(id: string): Promise<void> { this.check(id); await this.store().remove(myModelDir(id)); }

  /** 'verified' when the model file hashes to its id; the index is checked against the id recorded at save. */
  async verify(id: string): Promise<'verified' | 'incomplete' | 'corrupt'> {
    let f: { saved: SavedModel; model: Blob; index?: Blob };
    try { f = await this.files(id); } catch { return 'incomplete'; }
    if ((await modelIdOf(f.model)) !== id) return 'corrupt';
    if (f.saved.indexId && (!f.index || (await modelIdOf(f.index)) !== f.saved.indexId)) return f.index ? 'corrupt' : 'incomplete';
    return 'verified';
  }

  /** Bytes per saved model, and whether its files are all present (cheap; verify() hashes). */
  async sizes(): Promise<{ id: string; name: string; bytes: number; complete: boolean }[]> {
    const bytes = new Map<string, number>();
    for (const f of await this.store().list(MY_MODELS_ROOT + '/')) {
      const id = myModelIdOf(f.path)!;
      bytes.set(id, (bytes.get(id) ?? 0) + f.bytes);
    }
    const out = [];
    for (const [id, b] of bytes) {
      const saved = await this.get(id).catch(() => undefined);
      const s = this.store();
      const complete = !!saved && (await s.has(myModelFile(id, saved.source))) && (!saved.indexId || (await s.has(myModelFile(id, 'index'))));
      out.push({ id, name: saved?.name ?? id, bytes: b, complete });
    }
    return out;
  }
}
