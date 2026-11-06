// Public model registry (§17.5) and the repo that downloads, caches and opens its files.
import { AppError } from '../shared/errors';
import type { DownloadItem } from '../shared/types';
import { readPth, type Pth } from '../formats/pth';
import { blobChunks, download, readVerified, sha256Hex, type BlobStore } from './store';
import { checkManifest, type Manifest } from './loader';
import { checkRegistry, type PublicModel, type Registry } from './contract';

export type { PublicId, PublicModel, Registry } from './contract';

const LABELS: Record<string, string> = { contentvec: 'ContentVec (HuBERT)', rmvpe: 'RMVPE', 'uvr-hp-vocal': 'UVR VR (vocal)' };

// Bundled registry written by tools/publish.py; absent during early development, then fetched from the base URL.
const bundled = Object.values(import.meta.glob<Registry>('./registry.gen.json', { eager: true, import: 'default' }))[0];

export async function loadRegistry(baseUrl: string, f: typeof fetch = fetch): Promise<Registry> {
  if (bundled) return checkRegistry(bundled);
  let json: unknown;
  try {
    const r = await f(new URL('registry.json', baseUrl).href);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    json = await r.json();
  } catch (e) {
    throw new AppError('NET_DOWNLOAD', 'cannot load model registry', String(e), true);
  }
  return checkRegistry(json);
}

export interface Opened { graph: Uint8Array; manifest: Manifest; pth?: Pth; params?: unknown }

export class ModelRepo {
  constructor(public reg: Registry, private store: BlobStore, private baseUrl: string, private f: typeof fetch = fetch) {}

  private local = (path: string) => `public/${this.reg.revision}/${path}`;
  /** Marker holding the sha256 a file was verified against (download, import, verify). Loads trust it (B6). */
  static readonly MARK = '.verified';
  private mark = (path: string) => this.store.write(this.local(path) + ModelRepo.MARK, [new TextEncoder().encode(this.meta(path).sha256)]);
  private async marked(path: string) {
    const m = this.local(path) + ModelRepo.MARK;
    return (await this.store.has(m)) && (await (await this.store.read(m)).text()) === this.meta(path).sha256;
  }

  /** Registry paths needed for a public model or an RVC template id. */
  files(id: string): string[] {
    const m = (this.reg.models as Record<string, PublicModel & { params?: string }>)[id];
    if (m) return [m.graph, m.manifest, m.weights, ...(m.params ? [m.params] : [])];
    const t = Object.values(this.reg.rvcTemplates).find((x) => x.id === id);
    if (t) return [t.graph, t.manifest];
    throw new AppError('INTERNAL', `unknown model ${id}`);
  }

  label(id: string) { return LABELS[id] ?? `RVC template ${id}`; }

  /** Public model and template ids, in registry order. */
  ids(): string[] { return [...Object.keys(this.reg.models), ...Object.values(this.reg.rvcTemplates).map((t) => t.id)]; }

  /** Re-hashes the cached files of `id`; mismatching files are deleted (readVerified). */
  async verify(id: string): Promise<'verified' | 'incomplete' | 'corrupt'> {
    let status: 'verified' | 'incomplete' | 'corrupt' = 'verified';
    for (const p of this.files(id)) {
      if (!(await this.isCached(p))) { if (status === 'verified') status = 'incomplete'; continue; }
      try { await readVerified(this.store, this.local(p), this.meta(p).sha256); await this.mark(p); }
      catch { status = 'corrupt'; await this.store.remove(this.local(p) + ModelRepo.MARK); }
    }
    return status;
  }

  /**
   * PRD R10: stores a local copy of a registry file. `name` is matched against registry paths by its longest
   * path suffix (a folder upload keeps "rvc/v2-f0-40k.onnx", a plain pick only has the basename).
   */
  async importFile(file: Blob, name: string): Promise<{ path?: string; status: 'imported' | 'unknown' | 'mismatch' }> {
    const norm = name.replace(/\\/g, '/');
    const hits = Object.keys(this.reg.files).filter((p) => norm === p || norm.endsWith('/' + p) || p.endsWith('/' + norm));
    if (hits.length !== 1) return { status: 'unknown' }; // none, or an ambiguous basename
    const path = hits[0], m = this.meta(path);
    if (file.size !== m.bytes || (await sha256Hex(file)) !== m.sha256) return { path, status: 'mismatch' };
    await this.store.write(this.local(path), blobChunks(file));
    await this.mark(path);
    return { path, status: 'imported' };
  }

  templateFor(signature: string) { return this.reg.rvcTemplates[signature]; }

  private meta(path: string) {
    const m = this.reg.files[path];
    if (!m) throw new AppError('INTERNAL', `registry has no entry for ${path}`);
    return m;
  }

  async isCached(path: string) {
    const s = this.store;
    return (await s.has(this.local(path))) && (await s.size(this.local(path))) === this.meta(path).bytes;
  }

  async plan(ids: string[]): Promise<DownloadItem[]> {
    return Promise.all(ids.map(async (id) => {
      const files = this.files(id);
      const cached = (await Promise.all(files.map((p) => this.isCached(p)))).every(Boolean);
      const bytes = files.reduce((a, p) => a + this.meta(p).bytes, 0);
      return { id, label: this.label(id), bytes, cached };
    }));
  }

  async download(path: string, o: { onBytes?: (n: number) => void; throwIfCancelled?: () => void }) {
    await this.store.remove(this.local(path) + ModelRepo.MARK);
    await download(this.store, new URL(path, this.baseUrl).href, this.local(path), this.meta(path), { ...o, fetch: this.f });
    await this.mark(path);
  }

  /**
   * Read of one cached registry file. Never downloads (AC-18): callers plan()/download() first.
   * QA B6: a file verified once (marker) is only size-checked; cache.verify() re-hashes in full.
   */
  async blob(path: string): Promise<Blob> {
    if (!(await this.isCached(path))) throw new AppError('MODELS_MISSING', `${path} is not downloaded`, path, true);
    if (await this.marked(path)) return this.store.read(this.local(path));
    const b = await readVerified(this.store, this.local(path), this.meta(path).sha256); // caches from before the marker
    await this.mark(path);
    return b;
  }

  async open(id: string): Promise<Opened> {
    const m = (this.reg.models as Record<string, PublicModel & { params?: string }>)[id];
    const t = m ? undefined : Object.values(this.reg.rvcTemplates).find((x) => x.id === id);
    const graphPath = m?.graph ?? t?.graph;
    const manPath = m?.manifest ?? t?.manifest;
    if (!graphPath || !manPath) throw new AppError('INTERNAL', `unknown model ${id}`);
    const graph = new Uint8Array(await (await this.blob(graphPath)).arrayBuffer());
    const manifest = JSON.parse(await (await this.blob(manPath)).text()) as Manifest;
    checkManifest(manifest);
    const pth = m ? await readPth(await this.blob(m.weights)).catch((e) => { throw new AppError('CACHE_INTEGRITY', `bad weights for ${id}`, String(e), true); }) : undefined;
    const params = m?.params ? JSON.parse(await (await this.blob(m.params)).text()) : undefined;
    return { graph, manifest, pth, params };
  }

  /** Removes cached files of other revisions (PRD 8.4). */
  async pruneOldRevisions() {
    for (const f of await this.store.list('public/')) {
      const rev = f.path.split('/')[1];
      if (rev !== this.reg.revision) await this.store.remove(`public/${rev}`);
    }
  }
}
