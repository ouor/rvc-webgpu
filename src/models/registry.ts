// Public model registry (§17.5) and the repo that downloads, caches and opens its files.
import { AppError } from '../shared/errors';
import type { DownloadItem, PublicId } from '../shared/types';
import { readPth, type Pth } from '../formats/pth';
import { blobChunks, download, readVerified, sha256Hex, type BlobStore } from './store';
import { checkManifest, type Manifest } from './loader';

export interface PublicModel { graph: string; weights: string; manifest: string }
export interface Registry {
  format: 1;
  revision: string;
  files: Record<string, { bytes: number; sha256: string }>;
  models: { contentvec: PublicModel; rmvpe: PublicModel; 'uvr-hp-vocal': PublicModel & { params: string } };
  rvcTemplates: Record<string, { id: string; graph: string; manifest: string }>;
}
export type { PublicId };

const LABELS: Record<string, string> = { contentvec: 'ContentVec (HuBERT)', rmvpe: 'RMVPE', 'uvr-hp-vocal': 'UVR VR (vocal)' };

// Bundled registry written by tools/publish.py; absent during early development, then fetched from the base URL.
const bundled = Object.values(import.meta.glob<Registry>('./registry.gen.json', { eager: true, import: 'default' }))[0];

export async function loadRegistry(baseUrl: string, f: typeof fetch = fetch): Promise<Registry> {
  if (bundled) return bundled;
  try {
    const r = await f(new URL('registry.json', baseUrl).href);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as Registry;
  } catch (e) {
    throw new AppError('NET_DOWNLOAD', 'cannot load model registry', String(e), true);
  }
}

export interface Opened { graph: Uint8Array; manifest: Manifest; pth?: Pth; params?: unknown }

export class ModelRepo {
  constructor(public reg: Registry, private store: BlobStore, private baseUrl: string, private f: typeof fetch = fetch) {}

  private local = (path: string) => `public/${this.reg.revision}/${path}`;

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
      try { await readVerified(this.store, this.local(p), this.meta(p).sha256); } catch { status = 'corrupt'; }
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
    await download(this.store, new URL(path, this.baseUrl).href, this.local(path), this.meta(path), { ...o, fetch: this.f });
  }

  /** Verified read of one cached registry file. Never downloads (AC-18): callers plan()/download() first. */
  async blob(path: string): Promise<Blob> {
    if (!(await this.isCached(path))) throw new AppError('MODELS_MISSING', `${path} is not downloaded`, path, true);
    return readVerified(this.store, this.local(path), this.meta(path).sha256);
  }

  async open(id: string): Promise<Opened> {
    const m = (this.reg.models as Record<string, PublicModel & { params?: string }>)[id];
    const t = m ? undefined : Object.values(this.reg.rvcTemplates).find((x) => x.id === id);
    const graphPath = m?.graph ?? t?.graph;
    const manPath = m?.manifest ?? t?.manifest;
    if (!graphPath || !manPath) throw new AppError('INTERNAL', `unknown model ${id}`);
    const graph = new Uint8Array(await (await this.blob(graphPath)).arrayBuffer());
    const manifest = JSON.parse(await (await this.blob(manPath)).text()) as Manifest;
    if (manifest.format !== 1) throw new AppError('CACHE_INTEGRITY', `manifest format ${manifest.format} for ${id}`);
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
