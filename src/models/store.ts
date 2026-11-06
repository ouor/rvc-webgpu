// Blob storage (OPFS in the browser, memory in tests), resumable download and SHA-256 checks (§8.2).
import { AppError } from '../shared/errors';

export interface BlobStore {
  has(path: string): Promise<boolean>;
  read(path: string): Promise<Blob>; // OPFS File is a Blob, so it can be sliced
  write(path: string, chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>, append?: boolean): Promise<void>;
  size(path: string): Promise<number>;
  remove(path: string): Promise<void>; // file or folder
  list(prefix: string): Promise<{ path: string; bytes: number }[]>;
}

const notFound = (p: string) => new AppError('NOT_FOUND', `no stored file ${p}`);
const isQuota = (e: unknown) => e instanceof DOMException && e.name === 'QuotaExceededError';

export class MemoryStore implements BlobStore {
  private files = new Map<string, Uint8Array[]>();
  async has(p: string) { return this.files.has(p); }
  async read(p: string) {
    const f = this.files.get(p);
    if (!f) throw notFound(p);
    return new Blob(f as Uint8Array<ArrayBuffer>[]);
  }
  async write(p: string, chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>, append = false) {
    const parts = append ? (this.files.get(p) ?? []) : [];
    this.files.set(p, parts);
    for await (const c of chunks) parts.push(c.slice());
  }
  async size(p: string) { return (this.files.get(p) ?? []).reduce((a, c) => a + c.length, 0); }
  async remove(p: string) { for (const k of [...this.files.keys()]) if (k === p || k.startsWith(p.replace(/\/?$/, '/'))) this.files.delete(k); }
  async list(prefix: string) {
    return Promise.all([...this.files.keys()].filter((k) => k.startsWith(prefix)).map(async (path) => ({ path, bytes: await this.size(path) })));
  }
}

type Dir = FileSystemDirectoryHandle;
interface SyncHandle { write(b: Uint8Array, o: { at: number }): number; truncate(n: number): void; getSize(): number; flush(): void; close(): void }

/** OPFS, worker only (createSyncAccessHandle). */
export class OpfsStore implements BlobStore {
  private rootP: Promise<Dir> | undefined;
  private root() { return (this.rootP ??= navigator.storage.getDirectory()); }

  private async dir(parts: string[], create: boolean): Promise<Dir> {
    let d = await this.root();
    for (const s of parts) d = await d.getDirectoryHandle(s, { create });
    return d;
  }
  private async file(path: string, create = false) {
    const parts = path.split('/').filter(Boolean);
    const name = parts.pop()!;
    try {
      return await (await this.dir(parts, create)).getFileHandle(name, { create });
    } catch (e) {
      if (e instanceof DOMException && (e.name === 'NotFoundError' || e.name === 'TypeMismatchError')) throw notFound(path);
      throw e;
    }
  }
  async has(p: string) { return this.file(p).then(() => true, () => false); }
  async read(p: string) { return (await this.file(p)).getFile(); }
  async size(p: string) { return (await this.read(p)).size; }
  async write(p: string, chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>, append = false) {
    const fh = await this.file(p, true);
    const h = await (fh as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle();
    try {
      if (!append) h.truncate(0);
      let at = h.getSize();
      for await (const c of chunks) at += h.write(c, { at });
      h.flush();
    } catch (e) {
      if (isQuota(e)) throw new AppError('STORAGE_QUOTA', 'storage quota exceeded', String(e), true);
      throw e;
    } finally {
      h.close();
    }
  }
  async remove(p: string) {
    const parts = p.split('/').filter(Boolean);
    const name = parts.pop()!;
    try { await (await this.dir(parts, false)).removeEntry(name, { recursive: true }); } catch { /* already absent */ }
  }
  async list(prefix: string) {
    const out: { path: string; bytes: number }[] = [];
    const walk = async (d: Dir, base: string) => {
      for await (const [name, h] of (d as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
        const path = base + name;
        if (h.kind === 'directory') await walk(h as Dir, path + '/');
        else if (path.startsWith(prefix)) out.push({ path, bytes: (await (h as FileSystemFileHandle).getFile()).size });
      }
    };
    const parts = prefix.split('/').filter(Boolean);
    const top = prefix.endsWith('/') ? parts : parts.slice(0, -1);
    try { await walk(await this.dir(top, false), top.length ? top.join('/') + '/' : ''); } catch { /* no folder yet */ }
    return out;
  }
}

export async function sha256Hex(blob: Blob | ArrayBuffer): Promise<string> {
  const buf = blob instanceof ArrayBuffer ? blob : await blob.arrayBuffer();
  const h = new Uint8Array(await crypto.subtle.digest('SHA-256', buf));
  return Array.from(h, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Reads a stored file and checks its hash; a mismatch deletes it and throws CACHE_INTEGRITY (AC-20). */
export async function readVerified(store: BlobStore, path: string, sha256: string): Promise<Blob> {
  const blob = await store.read(path);
  if ((await sha256Hex(blob)) !== sha256) {
    await store.remove(path);
    throw new AppError('CACHE_INTEGRITY', `hash mismatch for ${path}`, path, true);
  }
  return blob;
}

export interface DownloadOpts {
  fetch?: typeof fetch;
  onBytes?: (received: number) => void;
  throwIfCancelled?: () => void;
  retries?: number;
}

/** Streams url into store[path], resuming with Range after a drop, then verifies size and sha256. */
export async function download(store: BlobStore, url: string, path: string, expect: { bytes: number; sha256: string }, o: DownloadOpts = {}): Promise<void> {
  const f = o.fetch ?? fetch;
  let have = (await store.has(path)) ? await store.size(path) : 0;
  if (have > expect.bytes) { await store.remove(path); have = 0; }
  let lastErr: unknown;
  for (let attempt = 0; have < expect.bytes && attempt <= (o.retries ?? 3); attempt++) {
    try {
      o.throwIfCancelled?.();
      const res = await f(url, { headers: have ? { Range: `bytes=${have}-` } : {}, cache: 'no-store' });
      if (!res.ok || !res.body) throw new AppError('NET_DOWNLOAD', `HTTP ${res.status} for ${url}`, url, true);
      if (have && res.status !== 206) have = 0; // server ignored Range: start over
      const reader = res.body.getReader();
      let got = have;
      const chunks = {
        async *[Symbol.asyncIterator]() {
          for (;;) {
            o.throwIfCancelled?.();
            const { done, value } = await reader.read();
            if (done) return;
            got += value.length;
            o.onBytes?.(got);
            yield value;
          }
        },
      };
      try {
        await store.write(path, chunks, have > 0);
      } finally {
        have = await store.size(path);
        reader.cancel().catch(() => {});
      }
    } catch (e) {
      if (e instanceof AppError && (e.code === 'CANCELLED' || e.code === 'STORAGE_QUOTA')) throw e;
      if (isQuota(e)) throw new AppError('STORAGE_QUOTA', 'storage quota exceeded', String(e), true);
      lastErr = e;
      have = (await store.has(path)) ? await store.size(path) : 0;
    }
  }
  if (have !== expect.bytes) {
    if (have > expect.bytes) await store.remove(path);
    throw lastErr instanceof AppError ? lastErr : new AppError('NET_DOWNLOAD', `download failed: ${url}`, String(lastErr ?? `size ${have}/${expect.bytes}`), true);
  }
  await readVerified(store, path, expect.sha256);
}
