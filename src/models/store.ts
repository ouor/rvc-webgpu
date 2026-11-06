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

/**
 * A Blob's bytes as chunks for BlobStore.write (ReadableStream is not async-iterable in Safari).
 * Never pass a Blob read from the same path you write: OPFS truncates first (see models/my-models.ts).
 */
export async function* blobChunks(b: Blob): AsyncIterable<Uint8Array> {
  const r = b.stream().getReader();
  try { for (;;) { const { done, value } = await r.read(); if (done) return; yield value; } }
  finally { r.releaseLock(); }
}

const notFound = (p: string) => new AppError('NOT_FOUND', `no stored file ${p}`);
const isQuota = (e: unknown) => e instanceof DOMException && e.name === 'QuotaExceededError';

/**
 * In-memory store with OPFS semantics where they matter: write() truncates before it consumes its chunks, and a Blob
 * returned by read() becomes unreadable (NotReadableError) once its file is written or removed, like an OPFS File.
 */
export class MemoryStore implements BlobStore {
  private files = new Map<string, Uint8Array[]>();
  private gen = new Map<string, number>();
  private touch(p: string) { this.gen.set(p, (this.gen.get(p) ?? 0) + 1); }
  async has(p: string) { return this.files.has(p); }
  async read(p: string): Promise<Blob> {
    const f = this.files.get(p);
    if (!f) throw notFound(p);
    const g = this.gen.get(p) ?? 0;
    return snapshot(new Blob(f.map((c) => c.slice())), () => (this.gen.get(p) ?? 0) === g, p);
  }
  async write(p: string, chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>, append = false) {
    const parts = append ? (this.files.get(p) ?? []) : [];
    this.files.set(p, parts);
    this.touch(p);
    for await (const c of chunks) parts.push(c.slice());
  }
  async size(p: string) { return (this.files.get(p) ?? []).reduce((a, c) => a + c.length, 0); }
  async remove(p: string) {
    for (const k of [...this.files.keys()]) if (k === p || k.startsWith(p.replace(/\/?$/, '/'))) { this.files.delete(k); this.touch(k); }
  }
  async list(prefix: string) {
    return Promise.all([...this.files.keys()].filter((k) => k.startsWith(prefix)).map(async (path) => ({ path, bytes: await this.size(path) })));
  }
}

/** Blob whose reads fail once `valid()` turns false (OPFS File snapshot behaviour). */
function snapshot(b: Blob, valid: () => boolean, p: string): Blob {
  const check = () => { if (!valid()) throw new DOMException(`${p} changed after it was read`, 'NotReadableError'); };
  return new Proxy(b, {
    get(t, k) {
      if (k === 'arrayBuffer' || k === 'text' || k === 'bytes') return async () => { check(); return (t as unknown as Record<string, () => Promise<unknown>>)[k](); };
      if (k === 'stream') return () => {
        const r = t.stream().getReader();
        return new ReadableStream({ async pull(c) { try { check(); } catch (e) { c.error(e); return; } const x = await r.read(); if (x.done) c.close(); else c.enqueue(x.value); } });
      };
      if (k === 'slice') return (...a: Parameters<Blob['slice']>) => snapshot(t.slice(...a), valid, p);
      const v = Reflect.get(t, k, t);
      return typeof v === 'function' ? v.bind(t) : v;
    },
  });
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
