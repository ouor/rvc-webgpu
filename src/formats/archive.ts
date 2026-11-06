// User model .zip (weights.gg / Hugging Face style): pick the model and the index (PRD MD-5).
import { unzipSync } from 'fflate';
import { AppError } from '../shared/errors';

export interface ModelArchive { model: File; index?: File }

const base = (p: string) => p.split('/').pop() ?? p;

// fflate allocates `originalSize` up front, so the header sizes are checked before extracting (R1 M7).
export const MAX_ENTRY_BYTES = 1 << 30; // far above real RVC models (~50-200 MB) and indexes (< 700 MB)
const MAX_RATIO = 50; // .pth/.onnx/.index barely compress; a higher ratio is a zip bomb

/** Chooses one .pth/.onnx (largest wins) and one .index (added_* preferred over trained_*). */
export function pickArchiveEntries(names: { name: string; size: number }[]) {
  const files = names.filter((e) => !e.name.startsWith('__MACOSX/') && !base(e.name).startsWith('.'));
  const models = files.filter((e) => /\.(pth|onnx)$/i.test(e.name) && !/^(G|D)_\d+\.pth$/i.test(base(e.name)));
  const indexes = files.filter((e) => /\.index$/i.test(e.name));
  const model = models.sort((a, b) => b.size - a.size)[0];
  const index = indexes.find((e) => /^added_/i.test(base(e.name))) ?? indexes.find((e) => !/^trained_/i.test(base(e.name))) ?? indexes[0];
  return { model: model?.name, index: index?.name };
}

export async function readModelArchive(file: Blob): Promise<ModelArchive> {
  const buf = new Uint8Array(await file.arrayBuffer());
  const listing: { name: string; size: number; packed: number }[] = [];
  try {
    unzipSync(buf, { filter: (f) => { listing.push({ name: f.name, size: f.originalSize, packed: f.size }); return false; } });
  } catch (e) {
    throw new AppError('ZIP_NO_MODEL', 'cannot read zip', String(e));
  }
  const pick = pickArchiveEntries(listing);
  if (!pick.model) throw new AppError('ZIP_NO_MODEL', 'zip contains no .pth or .onnx model');
  const want = new Set([pick.model, pick.index].filter(Boolean) as string[]);
  for (const e of listing) {
    if (!want.has(e.name)) continue;
    if (e.size > MAX_ENTRY_BYTES || e.size > Math.max(e.packed, 1) * MAX_RATIO)
      throw new AppError('ZIP_NO_MODEL', `zip entry ${e.name} is too large (${e.size} bytes)`, e.name);
  }
  let out: ReturnType<typeof unzipSync>;
  try { out = unzipSync(buf, { filter: (f) => want.has(f.name) }); }
  catch (e) { throw new AppError('ZIP_NO_MODEL', 'cannot extract zip', String(e)); }
  const toFile = (n: string) => new File([out[n] as Uint8Array<ArrayBuffer>], base(n));
  return { model: toFile(pick.model), index: pick.index ? toFile(pick.index) : undefined };
}
