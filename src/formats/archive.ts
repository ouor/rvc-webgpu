// User model .zip (weights.gg / Hugging Face style): pick the model and the index (PRD MD-5).
import { unzipSync } from 'fflate';
import { AppError } from '../shared/errors';

export interface ModelArchive { model: File; index?: File }

const base = (p: string) => p.split('/').pop() ?? p;

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
  const listing: { name: string; size: number }[] = [];
  try {
    unzipSync(buf, { filter: (f) => { listing.push({ name: f.name, size: f.originalSize }); return false; } });
  } catch (e) {
    throw new AppError('ZIP_NO_MODEL', 'cannot read zip', String(e));
  }
  const pick = pickArchiveEntries(listing);
  if (!pick.model) throw new AppError('ZIP_NO_MODEL', 'zip contains no .pth or .onnx model');
  const want = new Set([pick.model, pick.index].filter(Boolean) as string[]);
  const out = unzipSync(buf, { filter: (f) => want.has(f.name) });
  const toFile = (n: string) => new File([out[n] as Uint8Array<ArrayBuffer>], base(n));
  return { model: toFile(pick.model), index: pick.index ? toFile(pick.index) : undefined };
}
