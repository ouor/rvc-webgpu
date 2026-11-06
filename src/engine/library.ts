// The user's model and index held in the worker: parsing, one of each at a time (R1 M4), index attachment.
import { AppError } from '../shared/errors';
import type { IndexInfo, RvcModelInfo } from '../shared/types';
import { readFaissIndex, type IvfIndex } from '../formats/faiss';
import { readModelArchive } from '../formats/archive';
import { readOnnxInfo, type OnnxInfo } from '../formats/onnx-meta';
import { readRvcCheckpoint, type RvcCheckpoint } from '../formats/rvc-checkpoint';
import { modelIdOf } from '../models/my-models';
import { featDim } from './rvc/features';
import { describeOnnx } from './rvc/net';

// `stored` = my-model id when `file` is that saved model's OPFS file (see detachStored).
export type LoadedModel =
  | { kind: 'pth'; info: RvcModelInfo; ckpt: RvcCheckpoint; file: Blob; indexFile?: Blob; stored?: string }
  | { kind: 'onnx'; info: RvcModelInfo; bytes: Uint8Array; onnx: OnnxInfo; file: Blob; indexFile?: Blob; stored?: string };
export interface LoadedIndex { info: IndexInfo; index: IvfIndex; file: Blob }

const baseName = (n: string) => n.replace(/\.[^.]+$/, '');

export class Library {
  private models = new Map<string, LoadedModel>();
  private indexes = new Map<string, LoadedIndex>();

  /** `busy`: while a job runs, earlier entries are kept (it still uses them) until the next load. */
  constructor(private templateFor: (signature: string) => { id: string } | undefined, private busy: () => boolean) {}

  model(id: string | undefined): LoadedModel {
    const m = id ? this.models.get(id) : undefined;
    if (!m) throw new AppError('NOT_FOUND', `unknown model ${id}`);
    return m;
  }

  index(id: string | undefined): LoadedIndex {
    const x = id ? this.indexes.get(id) : undefined;
    if (!x) throw new AppError('NOT_FOUND', `unknown index ${id}`);
    return x;
  }

  loaded(id: string): LoadedModel | undefined { return this.models.get(id); }
  loadedIndex(id: string): LoadedIndex | undefined { return this.indexes.get(id); }

  async loadIndex(file: Blob): Promise<IndexInfo> {
    const index = await readFaissIndex(file); // rejects dimensions other than 256/768
    const id = await modelIdOf(file);
    const info: IndexInfo = {
      id, nVectors: index.nVectors, dim: index.d as 256 | 768, nlist: index.nlist, nprobe: index.nprobe,
      downsampled: index.nVectors < index.ntotal ? index.nVectors : undefined,
    };
    if (!this.busy()) this.indexes.clear();
    this.indexes.set(id, { info, index, file });
    return info;
  }

  /**
   * .pth, .onnx or a .zip holding one (and maybe an index). `index` attaches a known index (a saved model's);
   * an index whose dimension does not fit the model is dropped, as is a broken one inside a zip.
   */
  async loadModel(file: Blob, name: string, opts: { index?: Blob; stored?: string } = {}): Promise<RvcModelInfo> {
    if (/\.zip$/i.test(name)) {
      const a = await readModelArchive(file);
      const info = await this.loadModel(a.model, a.model.name);
      if (a.index) await this.attach(info, a.index).catch(() => { /* the UI shows "no index" */ });
      return info;
    }
    const m = await this.parse(file, name);
    m.stored = opts.stored;
    if (!this.busy()) this.models.clear();
    this.models.set(m.info.id, m);
    if (opts.index) await this.attach(m.info, opts.index);
    return m.info;
  }

  private async parse(file: Blob, name: string): Promise<LoadedModel> {
    const id = await modelIdOf(file);
    if (/\.onnx$/i.test(name)) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let onnx: OnnxInfo;
      try { onnx = readOnnxInfo(bytes); } catch (e) { throw new AppError('ONNX_UNRECOGNIZED', 'not an ONNX model', String(e)); }
      const d = describeOnnx(onnx);
      const info: RvcModelInfo = { id, name: baseName(name), source: 'onnx', version: d.version, f0: d.f0, sr: d.sr ?? 40000, srUnknown: !d.sr, nSpk: 1, bytes: file.size };
      return { kind: 'onnx', info, bytes, onnx, file };
    }
    const ckpt = await readRvcCheckpoint(file);
    const t = this.templateFor(ckpt.signature);
    if (!t) throw new AppError('PTH_UNSUPPORTED_ARCH', 'no template for this model configuration', ckpt.signature);
    const root = ckpt.pth.root as Map<string, unknown>;
    const info: RvcModelInfo = {
      id, name: typeof root.get('model_name') === 'string' ? (root.get('model_name') as string) : baseName(name),
      source: 'pth', version: ckpt.version, f0: ckpt.f0, sr: ckpt.sr, nSpk: ckpt.nSpk, templateId: t.id, bytes: file.size, meta: ckpt.meta,
    };
    return { kind: 'pth', info, ckpt, file };
  }

  private async attach(info: RvcModelInfo, file: Blob) {
    const ii = await this.loadIndex(file);
    if (ii.dim !== featDim(info.version)) { this.indexes.delete(ii.id); return; }
    info.index = ii;
    info.indexId = ii.id;
    this.model(info.id).indexFile = file;
  }

  /** Before my-model `id` is deleted: loaded blobs that are its OPFS files get memory copies, so they stay usable. */
  async detachStored(id: string): Promise<void> {
    for (const m of this.models.values()) {
      if (m.stored !== id) continue;
      m.file = new Blob([await m.file.arrayBuffer()]);
      if (m.indexFile) {
        const copy = new Blob([await m.indexFile.arrayBuffer()]);
        for (const ix of this.indexes.values()) if (ix.file === m.indexFile) ix.file = copy;
        m.indexFile = copy;
      }
      m.stored = undefined;
    }
  }
}
