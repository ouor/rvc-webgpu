// RVC inference checkpoint (.pth) -> model description + template signature (ARCHITECTURE §7.2, §17.4).
import { AppError } from '../shared/errors';
import { isTensor, readPth, type Pth, type TensorRef } from './pth';

export interface RvcCheckpoint {
  pth: Pth;
  weights: Map<string, TensorRef>;
  version: 'v1' | 'v2';
  f0: boolean;
  sr: 32000 | 40000 | 48000;
  nSpk: number;
  signature: string;
  meta: { epoch?: number; vocoder?: string; embedder?: string; author?: string };
}

const SUPPORTED_SR = [32000, 40000, 48000];

/** §17.4: the exact string both tools and src use as registry key. */
export function templateSignature(version: string, f0: number, cfg: unknown[]): string {
  const n = (v: unknown) => (typeof v === 'number' ? Math.trunc(v) : v);
  const deep = (v: unknown): unknown => (Array.isArray(v) ? v.map(deep) : n(v));
  const sig = [version, f0, cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], String(cfg[9]),
    cfg[10], cfg[11], cfg[12], cfg[13], cfg[14], cfg[16], cfg[17]].map(deep);
  return JSON.stringify(sig);
}

const str = (v: unknown) => (typeof v === 'string' ? v : undefined);

export async function readRvcCheckpoint(blob: Blob): Promise<RvcCheckpoint> {
  const pth = await readPth(blob);
  const root = pth.root;
  if (!(root instanceof Map)) throw new AppError('PTH_CORRUPT', 'checkpoint root is not a dict');
  if (!root.has('weight')) {
    if (root.has('model') || root.has('optimizer') || root.has('iteration'))
      throw new AppError('PTH_TRAINING_CKPT', 'training checkpoint (G_/D_*.pth), not an inference model');
    throw new AppError('PTH_UNSUPPORTED_ARCH', 'no "weight" dict: not an RVC inference model');
  }
  const vocoder = str(root.get('vocoder'));
  if (vocoder && vocoder !== 'HiFi-GAN') throw new AppError('PTH_UNSUPPORTED_VOCODER', `vocoder ${vocoder}`, vocoder);
  const embedder = str(root.get('embedder_model'));
  if (embedder && !['contentvec', 'hubert_base'].includes(embedder))
    throw new AppError('PTH_UNSUPPORTED_EMBEDDER', `embedder ${embedder}`, embedder);

  const w = root.get('weight');
  const cfg = root.get('config');
  if (!(w instanceof Map) || !Array.isArray(cfg) || cfg.length < 18) throw new AppError('PTH_CORRUPT', 'missing weight/config');
  const weights = new Map<string, TensorRef>();
  for (const [k, v] of w) if (isTensor(v)) weights.set(String(k), v);
  const emb = weights.get('emb_g.weight');
  if (!emb) throw new AppError('PTH_UNSUPPORTED_ARCH', 'emb_g.weight missing');

  const version = (str(root.get('version')) ?? 'v1') as 'v1' | 'v2';
  if (version !== 'v1' && version !== 'v2') throw new AppError('PTH_UNSUPPORTED_ARCH', `version ${String(version)}`);
  const f0raw = root.has('f0') ? root.get('f0') : 1;
  const f0 = f0raw === true || f0raw === 1 ? 1 : 0;
  const sr = Number(cfg[cfg.length - 1]);
  if (!SUPPORTED_SR.includes(sr)) throw new AppError('PTH_UNSUPPORTED_ARCH', `sample rate ${sr}`);
  const epoch = Number.parseInt(String(root.get('info') ?? '').replace(/epoch$/, ''), 10);
  return {
    pth, weights, version, f0: f0 === 1, sr: sr as RvcCheckpoint['sr'], nSpk: emb.shape[0],
    signature: templateSignature(version, f0, cfg),
    meta: {
      epoch: Number.isFinite(epoch) ? epoch : typeof root.get('epoch') === 'number' ? (root.get('epoch') as number) : undefined,
      vocoder, embedder, author: str(root.get('author')),
    },
  };
}
