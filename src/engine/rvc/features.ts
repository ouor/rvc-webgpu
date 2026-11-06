// ContentVec / HuBERT features (pipeline.py vc(): output_layer 12 for v2, 9 + final_proj for v1).
import { f32, runSession, type OrtSession } from '../../models/ort';

export interface Feats { data: Float32Array; frames: number; dim: 256 | 768 } // 50 fps, before index blending

/** Feature (and index) dimension of an RVC version. */
export const featDim = (version: 'v1' | 'v2'): 256 | 768 => (version === 'v1' ? 256 : 768);

export async function extractFeatures(chunk16k: Float32Array, hubert: OrtSession, version: 'v1' | 'v2'): Promise<Feats> {
  const name = version === 'v1' ? 'feats256' : 'feats768';
  const r = await runSession(hubert, { [hubert.inputNames[0]]: f32(chunk16k, [1, chunk16k.length]) });
  const t = r[name];
  return { data: t.data as Float32Array, frames: Number(t.dims[1]), dim: featDim(version) };
}
