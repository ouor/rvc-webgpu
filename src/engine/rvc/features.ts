// ContentVec / HuBERT features (pipeline.py vc(): output_layer 12 for v2, 9 + final_proj for v1).
import { f32, runSession, type OrtSession } from '../../models/ort';

export interface Feats { data: Float32Array; frames: number; dim: 256 | 768 } // 50 fps, before index blending

export async function extractFeatures(chunk16k: Float32Array, hubert: OrtSession, version: 'v1' | 'v2'): Promise<Feats> {
  const name = version === 'v1' ? 'feats256' : 'feats768';
  const r = await runSession(hubert, { [hubert.inputNames[0]]: f32(chunk16k, [1, chunk16k.length]) });
  const t = r[name];
  const dim = version === 'v1' ? 256 : 768;
  return { data: t.data as Float32Array, frames: Number(t.dims[1]), dim };
}
