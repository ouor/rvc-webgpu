// RvcNet: hides the input differences between our weight-less templates (§17.3) and external ONNX (F4, §7.4).
import { AppError } from '../../shared/errors';
import type { Backend, RvcModelInfo } from '../../shared/types';
import type { Prng } from '../../dsp/prng';
import { numel, toFloat32 } from '../../formats/pth';
import type { RvcCheckpoint } from '../../formats/rvc-checkpoint';
import type { OnnxInfo } from '../../formats/onnx-meta';
import { createSession, ort, releaseSession, runSession } from '../../models/ort';
import { createSessionFromParts, type Manifest } from '../../models/loader';
import { checkMeta } from '../../models/contract';
import { featDim } from './features';

export interface NetInput { phone: Float32Array; frames: number; pitch?: BigInt64Array; pitchf?: Float32Array; sid: number }

export interface RvcNet {
  info: RvcModelInfo;
  backend: Backend;
  run(i: NetInput, rng: Prng): Promise<Float32Array>;
  release(): Promise<void>;
}

/** NSF sine phase in float64 (§17.3, A's sine_phase): frac of the running sum of f0/sr, restarting at 0 per chunk. */
export function sinePhase(pitchf: Float32Array, upp: number, sr: number): Float32Array {
  const out = new Float32Array(pitchf.length * upp);
  let acc = 0;
  for (let t = 0; t < pitchf.length; t++) {
    const r = (pitchf[t] / sr) % 1;
    for (let k = 0; k < upp; k++) { acc = (acc + r) % 1; out[t * upp + k] = acc; }
  }
  return out;
}

/** Template session + the user's .pth weights; g = emb_g[sid] is gathered here (§0). */
export async function createTemplateNet(
  info: RvcModelInfo, ckpt: RvcCheckpoint, template: { graph: Uint8Array; manifest: Manifest }, backend: Backend,
): Promise<RvcNet> {
  // emb_g is read and checked before the session exists, so a failure here leaks nothing (R1 L3)
  const emb = ckpt.weights.get('emb_g.weight')!;
  const gin = emb.shape[1];
  const g = template.manifest.inputs.find((x) => x.name === 'g');
  if (g && typeof g.shape[1] === 'number' && g.shape[1] !== gin) throw new AppError('PTH_UNSUPPORTED_ARCH', `speaker embedding ${gin}, template expects ${g.shape[1]}`);
  const embAll = new Float32Array(numel(emb.shape));
  toFloat32(emb.dtype, await ckpt.pth.read(emb), embAll);
  const { session, backend: used } = await createSessionFromParts(
    { graph: template.graph, manifest: template.manifest, pth: ckpt.pth, tensors: ckpt.weights }, backend, 'PTH_UNSUPPORTED_ARCH');
  const meta = checkMeta('template', template.manifest.meta);
  const upp = meta.upp, tplSr = meta.sr;
  const hasPhase = template.manifest.inputs.some((x) => x.name === 'phase');
  return {
    info, backend: used,
    async run(i, rng) {
      const T = i.frames, C = featDim(info.version);
      if (i.sid < 0 || i.sid >= info.nSpk) throw new AppError('INTERNAL', `speaker id ${i.sid} out of range`);
      const feeds: Record<string, ort.Tensor> = {
        phone: new ort.Tensor('float32', i.phone, [1, T, C]),
        phone_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(T)), [1]),
        g: new ort.Tensor('float32', embAll.slice(i.sid * gin, (i.sid + 1) * gin), [1, gin, 1]),
        rnd: new ort.Tensor('float32', rng.gaussian(192 * T), [1, 192, T]),
      };
      if (info.f0) {
        feeds.pitch = new ort.Tensor('int64', i.pitch!, [1, T]);
        feeds.pitchf = new ort.Tensor('float32', i.pitchf!, [1, T]);
        feeds.noise = new ort.Tensor('float32', rng.gaussian(T * upp), [1, T * upp, 1]);
        if (hasPhase) feeds.phase = new ort.Tensor('float32', sinePhase(i.pitchf!, upp, tplSr), [1, T * upp, 1]);
      }
      return (await runSession(session, feeds)).audio.data as Float32Array;
    },
    release: () => releaseSession(session),
  };
}

// ---- F4: external ONNX (RVC WebUI or Applio export) ----

export type OnnxSchema = 'webui' | 'applio';

export function describeOnnx(o: OnnxInfo): { schema: OnnxSchema; f0: boolean; version: 'v1' | 'v2'; sr?: 32000 | 40000 | 48000 } {
  const names = new Set(o.inputs.map((x) => x.name));
  const phone = o.inputs.find((x) => x.name === 'phone');
  if (!phone || !names.has('phone_lengths')) throw new AppError('ONNX_UNRECOGNIZED', 'ONNX has no phone/phone_lengths inputs');
  const schema: OnnxSchema | undefined = names.has('ds') && names.has('rnd') ? 'webui' : names.has('sid') && names.has('rate') ? 'applio' : undefined;
  if (!schema) throw new AppError('ONNX_UNRECOGNIZED', 'unknown RVC ONNX input schema', [...names].join(','));
  const f0 = schema === 'webui' ? names.has('pitch') && names.has('pitchf') : names.has('pitch') && names.has('nsff0');
  const c = phone.dims[2];
  let info: Record<string, unknown> = {};
  try { info = JSON.parse(o.metadata.model_info ?? '{}'); } catch { /* no metadata */ }
  const version = c === 256 ? 'v1' : c === 768 ? 'v2' : info.version === 'v1' ? 'v1' : 'v2';
  const sr = Number(info.sr ?? info.sample_rate);
  return { schema, f0, version, sr: [32000, 40000, 48000].includes(sr) ? (sr as 32000 | 40000 | 48000) : undefined };
}

/** float32 -> IEEE half bits (round to nearest even), for fp16-exported ONNX inputs. */
function toHalf(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i];
    const x = u[0], sign = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xff, m = x & 0x7fffff;
    if (e === 255) { out[i] = sign | 0x7c00 | (m ? 0x200 : 0); continue; }
    const ne = e - 127 + 15;
    if (ne >= 31) { out[i] = sign | 0x7c00; continue; }
    if (ne <= 0) { // subnormal
      if (ne < -10) { out[i] = sign; continue; }
      const mm = (m | 0x800000) >>> (1 - ne + 13);
      out[i] = sign | mm;
      continue;
    }
    let h = sign | (ne << 10) | (m >>> 13);
    const rest = m & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h++;
    out[i] = h;
  }
  return out;
}

export async function createOnnxNet(info: RvcModelInfo, model: Uint8Array, onnx: OnnxInfo, backend: Backend): Promise<RvcNet> {
  const d = describeOnnx(onnx);
  const { session, backend: used } = await createSession(model, undefined, backend);
  const half = new Set(onnx.inputs.filter((x) => x.elemType === 10).map((x) => x.name));
  const fl = (name: string, data: Float32Array, dims: number[]) =>
    half.has(name) ? new ort.Tensor('float16', toHalf(data), dims) : new ort.Tensor('float32', data, dims);
  const out = session.outputNames.includes('audio') ? 'audio' : session.outputNames[0];
  return {
    info, backend: used,
    async run(i, rng) {
      const T = i.frames, C = featDim(info.version);
      const feeds: Record<string, ort.Tensor> = {
        phone: fl('phone', i.phone, [1, T, C]),
        phone_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(T)), [1]),
      };
      const sid = new ort.Tensor('int64', BigInt64Array.of(BigInt(i.sid)), [1]);
      if (d.schema === 'webui') {
        feeds.ds = sid;
        feeds.rnd = fl('rnd', rng.gaussian(192 * T), [1, 192, T]);
        if (d.f0) { feeds.pitch = new ort.Tensor('int64', i.pitch!, [1, T]); feeds.pitchf = fl('pitchf', i.pitchf!, [1, T]); }
      } else {
        feeds.sid = sid;
        feeds.rate = fl('rate', Float32Array.of(1), [1]);
        if (d.f0) { feeds.pitch = new ort.Tensor('int64', i.pitch!, [1, T]); feeds.nsff0 = fl('nsff0', i.pitchf!, [1, T]); }
      }
      const t = (await runSession(session, feeds))[out];
      if (t.type === 'float16') {
        const src = t.data as Uint16Array, res = new Float32Array(src.length);
        const bytes = new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
        toFloat32('float16', bytes, res);
        return res;
      }
      return t.data as Float32Array;
    },
    release: () => releaseSession(session),
  };
}

