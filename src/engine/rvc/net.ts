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

// QA B3: ORT WebGPU (1.23) produces wrong audio after ~23 s in synthesis calls of T >= ~4300 frames (WASM matches
// the reference at 104 dB). 3600-frame pieces still lost level in one chunk; 3000 matched WASM on spectrum and RMS.
// On WebGPU longer chunks are synthesized in pieces (debug.html?mode=synth measures this).
export const WEBGPU_MAX_FRAMES = 3000;
const MIN_CONTEXT = 100; // frames (1 s): each piece also sees at least this much input on each side (RVC's x_pad)

/**
 * Pieces of [0, T): `s..e` is fed, `a..b` is kept. Every piece is fed a full `maxT` window around its core, so it sees
 * as much context as fits (the encoder attends over the whole input). One piece when T <= maxT.
 */
export function synthPieces(T: number, maxT: number): { s: number; e: number; a: number; b: number }[] {
  if (T <= maxT) return [{ s: 0, e: T, a: 0, b: T }];
  const n = Math.ceil(T / (maxT - 2 * MIN_CONTEXT)), core = Math.ceil(T / n), ctx = Math.floor((maxT - core) / 2);
  const out = [];
  for (let a = 0; a < T; a += core) {
    const b = Math.min(T, a + core);
    const s = Math.min(Math.max(0, a - ctx), T - maxT);
    out.push({ s, e: s + maxT, a, b });
  }
  return out;
}

/** Runs `piece(s, e)` (returns (e - s) * upp samples) over synthPieces and joins the kept parts. */
export async function runPieces(T: number, upp: number, maxT: number, piece: (s: number, e: number) => Promise<Float32Array>): Promise<Float32Array> {
  const ps = synthPieces(T, maxT);
  if (ps.length === 1) return piece(0, T);
  const out = new Float32Array(T * upp);
  for (const p of ps) out.set((await piece(p.s, p.e)).subarray((p.a - p.s) * upp, (p.b - p.s) * upp), p.a * upp);
  return out;
}

/** Columns s..e of a row-major [rows, T] array, as [rows, e - s]. */
function cols(x: Float32Array, rows: number, T: number, s: number, e: number): Float32Array {
  if (s === 0 && e === T) return x;
  const n = e - s, out = new Float32Array(rows * n);
  for (let r = 0; r < rows; r++) out.set(x.subarray(r * T + s, r * T + e), r * n);
  return out;
}

/** Template session + the user's .pth weights; g = emb_g[sid] is gathered here (§0). */
export async function createTemplateNet(
  info: RvcModelInfo, ckpt: RvcCheckpoint, template: { graph: Uint8Array; manifest: Manifest }, backend: Backend, maxFrames?: number,
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
      // whole-chunk noise and phase, sliced per piece: same rng order and a continuous NSF phase
      const rnd = rng.gaussian(192 * T);
      const noise = info.f0 ? rng.gaussian(T * upp) : undefined;
      const phase = info.f0 && hasPhase ? sinePhase(i.pitchf!, upp, tplSr) : undefined;
      const g = new ort.Tensor('float32', embAll.slice(i.sid * gin, (i.sid + 1) * gin), [1, gin, 1]);
      return runPieces(T, upp, maxFrames ?? (used === 'webgpu' ? WEBGPU_MAX_FRAMES : Infinity), async (s, e) => {
        const n = e - s;
        const feeds: Record<string, ort.Tensor> = {
          phone: new ort.Tensor('float32', i.phone.slice(s * C, e * C), [1, n, C]),
          phone_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(n)), [1]),
          g,
          rnd: new ort.Tensor('float32', cols(rnd, 192, T, s, e), [1, 192, n]),
        };
        if (info.f0) {
          feeds.pitch = new ort.Tensor('int64', i.pitch!.slice(s, e), [1, n]);
          feeds.pitchf = new ort.Tensor('float32', i.pitchf!.slice(s, e), [1, n]);
          feeds.noise = new ort.Tensor('float32', noise!.slice(s * upp, e * upp), [1, n * upp, 1]);
          if (phase) feeds.phase = new ort.Tensor('float32', phase.slice(s * upp, e * upp), [1, n * upp, 1]);
        }
        return (await runSession(session, feeds)).audio.data as Float32Array;
      });
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
  // QA B12: some WebUI exports fix the time axis; they used to fail at run time with an ORT-internal number
  if (typeof phone.dims[1] === 'number') throw new AppError('ONNX_UNRECOGNIZED', 'fixed-length ONNX export (phone time axis is not dynamic)', `phone ${JSON.stringify(phone.dims)}`);
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
      const rnd = d.schema === 'webui' ? rng.gaussian(192 * T) : undefined;
      return runPieces(T, info.sr / 100, used === 'webgpu' ? WEBGPU_MAX_FRAMES : Infinity, async (s, e) => {
        const n = e - s;
        const feeds: Record<string, ort.Tensor> = {
          phone: fl('phone', i.phone.slice(s * C, e * C), [1, n, C]),
          phone_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(n)), [1]),
        };
        const sid = new ort.Tensor('int64', BigInt64Array.of(BigInt(i.sid)), [1]);
        const pitch = () => new ort.Tensor('int64', i.pitch!.slice(s, e), [1, n]);
        if (d.schema === 'webui') {
          feeds.ds = sid;
          feeds.rnd = fl('rnd', cols(rnd!, 192, T, s, e), [1, 192, n]);
          if (d.f0) { feeds.pitch = pitch(); feeds.pitchf = fl('pitchf', i.pitchf!.slice(s, e), [1, n]); }
        } else {
          feeds.sid = sid;
          feeds.rate = fl('rate', Float32Array.of(1), [1]);
          if (d.f0) { feeds.pitch = pitch(); feeds.nsff0 = fl('nsff0', i.pitchf!.slice(s, e), [1, n]); }
        }
        const t = (await runSession(session, feeds))[out];
        if (t.type === 'float16') {
          const src = t.data as Uint16Array, res = new Float32Array(src.length);
          toFloat32('float16', new Uint8Array(src.buffer, src.byteOffset, src.byteLength), res);
          return res;
        }
        return t.data as Float32Array;
      });
    },
    release: () => releaseSession(session),
  };
}

