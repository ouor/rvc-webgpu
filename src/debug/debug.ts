// Dev-only harness (debug.html, not part of the build): runs F1 through the real worker on this browser's backend and
// compares the separation with weights/reference (served by vite at /__dev/reference). Results land in window.__debug.
import { createEngineClient } from '../app/engine-client';
import { decodeFile } from '../audio/decode';
import { resample } from '../dsp/resample';
import { extractF0 } from '../engine/rvc/f0';
import { readPth } from '../formats/pth';
import { createSessionFromParts, tensorMap, type Manifest } from '../models/loader';
import { configureOrt, releaseSession } from '../models/ort';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type BackendPref, type JobResult, type Stereo } from '../shared/types';

const q = new URLSearchParams(location.search);
const model = q.get('model') ?? 'kyubey';
const src = q.get('src') ?? 'ref';
const mode = (q.get('mode') ?? 'song') as 'song' | 'voice';
const range = q.get('end') ? { start: Number(q.get('start') ?? 0), end: Number(q.get('end')) } : undefined;
const out = document.getElementById('log')!;
const lines: string[] = [];
const log = (s: string) => { lines.push(s); out.textContent = lines.join('\n'); };
const state: { done: boolean; error?: string; log: string[]; timings?: JobResult['timings']; metrics: Record<string, number> } = { done: false, log: lines, metrics: {} };
(window as unknown as { __debug: typeof state }).__debug = state;

const fetchBlob = async (url: string) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.blob(); };
const fetchF32 = async (url: string) => new Float32Array(await (await fetchBlob(url)).arrayBuffer());

function snrDb(got: Float32Array, ref: Float32Array): number {
  let s = 0, e = 0;
  for (let i = 0; i < ref.length; i++) { s += ref[i] ** 2; e += (got[i] - ref[i]) ** 2; }
  return 10 * Math.log10(s / e);
}

function logF0Corr(a: ArrayLike<number>, b: ArrayLike<number>, off: number): number {
  let n = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length && i + off < b.length; i++) {
    if (!(a[i] > 0 && b[i + off] > 0)) continue;
    const x = Math.log(a[i]), y = Math.log(b[i + off]);
    n++; sa += x; sb += y; sab += x * y; saa += x * x; sbb += y * y;
  }
  return (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
}

/** RMVPE (WebGPU, main thread, served model files) on 40 kHz signals. */
async function f0Of(signals: Float32Array[]): Promise<Float64Array[]> {
  configureOrt();
  const base = import.meta.env.VITE_MODEL_BASE_URL ?? '/models/';
  const pth = await readPth(await fetchBlob(base + 'rmvpe/rmvpe.pth'));
  const graph = new Uint8Array(await (await fetchBlob(base + 'rmvpe/rmvpe.onnx')).arrayBuffer());
  const manifest = JSON.parse(await (await fetchBlob(base + 'rmvpe/rmvpe.manifest.json')).text()) as Manifest;
  const { session, backend } = await createSessionFromParts({ graph, manifest, pth, tensors: tensorMap(pth.root) }, 'webgpu', 'CACHE_INTEGRITY');
  const ctx = { backend, progress: () => {}, throwIfCancelled: () => {}, rng: { next: () => 0, gaussian: (n: number) => new Float32Array(n) } };
  try {
    const out: Float64Array[] = [];
    for (const x of signals) out.push(await extractF0(resample(x, 40000, 16000), session, ctx, manifest.meta as never));
    return out;
  } finally { await releaseSession(session); }
}

async function input(): Promise<Stereo> {
  if (src === 'mp3') return (await decodeFile(await fetchBlob('/__dev/external/audio-bunnygirl.mp3'))).audio;
  // ref: python decode (same samples as the reference); refvocals: the reference's separated vocals, so mode=voice
  // feeds the conversion exactly the reference vocal16k and the converted waveform is comparable.
  const x = await fetchF32(`/__dev/reference/bunnygirl-separate/${src === 'refvocals' ? 'vocals44k' : 'input44k'}.bin`);
  const n = x.length / 2;
  return { sr: 44100, l: x.slice(0, n), r: x.slice(n) };
}

async function run() {
  const t0 = performance.now();
  const engine = createEngineClient();
  const env = await engine.init({ backend: (q.get('backend') ?? 'auto') as BackendPref, modelBaseUrl: import.meta.env.VITE_MODEL_BASE_URL ?? '/models/' });
  log('env ' + JSON.stringify(env));
  const audio = await input();
  const inputId = await engine.setInput(audio, { name: 'bunnygirl', size: 0, lastModified: 0 });
  const rvc = await engine.loadRvcModel(new File([await fetchBlob(`/__dev/external/weight/rvc-${model}.pth`)], `rvc-${model}.pth`));
  log('model ' + JSON.stringify(rvc));
  const index = await engine.loadIndex(new File([await fetchBlob(`/__dev/external/weight/rvc-${model}.index`)], `rvc-${model}.index`));
  log('index ' + JSON.stringify(index));
  const job = { inputId, range, mode, separate: DEFAULT_SEPARATE, modelId: rvc.id, indexId: index.id, convert: DEFAULT_CONVERT };
  const missing = (await engine.plan(job)).filter((d) => !d.cached);
  if (missing.length) {
    log('download ' + missing.map((d) => d.id).join(', '));
    await engine.download(missing.map((d) => d.id), (p) => { if (p.done === p.total) log(`  ${p.file ?? p.stage} ${p.phase ?? ''} done`); });
  }
  let last = '';
  const r = await engine.run(job, (p) => {
    const k = `${p.stage}/${p.phase ?? 'run'}/${p.backend ?? ''}${p.reused ? ' (reused)' : ''}`;
    if (k !== last) { log(`stage ${k} ${p.done}/${p.total}  +${((performance.now() - t0) / 1000).toFixed(1)}s`); last = k; }
  });
  state.timings = r.timings;
  log('timings ms ' + JSON.stringify(r.timings));
  log(`converted ${r.converted.data.length} samples @ ${r.converted.sr}`);
  if (r.vocals && src === 'ref' && !range) {
    const voc = await fetchF32('/__dev/reference/bunnygirl-separate/vocals44k.bin');
    const m = voc.length / 2;
    state.metrics.vocalsSnrL = snrDb(r.vocals.l.subarray(0, m), voc.subarray(0, m));
    state.metrics.vocalsSnrR = snrDb(r.vocals.r.subarray(0, m), voc.subarray(m));
    log(`separate vs reference: SNR L ${state.metrics.vocalsSnrL.toFixed(1)} dB, R ${state.metrics.vocalsSnrR.toFixed(1)} dB (need > 30)`);
    const per: string[] = []; // per 10 s, to tell segment seams (60 s, 4 s crossfade) from uniform error
    for (let a = 0; a + 441000 <= m; a += 441000) per.push(snrDb(r.vocals.l.subarray(a, a + 441000), voc.subarray(a, a + 441000)).toFixed(0));
    log('  per 10 s: ' + per.join(' '));
  }
  if (src.startsWith('ref') && !range && model === 'kyubey') {
    const ref = await fetchF32('/__dev/reference/bunnygirl-rvc-kyubey/convertedIr75.bin');
    const m = Math.min(ref.length, r.converted.data.length);
    // Waveforms differ (our PRNG vs the reference's numpy rnd/noise), so judge AC-2: RMVPE log-f0 of the output vs
    // the input vocal (reference f0Hz of the padded input, +100 frames = t_pad), next to the same for the reference output.
    const f0In = await fetchF32('/__dev/reference/bunnygirl-rvc-kyubey/f0Hz.bin');
    const [ours, theirs] = await f0Of([r.converted.data.subarray(0, m), ref.subarray(0, m)]);
    state.metrics.f0CorrOurs = logF0Corr(ours, f0In, 100);
    state.metrics.f0CorrRef = logF0Corr(theirs, f0In, 100);
    log(`converted log-f0 corr vs input (AC-2 >= 0.95): ours ${state.metrics.f0CorrOurs.toFixed(4)}, reference ${state.metrics.f0CorrRef.toFixed(4)}`);
  }
  const t1 = performance.now();
  const again = await engine.run({ ...job, convert: { ...DEFAULT_CONVERT, f0UpKey: 2 } }, () => {});
  state.metrics.rerunKeyMs = performance.now() - t1;
  log(`re-run with key +2: ${state.metrics.rerunKeyMs.toFixed(0)} ms, reused ${again.reused.join(',')}`);
  log(`total ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  engine.terminate();
}

document.getElementById('run')!.onclick = () => run().catch((e) => { state.error = String(e?.code ?? '') + ' ' + String(e?.message ?? e); log('ERROR ' + state.error); }).finally(() => { state.done = true; });
if (q.has('autorun')) document.getElementById('run')!.click();
