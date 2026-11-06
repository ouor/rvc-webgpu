// Dev-only harness (debug.html, not part of the build): runs F1 through the real worker on this browser's backend and
// compares the separation with weights/reference (served by vite at /__dev/reference). Results land in window.__debug.
import { createEngineClient } from '../app/engine-client';
import { decodeFile } from '../audio/decode';
import { resample } from '../dsp/resample';
import { stft } from '../dsp/stft';
import { extractF0 } from '../engine/rvc/f0';
import { createTemplateNet, WEBGPU_MAX_FRAMES } from '../engine/rvc/net';
import { synthesizeChunk } from '../engine/rvc/pipeline';
import { readRvcCheckpoint } from '../formats/rvc-checkpoint';
import { readPth } from '../formats/pth';
import { createSessionFromParts, tensorMap, type Manifest } from '../models/loader';
import { configureOrt, releaseSession } from '../models/ort';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type BackendPref, type JobResult, type Stereo } from '../shared/types';

const q = new URLSearchParams(location.search);
const model = q.get('model') ?? 'kyubey';
const src = q.get('src') ?? 'ref';
const mode = (q.get('mode') ?? 'song') as 'song' | 'voice'; // or 'synth' (QA B3 stage test)
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

/** Pearson correlation of log-magnitude spectra (1024/512) of one second; phase-insensitive, as QA's log-mel check. */
function specCorr(a: Float32Array, b: Float32Array): number {
  const A = stft(a, 1024, 512), B = stft(b, 1024, 512);
  let n = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < A.re.length; i++) {
    const x = Math.log(1e-6 + Math.hypot(A.re[i], A.im[i])), y = Math.log(1e-6 + Math.hypot(B.re[i], B.im[i]));
    n++; sa += x; sb += y; sab += x * y; saa += x * x; sbb += y * y;
  }
  return (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
}
const rms = (x: Float32Array) => Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length);

/** QA B3 stage test: the reference feats/pitch/rnd/noise through synthesizeChunk on `backend`, vs rawIr0 per second. */
async function synth() {
  configureOrt();
  const backend = (q.get('backend') ?? 'webgpu') as 'webgpu' | 'wasm';
  const R = '/__dev/reference/bunnygirl-rvc-kyubey/';
  const meta = (await (await fetchBlob(R + 'case.json')).text());
  const chunks = (JSON.parse(meta).meta.chunks) as { audio: [number, number]; f0Frames: [number, number]; featFrames: number; T: number }[];
  const [feats, pitchf, rnd, noise, raw] = await Promise.all(['feats', 'pitchf', 'rnd', 'noise', 'rawIr0'].map((n) => fetchF32(R + n + '.bin')));
  const pitch = new BigInt64Array(await (await fetchBlob(R + 'pitch.bin')).arrayBuffer());
  const ckpt = await readRvcCheckpoint(await fetchBlob('/__dev/external/weight/rvc-kyubey.pth'));
  const base = import.meta.env.VITE_MODEL_BASE_URL ?? '/models/';
  const graph = new Uint8Array(await (await fetchBlob(base + 'rvc/v2-f0-40k.onnx')).arrayBuffer());
  const manifest = JSON.parse(await (await fetchBlob(base + 'rvc/v2-f0-40k.manifest.json')).text()) as Manifest;
  const info = { id: 'k', name: 'k', source: 'pth', version: 'v2', f0: true, sr: 40000, nSpk: ckpt.nSpk, bytes: 0 } as const;
  const net = await createTemplateNet({ ...info }, ckpt, { graph, manifest }, backend, q.get('maxFrames') ? Number(q.get('maxFrames')) : undefined);
  log(`synth on ${net.backend}, max frames per call ${q.get('maxFrames') ?? (net.backend === 'webgpu' ? WEBGPU_MAX_FRAMES : 'all')}`);
  const totalT = chunks.reduce((a, c) => a + c.T, 0);
  let fo = 0, ro = 0, no = 0, oo = 0;
  const bad: number[] = [], badSpec: number[] = [];
  let minSpec = 1, minRms = Infinity, maxRms = 0;
  let s = 0, e = 0;
  for (const c of chunks) {
    const r = c.T;
    const rc = new Float32Array(192 * r);
    for (let k = 0; k < 192; k++) rc.set(rnd.subarray(k * totalT + ro, k * totalT + ro + r), k * r);
    const draws = [rc, noise.subarray(no, no + r * 400)];
    const rng = { next: () => 0, gaussian: () => draws.shift()! };
    const f = { data: feats.subarray(fo * 768, (fo + c.featFrames) * 768), frames: c.featFrames, dim: 768 as const };
    const p = { pitch: pitch.subarray(c.f0Frames[0], c.f0Frames[1]), pitchf: pitchf.subarray(c.f0Frames[0], c.f0Frames[1]) };
    const t0 = performance.now();
    const y = await synthesizeChunk(c.audio[1] - c.audio[0], f, undefined, p, net, { ...DEFAULT_CONVERT, indexRate: 0 }, rng);
    const got = y.subarray(40000, y.length - 40000);
    const ref = raw.subarray(oo, oo + got.length);
    for (let a = 0; a + 40000 <= got.length; a += 40000) {
      let ss = 0, ee = 0;
      for (let i = a; i < a + 40000; i++) { ss += ref[i] ** 2; ee += (got[i] - ref[i]) ** 2; }
      s += ss; e += ee;
      if (10 * Math.log10(ss / ee) < 20) bad.push(Math.round((oo + a) / 40000));
      if (ss / 40000 > 1e-5) { // skip near-silent seconds
        const g1 = got.subarray(a, a + 40000), r1 = ref.subarray(a, a + 40000);
        const c = specCorr(g1, r1), q1 = rms(g1) / rms(r1);
        minSpec = Math.min(minSpec, c); minRms = Math.min(minRms, q1); maxRms = Math.max(maxRms, q1);
        if (c < 0.85) badSpec.push(Math.round((oo + a) / 40000));
        if (q1 < 0.5 || q1 > 2) log(`    second ${Math.round((oo + a) / 40000)}: RMS ratio ${q1.toFixed(2)} (ref RMS ${rms(r1).toFixed(4)}), spec corr ${c.toFixed(3)}`);
      }
    }
    log(`  chunk T=${c.T}: ${(performance.now() - t0).toFixed(0)} ms, SNR ${snrDb(got, ref).toFixed(1)} dB`);
    fo += c.featFrames; ro += r; no += r * 400; oo += got.length;
  }
  state.metrics.synthSnr = 10 * Math.log10(s / e);
  state.metrics.badSeconds = bad.length;
  state.metrics.badSpecSeconds = badSpec.length;
  log(`spectral corr per second: min ${minSpec.toFixed(3)}, under 0.85: ${badSpec.length} [${badSpec.join(',')}]; RMS ratio ${minRms.toFixed(2)}..${maxRms.toFixed(2)}`);
  log(`synth vs rawIr0: SNR ${state.metrics.synthSnr.toFixed(1)} dB, seconds under 20 dB: ${bad.length} [${bad.join(',')}]`);
  await net.release();
}

async function input(): Promise<Stereo> {
  if (src === 'mp3') return (await decodeFile(await fetchBlob('/__dev/external/audio-bunnygirl.mp3'))).audio;
  // ref: python decode (same samples as the reference); refvocals: the reference's separated vocals, so mode=voice
  // feeds the conversion exactly the reference vocal16k and the converted waveform is comparable.
  const x = await fetchF32(`/__dev/reference/bunnygirl-separate/${src === 'refvocals' ? 'vocals44k' : 'input44k'}.bin`);
  const n = x.length / 2, k = Number(q.get('loop') ?? 1); // loop=3: a ~11 min input (QA B6)
  const l = new Float32Array(n * k), r = new Float32Array(n * k);
  for (let i = 0; i < k; i++) { l.set(x.subarray(0, n), i * n); r.set(x.subarray(n), i * n); }
  return { sr: 44100, l, r };
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
  // QA B6: tab memory (main + workers, needs crossOriginIsolated), sampled while running; max per stage
  const mem = performance as Performance & { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
  const memMax: Record<string, number> = {};
  let stageNow = 'setup', memOn = !!mem.measureUserAgentSpecificMemory;
  const sample = async () => {
    while (memOn) {
      const b = (await mem.measureUserAgentSpecificMemory!().catch(() => ({ bytes: 0 }))).bytes;
      memMax[stageNow] = Math.max(memMax[stageNow] ?? 0, b);
      await new Promise((r) => setTimeout(r, 2000));
    }
  };
  void sample();
  // QA B5: cancelAt=N cancels at separation patch N and times how long run() takes to reject
  const cancelAt = Number(q.get('cancelAt') ?? 0);
  let cancelT = 0;
  const r = await engine.run(job, (p) => {
    stageNow = p.stage;
    const k = `${p.stage}/${p.phase ?? 'run'}/${p.backend ?? ''}${p.reused ? ' (reused)' : ''}`;
    if (k !== last) { log(`stage ${k} ${p.done}/${p.total}  +${((performance.now() - t0) / 1000).toFixed(1)}s`); last = k; }
    else if (p.stage === 'separate' && p.done === 1) log(`  first patch +${((performance.now() - t0) / 1000).toFixed(1)}s`);
    if (cancelAt && !cancelT && p.stage === 'separate' && p.phase === 'run' && p.done >= cancelAt) { cancelT = performance.now(); void engine.cancel(); }
    const cancelMs = Number(q.get('cancelMs') ?? 0); // cancel this long after separation starts (pass 1 = analysis)
    if (cancelMs && !cancelT && p.stage === 'separate' && p.phase === 'run') { cancelT = -1; setTimeout(() => { cancelT = performance.now(); void engine.cancel(); }, cancelMs); }
  }).catch((e) => {
    memOn = false;
    if (cancelT > 0) log(`cancel -> rejected (${e?.code}) after ${((performance.now() - cancelT) / 1000).toFixed(2)} s (QA B5 <= 5 s)`);
    throw e;
  });
  memOn = false;
  if (Object.keys(memMax).length) log('tab memory max MB per stage ' + JSON.stringify(Object.fromEntries(Object.entries(memMax).map(([k, v]) => [k, Math.round(v / 2 ** 20)]))));
  state.timings = r.timings;
  log('timings ms ' + JSON.stringify(r.timings));
  log(`converted ${r.converted.data.length} samples @ ${r.converted.sr}`);
  if (q.get('metrics') !== '0' && r.vocals && src === 'ref' && !range) {
    const voc = await fetchF32('/__dev/reference/bunnygirl-separate/vocals44k.bin');
    const m = voc.length / 2;
    state.metrics.vocalsSnrL = snrDb(r.vocals.l.subarray(0, m), voc.subarray(0, m));
    state.metrics.vocalsSnrR = snrDb(r.vocals.r.subarray(0, m), voc.subarray(m));
    log(`separate vs reference: SNR L ${state.metrics.vocalsSnrL.toFixed(1)} dB, R ${state.metrics.vocalsSnrR.toFixed(1)} dB (need > 30)`);
    const per: string[] = []; // per 10 s, to tell segment seams (60 s, 4 s crossfade) from uniform error
    for (let a = 0; a + 441000 <= m; a += 441000) per.push(snrDb(r.vocals.l.subarray(a, a + 441000), voc.subarray(a, a + 441000)).toFixed(0));
    log('  per 10 s: ' + per.join(' '));
  }
  if (q.get('metrics') !== '0' && src.startsWith('ref') && !range && model === 'kyubey') {
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

document.getElementById('run')!.onclick = () => (q.get('mode') === 'synth' ? synth() : run()).catch((e) => { state.error = String(e?.code ?? '') + ' ' + String(e?.message ?? e); log('ERROR ' + state.error); }).finally(() => { state.done = true; });
if (q.has('autorun')) document.getElementById('run')!.click();
