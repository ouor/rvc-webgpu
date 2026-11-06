// Dev-only harness (debug.html, not part of the build): runs F1 through the real worker on this browser's backend and
// compares the separation with weights/reference (served by vite at /__dev/reference). Results land in window.__debug.
import { createEngineClient } from '../app/engine-client';
import { decodeFile } from '../audio/decode';
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

async function input(): Promise<Stereo> {
  if (src === 'mp3') return (await decodeFile(await fetchBlob('/__dev/external/audio-bunnygirl.mp3'))).audio;
  const x = await fetchF32('/__dev/reference/bunnygirl-separate/input44k.bin'); // python decode: same samples as the reference
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
  const t1 = performance.now();
  const again = await engine.run({ ...job, convert: { ...DEFAULT_CONVERT, f0UpKey: 2 } }, () => {});
  state.metrics.rerunKeyMs = performance.now() - t1;
  log(`re-run with key +2: ${state.metrics.rerunKeyMs.toFixed(0)} ms, reused ${again.reused.join(',')}`);
  log(`total ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  engine.terminate();
}

document.getElementById('run')!.onclick = () => run().catch((e) => { state.error = String(e?.code ?? '') + ' ' + String(e?.message ?? e); log('ERROR ' + state.error); }).finally(() => { state.done = true; });
if (q.has('autorun')) document.getElementById('run')!.click();
