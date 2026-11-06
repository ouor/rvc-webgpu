// UI actions: the only place that talks to the engine (via engine-client) and to src/audio, and the only writer of the store.
import { toast } from 'sonner';
import { createEngineClient } from './engine-client';
import { decodeFile, peaks, toAudioBuffer } from '@/audio/decode';
import { renderMix, type MixParams } from '@/audio/mixer';
import { encodeAudio, type AudioFormat } from '@/audio/wav';
import { toAppError, type AppError } from '@/shared/errors';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type BackendPref, type JobSpec, type Progress, type RvcModelInfo, type Stereo } from '@/shared/types';
import { AUDIO_EXT, INDEX_EXT, MODEL_EXT, download, extOf, safe, stripExt, type RunStage } from './lib/format';
import { DEFAULT_BASE_URL, STEPS, confirmAsync, get, isBusy, selEngineBusy, selModelBaseUrl, set, type Job, type Options, type Result, type Settings, type SettingsTab, type Step, type StepInfo } from './store';
import { version } from '../../package.json';
import { errText, t } from './i18n';

// Engine rejections are always AppError (ENGINE_API §6); toAppError stays only where UI-thread code (decode, encode, render) can throw.
const engineErr = (e: unknown) => e as AppError;
const reason = (e: unknown) => ({ description: errText(engineErr(e)).title });

const engine = createEngineClient();
const RUNNING_FLAG = 'web-rvc:running'; // DESIGN S6: set while processing; still present on load = the tab died

// ---------- environment ----------

export async function initEngine() {
  set({ engineReady: false, envError: undefined });
  try {
    const env = await engine.init({ backend: get().settings.backend, modelBaseUrl: selModelBaseUrl(get()) });
    set({ env, engineReady: true });
    void refreshPlan();
    void refreshLists();
  } catch (e) {
    set({ envError: engineErr(e) });
  }
}

export function detectCrash() {
  try {
    if (sessionStorage.getItem(RUNNING_FLAG)) { set({ crashed: true }); sessionStorage.removeItem(RUNNING_FLAG); }
  } catch { /* storage blocked */ }
}

const setRunningFlag = (on: boolean) => { try { if (on) sessionStorage.setItem(RUNNING_FLAG, '1'); else sessionStorage.removeItem(RUNNING_FLAG); } catch { /* ignore */ } };
// A deliberate reload/close fires pagehide; a crashed tab does not, so the flag survives only crashes.
addEventListener('pagehide', () => setRunningFlag(false));

export const dismiss = (id: string) => set({ dismissed: [...get().dismissed, id] });

// Stale async results: each call takes a ticket; only the newest ticket per slot may write back.
const tickets: Record<'song' | 'model' | 'index' | 'plan' | 'lists', number> = { song: 0, model: 0, index: 0, plan: 0, lists: 0 };
const latest = (slot: keyof typeof tickets) => { const my = ++tickets[slot]; return () => tickets[slot] === my; };

// ---------- UI state ----------

export const updateSettings = (p: Partial<Settings>) => set((s) => ({ settings: { ...s.settings, ...p } }));
export const openSettings = (tab?: SettingsTab) => { set({ settingsTab: tab }); if (tab) void refreshLists(); };
export const setOnline = () => set({ online: navigator.onLine });
export const setResolvedTheme = (resolvedTheme: 'light' | 'dark') => set({ resolvedTheme });
export const setAB = (ab: 'A' | 'B') => set({ ab });
export const setMix = (p: Partial<MixParams>) => set((s) => ({ mix: { ...s.mix, ...p } }));
export const setFormat = (format: AudioFormat) => set({ format });
export const ackNotice = () => updateSettings({ noticeAck: true });
export const showNoticeAgain = () => { updateSettings({ noticeAck: false }); openSettings(undefined); };
export const answerConfirm = (ok: boolean) => { get().confirm?.resolve(ok); set({ confirm: undefined }); };

export function setBackend(backend: BackendPref) {
  if (backend === get().settings.backend) return;
  updateSettings({ backend });
  void initEngine();
}

/** A3: the deployment default is never stored, so a redeploy with a new default reaches everyone who did not change it. */
export function setModelBaseUrl(url: string | undefined) {
  updateSettings({ modelBaseUrl: url === DEFAULT_BASE_URL ? undefined : url });
  void initEngine();
}

/** Banners: open the advanced options and limit the range to 30 s. */
export function shortenToClip() {
  const d = get().song?.duration;
  updateSettings({ advancedOpen: true });
  if (d && d > 30) setOpts({ range: { start: 0, end: 30 } });
  document.getElementById('advanced')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

export function copyEnvInfo() {
  const s = get();
  const info = { app: version, ua: navigator.userAgent, env: s.env, backend: s.settings.backend, modelBase: selModelBaseUrl(s), deviceMemory: (navigator as { deviceMemory?: number }).deviceMemory, cores: navigator.hardwareConcurrency, lang: s.settings.lang };
  void navigator.clipboard.writeText(JSON.stringify(info, null, 2));
  toast(t('toast.copied'));
}

/** Saved models and cache listing (settings, my-models picker); refreshed after every action that changes them. */
export async function refreshLists() {
  if (!get().engineReady) return;
  const current = latest('lists');
  const [myModels, entries, est, persisted] = await Promise.all([
    engine.myModels.list().catch(() => []),
    engine.cache.list().catch(() => []),
    navigator.storage?.estimate?.().catch(() => undefined),
    navigator.storage?.persisted?.().catch(() => false),
  ]);
  if (current()) set({ myModels, cache: { entries, usage: est?.usage ?? 0, quota: est?.quota ?? 0, persisted: !!persisted } });
}

// ---------- files ----------

/** Routes dropped/picked files by extension (DESIGN Dropzone: whole-page drop allowed). */
export async function addFiles(files: File[]) {
  const songs = files.filter((f) => AUDIO_EXT.includes(extOf(f.name)) || (f.type.startsWith('audio/') && !MODEL_EXT.includes(extOf(f.name))));
  const models = files.filter((f) => MODEL_EXT.includes(extOf(f.name)));
  const indexes = files.filter((f) => INDEX_EXT.includes(extOf(f.name)));
  for (const f of files) if (!songs.includes(f) && !models.includes(f) && !indexes.includes(f)) toast(t('toast.badExt', { ext: extOf(f.name) || f.name }));
  if (!songs.length && !models.length && !indexes.length) return;
  if (!(await guardReplace())) return;
  if (songs[0]) void setSong(songs[0]);
  if (models[0]) await setModel(models[0]);
  if (indexes[0]) void setIndex(indexes[0]);
}

/** ConfirmDialog before replacing inputs while a job runs or a result exists. */
async function guardReplace() {
  const s = get();
  if (isBusy()) {
    const ok = await confirmAsync({ title: t('confirm.stopTitle'), body: t('confirm.stopBody'), action: t('confirm.stopAction'), destructive: true });
    if (!ok) return false;
    await cancelJob();
  } else if (s.result && s.view === 'result') {
    const ok = await confirmAsync({ title: t('confirm.replaceTitle'), body: t('confirm.replaceBody'), action: t('confirm.replaceAction') });
    if (!ok) return false;
  }
  return true;
}

export async function setSong(file: File) {
  const current = latest('song');
  set({ song: { status: 'parsing', name: file.name, size: file.size }, view: 'work', result: undefined, prev: undefined, stems: undefined, opts: { ...get().opts, range: undefined } });
  try {
    const dec = await decodeFile(file);
    const pk = peaks(dec.audio.l, 1200);
    const inputId = await engine.setInput(dec.audio, { name: file.name, size: file.size, lastModified: file.lastModified });
    if (!current()) return; // replaced meanwhile
    set({ song: { status: 'ready', name: file.name, size: file.size, duration: dec.duration, channels: dec.originalChannels, peaks: pk, buffer: dec.buffer, inputId } });
    toast(t('toast.songLoaded', { name: stripExt(file.name) }));
    void readSourceRate(file, current);
    void refreshPlan();
  } catch (e) {
    if (current()) set({ song: { status: 'error', name: file.name, size: file.size, error: toAppError(e, 'AUDIO_DECODE') } });
  }
}

/** IN-2: the original sample rate (decodeAudioData resamples to 44.1k, so read it from the container). */
async function readSourceRate(file: File, current: () => boolean) {
  try {
    const mb = await import('mediabunny');
    const input = new mb.Input({ source: new mb.BlobSource(file), formats: mb.ALL_FORMATS });
    const track = await input.getPrimaryAudioTrack();
    const s = get().song;
    if (track && s && current()) set({ song: { ...s, sourceRate: track.sampleRate } });
  } catch { /* metadata is optional */ }
}

export const clearSong = () => { latest('song'); set({ song: undefined, plan: undefined, result: undefined, prev: undefined, stems: undefined, view: 'work' }); };

/** A model brings its own index (zip / saved model) or none; a bundled index never outlives its model, a user-added one stays. */
function applyModelIndex(info: RvcModelInfo, fileName: string) {
  if (info.indexId) { latest('index'); set({ index: { status: 'ready', origin: 'bundled', fileName, id: info.indexId, info: info.index } }); }
  else if (get().index?.origin === 'bundled') { latest('index'); set({ index: undefined }); }
}

export async function setModel(file: File) {
  const current = latest('model');
  set({ model: { status: 'parsing', fileName: file.name, fromZip: extOf(file.name) === 'zip' }, plan: undefined });
  try {
    const info = await engine.loadRvcModel(file);
    if (!current()) return;
    set({ model: { status: 'ready', fileName: file.name, fromZip: extOf(file.name) === 'zip', info } });
    applyModelIndex(info, file.name);
    setOpts({ sid: 0 }); // speaker ids are per model
    toast(t('toast.modelLoaded', { name: info.name, v: info.version, sr: `${info.sr / 1000}kHz` }) + (info.indexId ? ` · ${t('toast.zipIndex')}` : ''));
    void refreshPlan();
  } catch (e) {
    if (current()) set({ model: { status: 'error', fileName: file.name, error: engineErr(e) } });
  }
}

export async function openMyModel(id: string) {
  if (!(await guardReplace())) return;
  const current = latest('model');
  set({ model: { status: 'parsing', fileName: id, fromMine: true }, settingsTab: undefined });
  try {
    const info = await engine.myModels.open(id);
    if (!current()) return;
    set({ model: { status: 'ready', fileName: info.name, fromMine: true, info, saved: true } });
    void refreshLists(); // last used
    applyModelIndex(info, info.name);
    setOpts({ sid: 0 });
    void refreshPlan();
  } catch (e) {
    if (current()) set({ model: { status: 'error', fileName: id, error: engineErr(e) } });
  }
}

export const clearModel = () => { latest('model'); latest('index'); set({ model: undefined, index: undefined, plan: undefined }); };

export async function setIndex(file: File) {
  const current = latest('index');
  set({ index: { status: 'parsing', origin: 'user', fileName: file.name } });
  try {
    const info = await engine.loadIndex(file);
    if (!current()) return;
    set({ index: { status: 'ready', origin: 'user', fileName: file.name, id: info.id, info } });
    if (get().model?.saved) void saveMine(true);
  } catch (e) {
    if (current()) set({ index: { status: 'error', origin: 'user', fileName: file.name, error: engineErr(e) } });
  }
}

export const clearIndex = () => { latest('index'); set({ index: undefined }); };

export async function setOnnxSr(sr: 32000 | 40000 | 48000) {
  const m = get().model;
  if (!m?.info) return;
  await engine.setOnnxModelSr(m.info.id, sr);
  set({ model: { ...m, info: { ...m.info, sr, srUnknown: false } } });
}

/** MD-9: keep the model (+ index) in OPFS. Off removes it again. */
export async function saveMine(on: boolean) {
  const m = get().model;
  if (!m?.info) return;
  try {
    if (on) await engine.myModels.save(m.info.id, get().index?.id);
    else await engine.myModels.remove(m.info.id);
    const cur = get().model;
    if (cur) set({ model: { ...cur, saved: on } });
    void refreshLists();
    if (on) toast.success(t('toast.savedMine', { name: m.info.name }));
  } catch (e) {
    toast.error(t('toast.saveFailed'), reason(e));
  }
}

export async function removeMine(id: string, name: string) {
  const ok = await confirmAsync({ title: t('confirm.delModelTitle', { name }), body: t('confirm.delModelBody'), action: t('common.delete'), destructive: true });
  if (!ok) return;
  try { await engine.myModels.remove(id); } catch (e) { toast.error(t('toast.deleteFailed'), reason(e)); }
  const m = get().model;
  if (m?.info?.id === id) set({ model: { ...m, saved: false } });
  void refreshLists();
}

// ---------- options / plan ----------

export function setOpts(patch: Partial<Options>) {
  const modeChanged = patch.mode && patch.mode !== get().opts.mode;
  set({ opts: { ...get().opts, ...patch } });
  if (modeChanged) void refreshPlan();
}

export function buildJob(profile: 'normal' | 'low' = get().crashed ? 'low' : 'normal'): JobSpec | undefined {
  const { song, model, index, opts } = get();
  if (!song?.inputId || !model?.info) return undefined;
  return {
    inputId: song.inputId,
    mode: opts.mode,
    range: opts.range,
    separate: { ...DEFAULT_SEPARATE, aggressiveness: opts.aggressiveness },
    modelId: model.info.id,
    indexId: index?.status === 'ready' ? index.id : undefined,
    convert: {
      ...DEFAULT_CONVERT, chunkProfile: profile,
      f0UpKey: model.info.f0 ? opts.key : 0,
      indexRate: index?.status === 'ready' ? opts.indexRate : 0,
      protect: opts.protect, rmsMixRate: opts.rmsMixRate,
      sid: Math.min(opts.sid, model.info.nSpk - 1),
    },
  };
}

export async function refreshPlan() {
  const current = latest('plan');
  const job = buildJob();
  if (!job || !get().engineReady) return set({ plan: undefined });
  try { const plan = await engine.plan(job); if (current()) set({ plan }); } catch { if (current()) set({ plan: undefined }); }
}

// ---------- start: plan -> (download) -> run ----------

let pending: { profile: 'normal' | 'low'; rerun: boolean } | undefined;

export async function start(opts: { rerun?: boolean; profile?: 'normal' | 'low' } = {}) {
  const job = buildJob(opts.profile);
  if (!job || selEngineBusy(get())) return;
  pending = { profile: job.convert.chunkProfile, rerun: !!opts.rerun };
  let plan;
  try { plan = await engine.plan(job); } catch (e) { return toast.error(t('err.INTERNAL.title'), reason(e)); }
  const missing = plan.filter((d) => !d.cached);
  if (missing.length) {
    // AC-18: nothing is requested before the user agrees.
    set({ dl: { status: navigator.onLine ? 'confirm' : 'offline', open: true, items: missing, done: 0, total: missing.reduce((a, d) => a + d.bytes, 0) } });
    return;
  }
  await run(pending.rerun, pending.profile);
}

export async function startDownload(retry = false) {
  const dl = get().dl;
  if (!dl || selEngineBusy(get())) return;
  if (!navigator.onLine) return set({ dl: { ...dl, status: 'offline' } });
  try {
    let items = dl.items;
    if (retry) {
      const job = buildJob(pending?.profile);
      if (job) items = (await engine.plan(job)).filter((d) => !d.cached);
    }
    const total = items.reduce((a, d) => a + d.bytes, 0);
    set({ dl: { ...dl, status: 'loading', items, done: 0, total, got: {}, cur: undefined, startedAt: performance.now(), error: undefined } });
    await engine.download(items.map((d) => d.id), (p: Progress) => {
      const cur = get().dl;
      if (cur) set({ dl: { ...cur, done: p.done, total: p.total || total, ...(p.item && { cur: p.item, got: { ...cur.got, [p.item]: p.itemDone ?? 0 } }) } });
    });
    set({ dl: undefined });
    void refreshPlan();
    void refreshLists();
    await run(pending?.rerun ?? false, pending?.profile ?? 'normal');
  } catch (e) {
    const err = engineErr(e);
    const cur = get().dl;
    if (err.code === 'CANCELLED') { set({ dl: undefined }); toast(t('toast.dlStopped')); void refreshPlan(); return; }
    set({ dl: { ...(cur ?? dl), open: true, status: err.code === 'STORAGE_QUOTA' ? 'storage' : navigator.onLine ? 'failed' : 'offline', error: err } });
  }
}

export const stopDownload = () => engine.cancel();
export const showDownload = (open: boolean) => { const dl = get().dl; if (dl) set({ dl: { ...dl, open } }); };
export const closeDownload = () => set({ dl: undefined });

function emptySteps(mode: Options['mode']): Record<Step, StepInfo> {
  const s = Object.fromEntries(STEPS.map((k) => [k, { state: 'wait', done: 0, total: 0 }])) as Record<Step, StepInfo>;
  if (mode === 'voice') { s.separate.state = 'skipped'; s.mix.state = 'skipped'; }
  return s;
}

const onUnload = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ''; };

async function run(rerun: boolean, profile: 'normal' | 'low') {
  const job = buildJob(profile);
  const s = get();
  if (!job || !s.song || !s.model?.info) return;
  const seconds = job.range ? job.range.end - job.range.start : s.song.duration ?? 0;
  const snapshot = { ...s.opts };
  const steps = emptySteps(job.mode);
  const st: Job = { status: 'running', rerun, mode: job.mode, seconds, startedAt: performance.now(), steps, change: rerun && s.result ? { from: s.result.opts, to: snapshot } : undefined };
  set({ job: st, stems: rerun ? s.stems : undefined });
  setRunningFlag(true);
  addEventListener('beforeunload', onUnload); // EX-5

  const now = () => performance.now();
  const patchStep = (k: Step, p: Partial<StepInfo>) => {
    const j = get().job;
    if (!j) return;
    set({ job: { ...j, current: k, steps: { ...j.steps, [k]: { ...j.steps[k], ...p } } } });
  };
  const finishOthers = (k: Step) => {
    const j = get().job;
    if (!j) return;
    for (const o of STEPS) {
      if (o === k) break;
      const x = j.steps[o];
      if (x.state === 'run' || x.state === 'load') patchStep(o, { state: 'done', ms: now() - (x.startedAt ?? now()) });
    }
  };
  let stemsAsked = false;
  const onProgress = (p: Progress) => {
    if (p.stage === 'download') return;
    const k = p.stage as RunStage;
    if (!get().job) return; // late event after cancel
    finishOthers(k);
    const cur = get().job!.steps[k];
    if (p.reused) patchStep(k, { state: 'reused' });
    else if (p.phase === 'load') patchStep(k, { state: 'load', startedAt: cur.startedAt ?? now() });
    else {
      const done = p.total > 0 && p.done >= p.total;
      patchStep(k, { state: done ? 'done' : 'run', done: p.done, total: p.total, startedAt: cur.startedAt ?? now(), ms: done ? now() - (cur.startedAt ?? now()) : undefined });
    }
    const j = get().job;
    if (j && p.backend === 'wasm' && get().env?.backend === 'webgpu' && !j.fallback) set({ job: { ...j, fallback: true } });
    // DESIGN S3: separated stems are playable while conversion continues.
    if (job.mode === 'song' && k !== 'separate' && !stemsAsked) {
      stemsAsked = true;
      void engine.getStems().then((x) => { if (x && get().job) set({ stems: { vocals: toAudioBuffer(x.vocals), inst: toAudioBuffer(x.instrumental) } }); });
    }
  };

  try {
    const r = await engine.run(job, onProgress);
    finishOthers('mix');
    if (job.mode === 'song') patchStep('mix', { state: 'done', ms: 0 });
    const prevResult = rerun ? get().result : undefined;
    const result: Result = {
      songName: stripExt(s.song.name), modelName: s.model.info.name, mode: job.mode, opts: snapshot, hasIndex: !!job.indexId,
      converted: toAudioBuffer(r.converted),
      vocals: r.vocals && toAudioBuffer(r.vocals), inst: r.instrumental && toAudioBuffer(r.instrumental),
      ms: now() - st.startedAt, seq: (get().result?.seq ?? 0) + 1,
    };
    set({ job: undefined, result, prev: prevResult, ab: 'B', view: 'result', stems: result.vocals && result.inst ? { vocals: result.vocals, inst: result.inst } : get().stems });
  } catch (e) {
    const err = toAppError(e);
    const j = get().job;
    if (err.code === 'CANCELLED') {
      set({ job: undefined });
      const kept = j && (j.steps.separate.state === 'done' || j.steps.separate.state === 'reused');
      toast(kept ? t('toast.cancelledKept') : t('toast.cancelled'), { action: { label: t('toast.restart'), onClick: () => void start({ rerun }) } });
    } else if (j) {
      const k = j.current ?? 'separate';
      set({ job: { ...j, status: 'error', error: err, steps: { ...j.steps, [k]: { ...j.steps[k], state: 'error' } } } });
    }
  } finally {
    setRunningFlag(false);
    removeEventListener('beforeunload', onUnload);
  }
}

export async function cancelJob() {
  const j = get().job;
  if (!j) return;
  if (j.status === 'error') return set({ job: undefined });
  set({ job: { ...j, status: 'cancelling' } });
  await engine.cancel();
  // run() resolves the state when the worker stops at the next chunk.
  for (let i = 0; i < 100 && get().job; i++) await new Promise((r) => setTimeout(r, 50));
}

/** Recovery actions offered by the progress panel error (DESIGN 5, EX-8). */
export async function recover(action: 'retry' | 'low' | 'wasm' | 'noIndex') {
  const rerun = get().job?.rerun ?? false;
  set({ job: undefined });
  if (action === 'noIndex') clearIndex();
  if (action === 'wasm') {
    set({ settings: { ...get().settings, backend: 'wasm' } });
    toast(t('toast.wasmKept'), { action: { label: t('settings.title'), onClick: () => openSettings('general') } });
    await initEngine();
  }
  await start({ rerun, profile: action === 'low' ? 'low' : undefined });
}

// ---------- results / export ----------

export const backToWork = () => set({ view: 'work' });

async function to44k(b: AudioBuffer) {
  if (b.sampleRate === 44100) return b;
  const ctx = new OfflineAudioContext(b.numberOfChannels, Math.ceil(b.duration * 44100), 44100);
  const src = ctx.createBufferSource();
  src.buffer = b;
  src.connect(ctx.destination);
  src.start();
  return ctx.startRendering();
}

export type StemKind = 'mix' | 'converted' | 'vocals' | 'inst';

export function fileName(r: Pick<Result, 'songName' | 'modelName' | 'mode'>, kind: StemKind) {
  const ext = get().format === 'mp3' ? 'mp3' : 'wav';
  const song = safe(r.songName), model = safe(r.modelName);
  const base = {
    mix: `${song} (${model} ${t('file.cover')})`,
    converted: r.mode === 'voice' ? `${song} (${model})` : `${song}_vocals_${model}`,
    vocals: `${song}_vocals_original`,
    inst: `${song}_instrumental`,
  }[kind];
  return `${base}.${ext}`;
}

export const currentResult = () => { const s = get(); return s.ab === 'A' && s.prev ? s.prev : s.result; };

export async function exportStem(kind: StemKind, r = currentResult()) {
  const s = get();
  const stems = { vocals: r?.vocals ?? s.stems?.vocals, inst: r?.inst ?? s.stems?.inst };
  try {
    let buf: AudioBuffer | undefined;
    if (kind === 'mix' && r) buf = await renderMix({ vocal: r.converted, inst: r.inst }, s.mix);
    else if (kind === 'converted' && r) buf = await to44k(r.converted); // PRD 7.4: output is 44.1 kHz
    else buf = kind === 'vocals' ? stems.vocals : stems.inst;
    if (!buf) return;
    const blob = await encodeAudio(buf, s.format);
    const name = fileName(r ?? { songName: stripExt(s.song?.name ?? 'song'), modelName: s.model?.info?.name ?? '', mode: 'song' }, kind);
    download(blob, name);
    toast.success(t('toast.saved', { name }));
  } catch (e) {
    toast.error(t('err.INTERNAL.title'), { description: toAppError(e).message }); // render/encode run here, not in the engine
  }
}

/** OOM recovery and progress panel: save the separated stems even when a later stage failed. */
export async function exportSeparated() {
  let st = get().stems;
  if (!st) {
    const x: { vocals: Stereo; instrumental: Stereo } | null = await engine.getStems();
    if (!x) return;
    st = { vocals: toAudioBuffer(x.vocals), inst: toAudioBuffer(x.instrumental) };
    set({ stems: st });
  }
  await exportStem('vocals');
  await exportStem('inst');
}

// ---------- storage ----------

export async function clearCache() {
  const ok = await confirmAsync({ title: t('confirm.clearTitle'), body: t('confirm.clearBody'), action: t('confirm.clearAction'), destructive: true });
  if (!ok) return false;
  try { await engine.cache.clear(); } catch (e) { toast.error(t('toast.deleteFailed'), reason(e)); }
  void refreshLists();
  void refreshPlan();
  return true;
}

/** Re-hashes one cache item now; 'corrupt' files are deleted by the engine. */
export async function verifyCacheItem(id: string) {
  try {
    const st = await engine.cache.verify(id);
    if (st === 'verified') toast.success(t('toast.verifyOk'));
    else toast.error(t(st === 'corrupt' ? 'toast.verifyCorrupt' : 'toast.verifyIncomplete'));
  } catch (e) {
    toast.error(t('toast.verifyFailed'), reason(e));
  }
  void refreshLists();
  void refreshPlan();
}

/** PRD R10: fill the public cache from model files the user downloaded elsewhere. */
export async function importCache(files: File[]) {
  try {
    const res = await engine.cache.importFiles(files);
    const n = (k: string) => res.filter((r) => r.status === k).length;
    const parts = [t('toast.importDone', { n: n('imported') })];
    if (n('mismatch')) parts.push(t('toast.importMismatch', { n: n('mismatch') }));
    if (n('unknown')) parts.push(t('toast.importUnknown', { n: n('unknown') }));
    (n('imported') ? toast.success : toast.error)(parts.join(' · '));
  } catch (e) {
    toast.error(t('toast.importFailed'), reason(e));
  }
  void refreshLists();
  void refreshPlan();
}

export async function renameMine(id: string, name: string) {
  try { await engine.myModels.rename(id, name); } catch (e) { return void toast.error(t('toast.renameFailed'), reason(e)); }
  const m = get().model;
  if (m?.info?.id === id) set({ model: { ...m, info: { ...m.info, name } } });
  void refreshLists();
}

export async function removeCacheItem(id: string) {
  try { await engine.cache.remove(id); } catch (e) { toast.error(t('toast.deleteFailed'), reason(e)); }
  void refreshLists();
  void refreshPlan();
}

