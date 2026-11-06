# Engine

The engine turns a song and an RVC voice model into a cover, entirely in the browser. It runs in a Web Worker (onnxruntime-web, WebGPU with a WASM fallback) and is used through `createEngineClient()`. Types are in `src/shared/types.ts` and errors in `src/shared/errors.ts`. The audio helpers in `src/audio` run on the main thread.

## Minimal example: one song to an MP3 cover

```ts
import { createEngineClient } from '@/app/engine-client';
import { decodeFile, toAudioBuffer } from '@/audio/decode';
import { createExporter } from '@/audio/export-client';
import { DEFAULT_MIX, renderMix } from '@/audio/mixer';
import { DEFAULT_CONVERT, DEFAULT_SEPARATE, type JobSpec } from '@/shared/types';

export async function makeCover(song: File, model: File, index?: File): Promise<Blob> {
  const engine = createEngineClient();
  await engine.init({ backend: 'auto', modelBaseUrl: import.meta.env.VITE_MODEL_BASE_URL ?? '/models/' });

  const dec = await decodeFile(song); // 44.1 kHz stereo
  const inputId = await engine.setInput(dec.audio, { name: song.name, size: song.size, lastModified: song.lastModified });
  const rvc = await engine.loadRvcModel(model); // .pth, .onnx or .zip
  const idx = index ? await engine.loadIndex(index) : rvc.index;

  const job: JobSpec = {
    inputId, mode: 'song', separate: DEFAULT_SEPARATE,
    modelId: rvc.id, indexId: idx?.id, convert: { ...DEFAULT_CONVERT, f0UpKey: 0 },
  };
  const missing = (await engine.plan(job)).filter((d) => !d.cached);
  if (missing.length) await engine.download(missing.map((d) => d.id), (p) => console.log('download', p.done, p.total));

  const r = await engine.run(job, (p) => console.log(p.stage, p.done, p.total));
  const mix = await renderMix({ vocal: toAudioBuffer(r.converted), inst: r.instrumental && toAudioBuffer(r.instrumental) }, DEFAULT_MIX);
  const mp3 = await createExporter().encode(mix, 'mp3');
  engine.terminate();
  return mp3;
}
```

## Setup

```ts
const engine = createEngineClient();
const env = await engine.init({ backend: 'auto', modelBaseUrl: '/models/' });
```

- `backend`: `'auto'` uses WebGPU when an adapter exists, otherwise WASM. You can also pass `'webgpu'` or `'wasm'` to force one. `env.backend` reports the result, along with `gpuAdapter`, `crossOriginIsolated`, `wasmThreads` and `storage`.
- `modelBaseUrl`: the folder that holds `registry.json` and the shared model files. It may be relative. The dev server serves `weights/dist/` at `/models/`; the published set is at `https://huggingface.co/trolley/rvc-webgpu/resolve/<revision>/`.
- Multithreaded WASM needs cross-origin isolation. In production, call `ensureIsolation()` from `@/coi` before creating the engine. It registers a service worker that adds COOP/COEP and reloads once.
- Use one engine per page. `engine.terminate()` stops the worker.

## Shared models

The engine needs ContentVec, RMVPE (f0 models), the UVR separator (song mode only) and the RVC synthesizer template that matches the model. `run()` never downloads them.

```ts
const plan = await engine.plan(job);             // [{ id, label, bytes, cached }]
await engine.download(plan.filter((d) => !d.cached).map((d) => d.id), onProgress);
```

Downloads resume after a dropped connection, are checked with SHA-256 and are kept in OPFS. Download progress: `done` and `total` are bytes over all files; `item`, `itemDone` and `itemTotal` give the same per plan item; `phase: 'verify'` marks the hash check. `cancel()` stops a download.

## Input

| Mode | Call | What runs |
|---|---|---|
| `'song'` | `setInput(audio, meta)` | separate, f0, features, convert |
| `'voice'` | `setInput(audio, meta)` | f0, features, convert (input is a vocal) |
| `'stems'` | `setStemsInput(vocals, instrumental, meta)` | f0, features, convert (separation is skipped) |

- `audio` is `Stereo` at 44.1 kHz; get it with `decodeFile(file).audio`.
- The sample buffers are transferred to the worker and become unusable on your side. Keep `dec.buffer` for playback.
- The engine keeps one input at a time. Input must be between 1 s and 20 min long.
- With `setStemsInput`, the shorter stem is padded with silence.
- Set `job.range = { start, end }` (in seconds) to process only part of the input.

## Voice model and index

```ts
const model = await engine.loadRvcModel(file);   // .pth (RVC v1/v2), .onnx, or .zip with .pth/.onnx (+ .index)
if (model.srUnknown) await engine.setOnnxModelSr(model.id, 40000);
const index = await engine.loadIndex(indexFile); // faiss IVF-Flat / Flat .index
```

- `.pth` files are parsed with a pickle allowlist and never executed.
- A `.zip` brings its `added_*.index` along: `model.indexId` and `model.index`.
- Only the last loaded model and the last loaded index stay loaded.
- An index has to match the model version (v1: 256 dimensions, v2: 768).

## Run

```ts
const result = await engine.run(job, onProgress);
```

`JobSpec`: `{ inputId, mode, range?, separate, modelId, indexId?, convert }`.

| Option | Default | Range | Effect |
|---|---|---|---|
| `convert.f0UpKey` | 0 | -24..24 | pitch shift in semitones |
| `convert.indexRate` | 0.75 | 0..1 | share of index features (timbre vs. clarity) |
| `convert.protect` | 0.33 | 0..0.5 | protects breaths and consonants (0.5 = off) |
| `convert.rmsMixRate` | 0.25 | 0..1 | 1 = loudness of the converted voice, 0 = loudness of the input |
| `convert.sid` | 0 | 0..nSpk-1 | speaker id |
| `convert.chunkProfile` | `'normal'` | `'normal'` \| `'low'` | `'low'` uses shorter chunks and less memory |
| `convert.seed` | 0 | any | noise seed (same seed, same output) |
| `separate.aggressiveness` | 10 | 0..20 | how hard vocals are removed from the instrumental |

Defaults are exported as `DEFAULT_CONVERT` and `DEFAULT_SEPARATE`.

**Progress events:** `{ stage, done, total, phase?, backend?, reused?, skipped? }`, at most about 10 per second.

- Stages run in order: `separate` (patches), `f0`, `features`, `convert` (chunks).
- `phase: 'load'` means a model is being prepared, `phase: 'run'` that it is processing.
- `backend` is the backend actually used, after any fallback.

**Cancel:** `engine.cancel()` makes the pending `run()` or `download()` reject with `CANCELLED`, usually within about a second.

**Re-runs:** running again on the same input reuses earlier stages.

| What changes | What runs again |
|---|---|
| `convert` options (key, index rate, …) | only `convert` (`result.reused` lists `separate`, `f0`, `features`) |
| `separate.aggressiveness` or `range` | everything |
| the model | `f0` and `features` are reused when the version matches |

Only one `run()` or `download()` can be active; a second call throws `BUSY`. After an out-of-memory error, `run()` retries once with the `'low'` profile.

## Result

```ts
interface JobResult {
  converted: Mono;                           // converted vocal, model sample rate (32/40/48 kHz)
  vocals?: Stereo; instrumental?: Stereo;    // 44.1 kHz, song and stems mode
  reused: StageId[]; timings: Partial<Record<StageId, number>>; // ms
}
```

`engine.getStems()` returns the separated stems as soon as the `separate` stage finishes, so they can play while conversion continues.

## Mix and export

```ts
import { MixPlayer, renderMix, DEFAULT_MIX } from '@/audio/mixer';
import { conformStems } from '@/audio/export';
import { createExporter } from '@/audio/export-client';

const player = new MixPlayer(new AudioContext({ sampleRate: 44100 }));
player.load({ vocal: toAudioBuffer(r.converted), inst: r.instrumental && toAudioBuffer(r.instrumental) });
player.set({ ...DEFAULT_MIX, vocalGainDb: 0, instGainDb: -3, reverb: 0.2, muteInst: false, solo: undefined });
player.play(); player.pause(); player.seek(30);

const mix = await renderMix(stems, params);    // same graph offline, 44.1 kHz stereo, peak at -1 dBFS
```

- `MixParams` fields:
  - `vocalGainDb` and `instGainDb`
  - `reverb` (0..1, a 2.2 s room, `REVERB_SECONDS`)
  - `muteVocal` and `muteInst` (applied in playback and in the render)
  - `solo` (playback only)
- `renderMix` follows the same graph, so the export matches what you heard.

Stems for download are conformed first: every stem becomes 44.1 kHz stereo with the same length. One scale factor is applied to all of them only when a stem would clip in 16-bit or MP3.

```ts
const c = await conformStems({ converted: r.converted, vocals: r.vocals, instrumental: r.instrumental },
  { range: job.range, sourceSeconds: dec.duration, alignToSource: false }); // true: pad to the original timeline
const ex = createExporter();                      // encoding runs in its own worker
const wav = await ex.encode(c.stems.converted, 'pcm16', c.scale); // 'pcm16' | 'f32' | 'mp3' (320 kbps)
const zip = await ex.zip([
  { name: 'cover.mp3', audio: mix },
  { name: 'vocals-converted.mp3', audio: c.stems.converted, scale: c.scale },
  { name: 'settings.txt', text: 'key +2' },
], 'mp3', (done, total) => {});
```

## Saved models, recent jobs, cache

| API | Purpose |
|---|---|
| `myModels.save(modelId, indexId?)`, `list()`, `open(id)`, `rename(id, name)`, `setMemo(id, memo)`, `remove(id)` | Keep the user's models (and their index) in OPFS. `open` returns a model that is ready to run. |
| `recent.save({ songName, songSeconds, appVersion, mix })`, `list()`, `open(id)`, `updateMix(id, mix)`, `remove(id)`, `clear()` | Keep the last 3 results (16-bit WAV) on the device. `save` stores the last `run()` result and returns `{ id }`, or `{ skipped: 'quota' }` when space runs out. Restored jobs can be listened to, remixed and exported, but not re-run. |
| `cache.list()`, `verify(id)`, `importFiles(files)`, `remove(id)`, `clear()` | Shared models, saved models and recent jobs with their size. `verify` re-hashes every file. `importFiles` fills the cache from local copies of the model files. `clear` removes only the shared models. |

## Errors

Every rejection is an `AppError { code, message, detail?, recoverable }`; check it with `isAppError(e)` from `@/shared/errors`. `message` is meant for developers; show your own text keyed by `code`.

| Code | Meaning | What to do |
|---|---|---|
| `AUDIO_DECODE` | the browser cannot decode the file | use another file |
| `AUDIO_TOO_LONG` / `AUDIO_TOO_SHORT` | input over 20 min / under 1 s | trim, or pick a range |
| `PTH_NOT_ZIP` | legacy (non-zip) `torch.save` file | re-save with a current PyTorch |
| `PTH_CORRUPT` | damaged, cut off or not a model | download the file again |
| `PTH_UNSAFE` | pickle uses objects outside the allowlist | refuse the file |
| `PTH_TRAINING_CKPT` | a `G_`/`D_` training checkpoint | use the exported model `.pth` |
| `PTH_UNSUPPORTED_ARCH` / `_VOCODER` / `_EMBEDDER` | no matching synthesizer (e.g. RefineGAN) | export to ONNX |
| `ONNX_UNRECOGNIZED` | unknown ONNX inputs, or a fixed-length export | export with dynamic length |
| `INDEX_EMPTY` / `INDEX_UNSUPPORTED` / `INDEX_DIM_MISMATCH` | trained-only index, other faiss type, v1/v2 mismatch | run without the index |
| `ZIP_NO_MODEL` | zip without a `.pth` or `.onnx` | use another file |
| `MODELS_MISSING` | `run()` before the shared models were downloaded | `plan()` then `download()` |
| `NET_DOWNLOAD` | download failed after retries | retry |
| `CACHE_INTEGRITY` | a cached file failed its check and was deleted | download again |
| `STORAGE_QUOTA` | not enough storage | free space (`cache.clear()`) |
| `BACKEND_INIT` | no backend could create a session | use a WebGPU/WASM capable browser |
| `OOM` | out of memory | use a shorter range or `chunkProfile: 'low'` |
| `GPU_LOST` | the WebGPU device was lost | `init({ backend: 'wasm' })`, then retry |
| `BUSY` | a run or download is already active | wait, or `cancel()` |
| `CANCELLED` | after `cancel()` | none |
| `NOT_FOUND` | unknown input, model, index or saved id | load it again |
| `INTERNAL` | anything else (`detail` has the stack) | report it |
