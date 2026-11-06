# tools: model files for web-rvc

Builds the files the browser downloads (graphs, weights, manifests, `registry.json`), the reference
outputs that the TypeScript engine is compared against, and uploads the tree to Hugging Face.
Contract with the app: `docs/dev/architecture/ARCHITECTURE.md` chapter 17.

## Setup (Windows, Python 3.11)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r tools/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

If torch 2.8 is already installed system-wide, `python -m venv --system-site-packages .venv` avoids a
second torch download; the venv still pins torchaudio 2.8 (must match torch), onnx and onnxruntime.

Input weights go in `external/weight/`: `hubert_base.pt`, `rmvpe.pt`, `uvr-vocal.pth`, and for the
reference run `rvc-kyubey.pth` (+ `rvc-kyubey.index`).

## Build the model tree (`weights/dist/`)

Run from the `tools/` directory (the scripts import each other):

```
cd tools
..\.venv\Scripts\python export_rvc_templates.py   # rvc/{v1,v2}-{f0,nof0}-{32k,40k,48k}.{onnx,manifest.json}
..\.venv\Scripts\python export_hubert.py          # contentvec/contentvec.{onnx,pth,manifest.json}
..\.venv\Scripts\python export_rmvpe.py           # rmvpe/rmvpe.{onnx,pth,manifest.json}
..\.venv\Scripts\python export_uvr.py             # uvr/uvr-hp-vocal.{onnx,pth,manifest.json}, uvr/4band_v2.json
..\.venv\Scripts\python publish.py                # registry.json (sha256 + size of every file), LICENSES.md
```

The dev server serves `weights/dist` as the model base URL (see `vite.config.ts`), so this is all the
app needs locally.

## Check

```
..\.venv\Scripts\python reference.py      # weights/reference/: separation + conversion stages of the test song (~1 min on GPU)
..\.venv\Scripts\python verify_onnx.py    # torch vs onnxruntime for every graph, writes weights/verify.json
..\.venv\Scripts\python make_fixtures.py  # tests/fixtures/<case>/ for vitest (needs reference.py output)
```

`verify_onnx.py` also checks that the public `.pth` files pass the browser's allowlist reader
(`docs/dev/architecture/spikes/pth-reader.mjs`, needs Node).

## Upload to Hugging Face

1. `hf auth login` (write token; the `hf` CLI ships with huggingface_hub), or set `HF_TOKEN`.
2. Read `weights/dist/LICENSES.md` and decide whether you may redistribute the weights (the repo is
   created **private** unless you pass `--public`).
3. Upload:

   ```
   ..\.venv\Scripts\python publish.py --upload <user>/web-rvc-models --app-registry
   ```

   This uploads every file, then `registry.json` with that commit as `revision`, copies the registry to
   `src/models/registry.gen.json`, and prints the value for the app build:
   `VITE_MODEL_BASE_URL=https://huggingface.co/<user>/web-rvc-models/resolve/<commit>/`.

   The app needs anonymous download (CORS from HF works for public repos; a private repo is only
   reachable with a token, so make it public when you are ready to deploy).

Manual alternative: upload the contents of `weights/dist/` as-is (same relative paths) with the HF web
UI or `hf upload <user>/web-rvc-models weights/dist .`, then set the base URL to the
resulting commit. Cloudflare R2 works the same way: copy the tree to `<bucket>/<revision>/`.

## Files

| File | Role |
|---|---|
| `common.py` | template export (weights moved to external `weights.bin`, manifest), loader twin, signature, fixture writer |
| `export_*.py` | one per model; patches needed for export are at the top of each script |
| `reference.py` | Python reference pipeline (RVC 2.2 `AudioPre` + rvc-nano), polyphase resampling, saved rnd/noise |
| `verify_onnx.py` | numeric checks (ARCHITECTURE 12 criteria) |
| `make_fixtures.py` | small vitest fixtures (ARCHITECTURE 17.6) |
| `publish.py` | `registry.json`, Hugging Face upload |
| `vendor/` | unmodified copies of rvc-nano and RVC WebUI 2.2.231006 model code |
