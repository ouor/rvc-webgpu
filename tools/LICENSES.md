# Model sources

Files in this repository are converted for the web-rvc browser app (graph `.onnx` without weights,
weights as fp16 `.pth` state_dict, `.manifest.json` layout). Numerical values are unchanged
(the original checkpoints are already fp16).

| Path | Origin | Code license | Weights |
|---|---|---|---|
| `contentvec/` | `hubert_base.pt` distributed with RVC WebUI (HuBERT base / ContentVec-compatible) | fairseq MIT, torchaudio BSD-2 (model code) | check upstream terms |
| `rmvpe/` | `rmvpe.pt` distributed with RVC WebUI (RMVPE, Wei et al. 2023) | MIT | check upstream terms |
| `uvr/` | `uvr-vocal.pth`: UVR VR-architecture model (HP family, 4band_v2) from RVC WebUI `uvr5_weights` | MIT (RVC WebUI 2.2.231006 `lib_v5`) | redistribution terms not stated upstream |
| `rvc/` | Weightless synthesizer graphs generated from RVC (rvc-nano infer_pack) | MIT | none (user models are loaded locally) |

The repository owner decides whether these weights may be redistributed (PRD R11).
