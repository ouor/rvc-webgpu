"""Python reference pipeline (ARCHITECTURE 11, 12): UVR VR separation -> rvc-nano conversion, stage by stage.

Usage: python tools/reference.py [--audio external/audio-bunnygirl.mp3] [--model rvc-kyubey] [--seed 0]
Writes 17.6 fixture cases under weights/reference/:
  <song>-separate/   input44k -> vocals44k, inst44k                     (AudioPre flow, global coef, mirroring)
  <song>-<model>/    vocal16k + rnd/noise -> f0, HuBERT feats, faiss search, converted audio
Torch fp32 everywhere (TF32 off). Deviations from the originals, same as the JS engine will do:
  * every resample is scipy.signal.resample_poly (VR synthesis used sinc_fastest/scipy FFT, rvc-nano ffmpeg)
  * rnd / noise come from numpy default_rng(seed), saved so the TS side can inject the same arrays
  * librosa >= 0.10 changed stft/rms default pad_mode to 'constant'; we pass 'reflect' = librosa 0.9.1
    (the version RVC 2.2.231006 and rvc-nano pinned)
"""
import argparse
import json
import math
import os

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from common import REFERENCE, VENDOR, WEIGHTS_IN, ROOT, write_case
from export_hubert import load_contentvec
from export_rvc_templates import build_net, make_template
from export_uvr import PARAMS_JSON, load_uvr
from rmvpe import RMVPE
from uvr import spec_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STFT_PAD = "reflect"
SOURCE = "tools/reference.py (RVC WebUI 2.2.231006 AudioPre, rvc-nano@dda674b pipeline, librosa %s)" % librosa.__version__


# ---------------------------------------------------------------- VR separation (AudioPre._path_audio_)
class BandParams:  # = lib_v5 ModelParameters for a json file (int band keys, missing flags False)
    def __init__(self, path=PARAMS_JSON):
        self.param = json.load(open(path), object_pairs_hook=lambda kv: {int(k) if k.isdigit() else k: v for k, v in kv})
        for k in ("mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"):
            self.param.setdefault(k, False)
        assert not any(self.param[k] for k in ("mid_side", "mid_side_b2", "reverse"))


def resample(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    g = math.gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=-1)


def stft2(wave, n_fft, hl):  # [2, N] -> [2, n_fft/2+1, frames]
    return np.stack([librosa.stft(np.asfortranarray(c), n_fft=n_fft, hop_length=hl, pad_mode=STFT_PAD) for c in wave])


def istft2(spec, hl):
    return np.stack([librosa.istft(np.asfortranarray(c), hop_length=hl) for c in spec])


def vr_analyze(wave44, mp):
    """Band STFTs -> combined complex spectrogram [2, 673, F] and the kept high end (mirroring)."""
    bands = mp.param["band"]
    n = len(bands)
    waves, specs = {}, {}
    for d in range(n, 0, -1):
        bp = bands[d]
        waves[d] = wave44 if d == n else resample(waves[d + 1], bands[d + 1]["sr"], bp["sr"])
        specs[d] = stft2(waves[d], bp["n_fft"], bp["hl"])
    bp = bands[n]
    high_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (mp.param["pre_filter_stop"] - mp.param["pre_filter_start"])
    high_end = specs[n][:, bp["n_fft"] // 2 - high_h: bp["n_fft"] // 2, :]
    return spec_utils.combine_spectrograms(specs, mp), high_h, high_end


def vr_mask(net, x_mag, batch=4):
    """utils.inference(): global coef, constant pad (offset 128, roi 256), 512-frame patches -> mask [2,673,F]."""
    coef = x_mag.max()
    n = x_mag.shape[2]
    pad_l, roi = net.offset, 512 - 2 * net.offset
    pad_r = roi - (n % roi) + pad_l
    xp = np.pad(x_mag / coef, ((0, 0), (0, 0), (pad_l, pad_r)))
    patches = np.stack([xp[:, :, i * roi: i * roi + 512] for i in range(int(np.ceil(n / roi)))]).astype(np.float32)
    with torch.no_grad():
        out = [net(torch.from_numpy(patches[i:i + batch]).to(DEV)).cpu().numpy() for i in range(0, len(patches), batch)]
    mask = np.concatenate(np.concatenate(out), axis=2)[:, :, :n]  # [B,2,673,256] -> [2,673,B*256]
    return mask, float(coef), patches


def apply_aggressiveness(mask, agg, split_bin):
    mask = mask.copy()
    v = agg / 100
    mask[:, :split_bin] **= 1 + v / 3
    mask[:, split_bin:] **= 1 + v
    return mask


def vr_synthesize(spec_m, mp, high_h, high_end):
    """cmb_spectrogram_to_wave with mirroring and polyphase resampling."""
    bands = mp.param["band"]
    n = len(bands)
    extra = spec_utils.mirroring("mirroring", spec_m, high_end, mp)
    offset, wave = 0, None
    for d in range(1, n + 1):
        bp = bands[d]
        s = np.zeros((2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        s[:, bp["crop_start"]: bp["crop_stop"]] = spec_m[:, offset: offset + h]
        offset += h
        if d == n:
            s[:, bp["n_fft"] // 2 - high_h: bp["n_fft"] // 2] = extra[:, :high_h]
            s = spec_utils.fft_hp_filter(s, bp["hpf_start"], bp["hpf_stop"] - 1)
            wave = wave + istft2(s, bp["hl"])
        elif d == 1:
            s = spec_utils.fft_lp_filter(s, bp["lpf_start"], bp["lpf_stop"])
            wave = resample(istft2(s, bp["hl"]), bp["sr"], bands[d + 1]["sr"])
        else:
            s = spec_utils.fft_hp_filter(s, bp["hpf_start"], bp["hpf_stop"] - 1)
            s = spec_utils.fft_lp_filter(s, bp["lpf_start"], bp["lpf_stop"])
            wave = resample(wave + istft2(s, bp["hl"]), bp["sr"], bands[d + 1]["sr"])
    return wave.astype(np.float32)


def separate(wave44, agg=10):
    mp = BandParams()
    X, high_h, high_end = vr_analyze(wave44, mp)
    mask, coef, _ = vr_mask(load_uvr().to(DEV), np.abs(X))
    mask = apply_aggressiveness(mask, agg, mp.param["band"][1]["crop_stop"])
    y = mask * np.abs(X) * np.exp(1j * np.angle(X))  # instrumental
    v = X - y
    return vr_synthesize(v, mp, high_h, high_end), vr_synthesize(y, mp, high_h, high_end), coef


# ---------------------------------------------------------------- RVC (rvc-nano Pipeline.pipeline, fp32)
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 6, 38, 41  # rvc-nano Config with is_half=False
SR, WINDOW = 16000, 160
HPF_B, HPF_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def split_points(audio):
    """opt_ts: cut points near every t_center where the 160-sample moving |x| sum is minimal."""
    t_query, t_center, t_max = SR * X_QUERY, SR * X_CENTER, SR * X_MAX
    audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
    opt_ts = []
    if audio_pad.shape[0] > t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(WINDOW):
            audio_sum += np.abs(audio_pad[i: i - WINDOW])
        for t in range(t_center, audio.shape[0], t_center):
            seg = audio_sum[t - t_query: t + t_query]
            opt_ts.append(t - t_query + int(np.where(seg == seg.min())[0][0]))
    return opt_ts


def quantize_f0(f0, up_key):
    f0 = f0 * pow(2, up_key / 12)
    mel_min, mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64), f0.astype(np.float32)


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2, pad_mode=STFT_PAD)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2, pad_mode=STFT_PAD)
    rms1 = F.interpolate(torch.from_numpy(rms1).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = F.interpolate(torch.from_numpy(rms2).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.clamp(rms2, min=1e-6)
    return data2 * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()


class Rvc:
    def __init__(self, model_path, index_path=None):
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.version, self.f0 = cpt.get("version", "v1"), int(cpt.get("f0", 1))
        self.sr = cpt["config"][-1]
        assert self.f0, "reference covers f0 models only"
        net = build_net(self.version, self.f0, cpt["config"], cpt["weight"])
        self.upp = math.prod(cpt["config"][12])
        self.emb_g = net.emb_g.weight.detach()
        self.tpl = make_template(net, self.f0).to(DEV)
        self.index = faiss.read_index(index_path) if index_path else None
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal) if self.index else None

    def search(self, feats):  # [F, C] -> blended [F, C], score, ix (rvc-nano vc(), k = 8)
        score, ix = self.index.search(feats, k=8)
        w = np.square(1 / score)
        w /= w.sum(axis=1, keepdims=True)
        return np.sum(self.big_npy[ix] * w[:, :, None], axis=1).astype(np.float32), score, ix

    def synth(self, feats, feats0, pitch, pitchf, rnd, noise, sid, protect):
        """feats/feats0: [F, C] at 50 fps (after/before index). Returns audio at model SR."""
        up = lambda f: torch.from_numpy(np.repeat(f, 2, axis=0))[None]  # F.interpolate(scale 2, nearest)
        x, x0 = up(feats), up(feats0)
        T = min(x.shape[1], pitch.shape[0])
        x, x0, pitch, pitchf = x[:, :T], x0[:, :T], pitch[:T], pitchf[:T]
        pf = torch.from_numpy(pitchf)[None]
        if protect < 0.5:
            w = torch.where(pf > 0, 1.0, protect)  # pitchf > 0 -> 1, pitchf < 1 -> protect (pitchf is 0 or >= 50)
            x = x * w[..., None] + x0 * (1 - w[..., None])
        args = [x, torch.tensor([T]), torch.from_numpy(pitch)[None], pf, self.emb_g[sid][None, :, None],
                torch.from_numpy(rnd[:, :T])[None], torch.from_numpy(noise[: T * self.upp])[None, :, None]]
        with torch.no_grad():
            return self.tpl(*[a.to(DEV) for a in args])[0, 0].cpu().numpy(), T


def convert(vocal16k, rvc, hubert, rmvpe, *, up_key=0, index_rates=(0.75, 0.0), protect=0.33, rms_mix_rate=0.25, sid=0, seed=0):
    t_pad, t_pad_tgt = SR * X_PAD, rvc.sr * X_PAD
    audio = signal.filtfilt(HPF_B, HPF_A, vocal16k)
    opt_ts = split_points(audio)
    audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // WINDOW

    with torch.no_grad():
        mel = rmvpe.mel_extractor(torch.from_numpy(audio_pad).float().to(DEV)[None], center=True)
        hidden = rmvpe.mel2hidden(mel)[0].cpu().numpy()
    f0 = rmvpe.decode(hidden, thred=0.03)
    pitch, pitchf = quantize_f0(f0, up_key)
    pitch, pitchf = pitch[:p_len], pitchf[:p_len]

    # chunk bounds in samples of audio_pad, exactly as the rvc-nano loop
    bounds, s = [], 0
    for t in opt_ts:
        t = t // WINDOW * WINDOW
        bounds.append((s, t + 2 * t_pad + WINDOW, s // WINDOW, (t + 2 * t_pad) // WINDOW))
        s = t
    last = opt_ts[-1] // WINDOW * WINDOW if opt_ts else 0  # rvc-nano: audio_pad[t:] after the loop
    bounds.append((last, audio_pad.shape[0], last // WINDOW, p_len))

    rng = np.random.default_rng(seed)
    out = {"chunks": [], "feats": [], "rnd": [], "noise": []}
    per_rate = {r: {"audio": [], "feats": [], "score": [], "ix": []} for r in index_rates}
    for a0, a1, f0a, f0b in bounds:
        chunk = audio_pad[a0:a1].astype(np.float32)
        with torch.no_grad():
            f768, f256 = hubert(torch.from_numpy(chunk)[None].to(DEV))
        feats = (f768 if rvc.version == "v2" else f256)[0].cpu().numpy()
        T = min(chunk.shape[0] // WINDOW, feats.shape[0] * 2)
        rnd = rng.standard_normal((192, T), dtype=np.float32)
        noise = rng.standard_normal(T * rvc.upp, dtype=np.float32)
        out["chunks"].append({"audio": [a0, a1], "f0Frames": [f0a, f0b], "featFrames": feats.shape[0], "T": T})
        out["feats"].append(feats), out["rnd"].append(rnd), out["noise"].append(noise)
        for r in index_rates:
            blended, score, ix = feats, np.zeros((feats.shape[0], 8), np.float32), np.zeros((feats.shape[0], 8), np.int64)
            if rvc.index is not None and r != 0:
                b, score, ix = rvc.search(feats)
                blended = (b * r + (1 - r) * feats).astype(np.float32)
            y, t_used = rvc.synth(blended, feats, pitch[f0a:f0b], pitchf[f0a:f0b], rnd, noise, sid, protect)
            assert t_used == T
            per_rate[r]["audio"].append(y[t_pad_tgt:-t_pad_tgt])
            per_rate[r]["feats"].append(blended), per_rate[r]["score"].append(score), per_rate[r]["ix"].append(ix)

    for r in index_rates:
        y = np.concatenate(per_rate[r]["audio"])
        per_rate[r]["raw"] = y
        if rms_mix_rate != 1:
            y = change_rms(audio, SR, y, rvc.sr, rms_mix_rate)
        peak = np.abs(y).max() / 0.99
        per_rate[r]["converted"] = (y / peak if peak > 1 else y).astype(np.float32)
    return dict(audio_pad=audio_pad.astype(np.float32), hidden=hidden.astype(np.float32), f0=f0.astype(np.float32),
                pitch=pitch, pitchf=pitchf, opt_ts=opt_ts, **out, per_rate=per_rate)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=os.path.join(ROOT, "external", "audio-bunnygirl.mp3"))
    ap.add_argument("--model", default="rvc-kyubey", help="name of <external/weight>/<name>.pth (+ .index)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agg", type=int, default=10)
    a = ap.parse_args()
    song = os.path.splitext(os.path.basename(a.audio))[0].replace("audio-", "")

    wave44, _ = librosa.load(a.audio, sr=44100, mono=False, dtype=np.float32)
    if wave44.ndim == 1:
        wave44 = np.stack([wave44, wave44])
    vocals, inst, coef = separate(wave44, a.agg)
    mp = BandParams()
    write_case(os.path.join(REFERENCE, f"{song}-separate"), SOURCE, {"input44k": wave44},
               {"vocals44k": vocals, "inst44k": inst},
               {"sr": 44100, "agg": a.agg, "split_bin": mp.param["band"][1]["crop_stop"], "coef": coef,
                "stftPadMode": STFT_PAD, "resample": "scipy.signal.resample_poly(window=('kaiser', 5.0))",
                "highEndProcess": "mirroring", "batch": 4, "decoder": "librosa.load (soundfile/mpg123)",
                "note": "outputs are 480*(frames-1) samples, slightly shorter/longer than input; align at 0"})
    print(f"separate: {wave44.shape} -> {vocals.shape}, coef {coef:.3f}")

    # rvc-nano RVCInference.infer(): mono 16k, peak 0.95
    vocal16k = resample(vocals.mean(axis=0), 44100, 16000).astype(np.float32)
    m = np.abs(vocal16k).max() / 0.95
    if m > 1:
        vocal16k /= m
    model_path = os.path.join(WEIGHTS_IN, a.model + ".pth")
    index_path = os.path.join(WEIGHTS_IN, a.model + ".index")
    rvc = Rvc(model_path, index_path if os.path.exists(index_path) else None)
    hubert = load_contentvec().to(DEV)
    rmvpe = RMVPE(os.path.join(WEIGHTS_IN, "rmvpe.pt"), is_half=False, device=DEV)
    params = dict(up_key=0, index_rates=(0.75, 0.0), protect=0.33, rms_mix_rate=0.25, sid=0, seed=a.seed)
    r = convert(vocal16k, rvc, hubert, rmvpe, **params)

    cat = np.concatenate
    outputs = {"audioPad16k": r["audio_pad"], "rmvpeHidden": r["hidden"], "f0Hz": r["f0"], "pitch": r["pitch"],
               "pitchf": r["pitchf"], "feats": cat(r["feats"])}
    for rate, v in r["per_rate"].items():
        tag = f"Ir{int(rate * 100)}"
        if rate != 0:
            outputs.update({f"index{tag}Score": cat(v["score"]), f"index{tag}Ix": cat(v["ix"]), f"feats{tag}": cat(v["feats"])})
        outputs.update({f"raw{tag}": v["raw"], f"converted{tag}": v["converted"]})
    write_case(os.path.join(REFERENCE, f"{song}-{a.model}"), SOURCE,
               {"vocal16k": vocal16k, "rnd": cat(r["rnd"], axis=1), "noise": cat(r["noise"])}, outputs,
               {"model": a.model + ".pth", "index": a.model + ".index" if rvc.index else None, "modelSr": rvc.sr,
                "version": rvc.version, "upp": rvc.upp, **{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
                "xPad": X_PAD, "xQuery": X_QUERY, "xCenter": X_CENTER, "xMax": X_MAX, "optTs": [int(t) for t in r["opt_ts"]],
                "chunks": r["chunks"], "hpf": {"b": HPF_B.tolist(), "a": HPF_A.tolist()}, "rmsPadMode": STFT_PAD,
                "nprobe": int(faiss.extract_index_ivf(rvc.index).nprobe) if rvc.index else None,
                "vocal16kFrom": "mean(vocals44k) -> resample_poly(160, 441) -> /max(1, peak/0.95)",
                "rnd/noise": "numpy default_rng(seed): per chunk rnd [192,T] then noise [T*upp]; concatenated",
                "featsLayout": "HuBERT 50fps per chunk, concatenated in chunk order (chunks[i].featFrames rows)",
                "rawLayout": "per chunk synth output with t_pad_tgt trimmed, concatenated (before change_rms)"})
    print("convert:", {k: v.shape for k, v in outputs.items()})


if __name__ == "__main__":
    main()
