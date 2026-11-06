"""Write the ARCHITECTURE 17.6 fixture cases to tests/fixtures/<case>/ (or --out).

Usage: python tools/make_fixtures.py [--out DIR] [--only dsp-rms ...]
DSP cases need nothing; model cases need external/weight; song cases read weights/reference
(run reference.py first). Fixture .bin files over 1 MB are not meant to be committed.
"""
import argparse
import math
import os

import faiss
import librosa
import numpy as np
import torch
from scipy import signal

import reference as R
from common import REFERENCE, ROOT, WEIGHTS_IN, write_case

SRC = f"make_fixtures.py rvc-nano@dda674b librosa {librosa.__version__}"
rng = np.random.default_rng(1234)


def test_signal(n, sr):
    t = np.arange(n) / sr
    return (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 3100 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


def reference_array(case, name):
    import json
    d = os.path.join(REFERENCE, case)
    e = {**(c := json.load(open(os.path.join(d, "case.json"))))["inputs"], **c["outputs"]}[name]
    return np.fromfile(os.path.join(d, e["file"]), dtype=e["dtype"]).reshape(e["shape"])


def resample_cases(out):
    for a, b in ((44100, 16000), (44100, 14700), (14700, 7350), (7350, 14700), (14700, 44100), (40000, 44100)):
        g = math.gcd(a, b)
        up, down = b // g, a // g
        max_rate = max(up, down)  # scipy.signal.resample_poly default filter design
        h = signal.firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up
        x = test_signal(a // 4 + 7, a)  # 0.25 s, odd length
        y = signal.resample_poly(x, up, down).astype(np.float32)
        write_case(os.path.join(out, f"dsp-resample-{a}-{b}"), SRC, {"x": x}, {"y": y, "h": h},
                   {"up": up, "down": down, "window": ["kaiser", 5.0], "halfLen": 10 * max_rate,
                    "design": "firwin(2*halfLen+1, 1/max(up,down), window) * up; resample_poly padtype constant"})


def filtfilt_case(out):
    x = test_signal(16000, 16000)
    write_case(os.path.join(out, "dsp-filtfilt-hpf48"), SRC, {"x": x}, {"y": signal.filtfilt(R.HPF_B, R.HPF_A, x)},
               {"b": R.HPF_B.tolist(), "a": R.HPF_A.tolist(), "padtype": "odd", "padlen": 3 * len(R.HPF_A)})


def stft_cases(out):
    for n_fft, hop, sr in ((1024, 160, 16000), (960, 480, 44100), (640, 80, 7350), (512, 160, 14700), (320, 80, 7350)):
        x = test_signal(sr // 2 + 3, sr)
        s = librosa.stft(x, n_fft=n_fft, hop_length=hop, pad_mode=R.STFT_PAD)
        y = librosa.istft(s, hop_length=hop)
        write_case(os.path.join(out, f"dsp-stft-{n_fft}"), SRC, {"x": x},
                   {"re": s.real.astype(np.float32), "im": s.imag.astype(np.float32), "y": y.astype(np.float32)},
                   {"nFft": n_fft, "hop": hop, "window": "hann (periodic)", "center": True, "padMode": R.STFT_PAD,
                    "istft": "librosa.istft(hop_length=hop), length = hop*(frames-1)"})


def rms_case(out):
    data1 = signal.filtfilt(R.HPF_B, R.HPF_A, test_signal(16000 * 3, 16000) * np.linspace(0.1, 1, 48000))
    data2 = test_signal(40000 * 3, 40000)
    y = R.change_rms(data1, 16000, data2.copy(), 40000, 0.25).astype(np.float32)
    write_case(os.path.join(out, "dsp-rms"), SRC, {"data1": data1, "data2": data2}, {"y": y},
               {"sr1": 16000, "sr2": 40000, "rate": 0.25, "padMode": R.STFT_PAD,
                "frame": "sr//2*2", "hop": "sr//2", "interp": "torch F.interpolate linear (align_corners=False)"})


def rmvpe_cases(out):
    from rmvpe import RMVPE
    rm = RMVPE(os.path.join(WEIGHTS_IN, "rmvpe.pt"), is_half=False, device="cpu")
    x = reference_array("bunnygirl-rvc-kyubey", "vocal16k")[16000 * 60: 16000 * 70].copy()
    with torch.no_grad():
        hidden = rm.mel2hidden(rm.mel_extractor(torch.from_numpy(x)[None], center=True))[0].numpy()
    f0 = rm.decode(hidden, thred=0.03).astype(np.float32)
    meta = {"threshold": 0.03, "centsOffset": 1997.3794084376191}
    write_case(os.path.join(out, "rmvpe-decode"), SRC, {"hidden": hidden}, {"f0": f0}, meta)
    write_case(os.path.join(out, "rmvpe-10s"), SRC, {"audio": x}, {"hidden": hidden, "f0": f0},
               {**meta, "segment": "bunnygirl vocal16k 60-70 s"})


def contentvec_case(out):
    from export_hubert import load_contentvec
    x = reference_array("bunnygirl-rvc-kyubey", "vocal16k")[16000 * 60: 16000 * 70].copy()
    with torch.no_grad():
        f768, f256 = [t.numpy() for t in load_contentvec()(torch.from_numpy(x)[None])]
    write_case(os.path.join(out, "contentvec-10s"), SRC, {"source": x[None]}, {"feats768": f768, "feats256": f256},
               {"segment": "bunnygirl vocal16k 60-70 s"})


def rvc_template_cases(out):
    from export_rvc_templates import build_net, example_inputs, make_template
    cpt = torch.load(os.path.join(WEIGHTS_IN, "rvc-kyubey.pth"), map_location="cpu", weights_only=True)
    net = build_net("v2", 1, cpt["config"], cpt["weight"])
    tpl = make_template(net, 1)
    for name, zero in (("rvc-template-v2-f0-40k", False), ("rvc-template-v2-f0-40k-f0zero", True)):
        x = example_inputs("v2", 1, 400, 40000, T=200, seed=0, f0_zero=zero)
        x["g"] = net.emb_g.weight[0].detach()[None, :, None]
        with torch.no_grad():
            audio = tpl(*x.values()).numpy()
        write_case(os.path.join(out, name), SRC, {k: v.numpy() for k, v in x.items()}, {"audio": audio},
                   {"weights": "rvc-kyubey.pth", "sid": 0, "g": "emb_g.weight[0]",
                    "criterion": "f0zero: SNR > 60 dB; with f0: log-mel L1 (NSF phase drift, ARCHITECTURE 12)"})


def uvr_cases(out):
    from export_uvr import load_uvr
    wave = reference_array("bunnygirl-separate", "input44k")[:, 44100 * 60: 44100 * 90].copy()
    mp = R.BandParams()
    X, _, _ = R.vr_analyze(wave, mp)
    mask, coef, patches = R.vr_mask(load_uvr().to(R.DEV), np.abs(X))
    write_case(os.path.join(out, "uvr-patch"), SRC, {"x": patches[1:2]},
               {"mask": np.ascontiguousarray(mask[None, :, :, 256:512])},
               {"coef": coef, "patch": 1, "note": "raw sigmoid mask, no aggressiveness"})
    vocals, inst, coef = R.separate(wave)
    write_case(os.path.join(out, "uvr-song-30s"), SRC, {"input44k": wave}, {"vocals44k": vocals, "inst44k": inst},
               {"segment": "bunnygirl 60-90 s", "agg": 10, "coef": coef, "stftPadMode": R.STFT_PAD,
                "resample": "resample_poly", "highEndProcess": "mirroring"})


def faiss_cases(out):
    for name, d, n in (("faiss-flat", 768, 30), ("faiss-ivf", 768, 4000)):
        base = rng.standard_normal((n, d), dtype=np.float32)
        if name == "faiss-flat":
            index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 16)
            index.train(base)
            index.nprobe = 1
        index.add(base)
        q = (base[rng.integers(0, n, 20)] + 0.3 * rng.standard_normal((20, d), dtype=np.float32)).astype(np.float32)
        score, ix = index.search(q, 8)
        case = os.path.join(out, name)
        write_case(case, SRC, {"queries": q}, {"distances": score, "labels": ix.astype(np.int64)},
                   {"indexFile": "index.faiss", "k": 8, "metric": "L2 squared", "nprobe": getattr(index, "nprobe", None)})
        faiss.write_index(index, os.path.join(case, "index.faiss"))


def pipeline_case(out):
    from export_hubert import load_contentvec
    from rmvpe import RMVPE
    vocal = reference_array("bunnygirl-rvc-kyubey", "vocal16k")[16000 * 60: 16000 * 80].copy()
    rvc = R.Rvc(os.path.join(WEIGHTS_IN, "rvc-kyubey.pth"), os.path.join(WEIGHTS_IN, "rvc-kyubey.index"))
    r = R.convert(vocal, rvc, load_contentvec().to(R.DEV), RMVPE(os.path.join(WEIGHTS_IN, "rmvpe.pt"), is_half=False, device=R.DEV),
                  index_rates=(0.75,), seed=0)
    v = r["per_rate"][0.75]
    write_case(os.path.join(out, "pipeline-voice-20s"), SRC,
               {"vocal16k": vocal, "rnd": np.concatenate(r["rnd"], axis=1), "noise": np.concatenate(r["noise"])},
               {"f0Hz": r["f0"], "feats": np.concatenate(r["feats"]), "featsIr75": np.concatenate(v["feats"]),
                "converted": v["converted"]},
               {"model": "rvc-kyubey.pth", "index": "rvc-kyubey.index", "upKey": 0, "indexRate": 0.75, "protect": 0.33,
                "rmsMixRate": 0.25, "sid": 0, "chunks": r["chunks"], "segment": "bunnygirl vocal16k 60-80 s",
                "note": "same stage layout as weights/reference/bunnygirl-rvc-kyubey (see its case.json meta)"})


CASES = {"resample": resample_cases, "filtfilt": filtfilt_case, "stft": stft_cases, "rms": rms_case,
         "rmvpe": rmvpe_cases, "contentvec": contentvec_case, "rvc-template": rvc_template_cases, "uvr": uvr_cases,
         "faiss": faiss_cases, "pipeline": pipeline_case}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures"))
    ap.add_argument("--only", nargs="*", choices=list(CASES))
    a = ap.parse_args()
    for name, fn in CASES.items():
        if not a.only or name in a.only:
            fn(a.out)
            print("done", name)
