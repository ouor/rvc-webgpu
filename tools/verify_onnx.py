"""Numerical check: torch (original weights, fp32) vs onnxruntime CPU (dist graph + dist .pth injected like the
browser loader). Criteria: ARCHITECTURE 12.

Usage: python tools/verify_onnx.py [contentvec rmvpe uvr rvc]   -> prints a table, writes weights/verify.json
Real audio comes from weights/reference (run reference.py first); without it, noise is used.
"""
import json
import os
import subprocess
import sys

import librosa
import numpy as np
import torch

from common import DIST, REFERENCE, ROOT, WEIGHTS_IN, load_session, snr_db

RESULTS = {}


def dist(*p):
    return os.path.join(DIST, *p)


def session(mid, name):
    sd = torch.load(dist(mid, name + ".pth"), map_location="cpu", weights_only=True)
    return load_session(dist(mid, name + ".onnx"), dist(mid, name + ".manifest.json"), sd)


def vocal16k(seconds, offset=60.0):
    path = os.path.join(REFERENCE, "bunnygirl-rvc-kyubey", "vocal16k.bin")
    if not os.path.exists(path):
        return np.random.default_rng(0).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.1
    x = np.fromfile(path, dtype="<f4")
    return x[int(offset * 16000): int((offset + seconds) * 16000)].copy()


def report(name, **metrics):
    RESULTS[name] = metrics
    print(f"{name:28s} " + "  ".join(f"{k}={v:.7g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))


def cos_rows(a, b):
    a, b = a.reshape(-1, a.shape[-1]).astype(np.float64), b.reshape(-1, b.shape[-1]).astype(np.float64)
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(c.mean()), float(c.min())


def check_contentvec():
    from export_hubert import load_contentvec

    net, sess = load_contentvec(), session("contentvec", "contentvec")
    for sec in (10.0, 43.0, 3.217):
        x = vocal16k(sec)
        with torch.no_grad():
            t768, t256 = [t.numpy() for t in net(torch.from_numpy(x)[None])]
        o768, o256 = sess.run(None, {"source": x[None]})
        assert o768.shape == t768.shape, (o768.shape, t768.shape)
        m768, n768 = cos_rows(t768, o768)
        m256, n256 = cos_rows(t256, o256)
        report(f"contentvec {sec:g}s", frames=o768.shape[1], cos768_mean=m768, cos768_min=n768,
               cos256_mean=m256, cos256_min=n256, ok=min(n768, n256) > 0.999)


def check_rmvpe():
    from rmvpe import RMVPE

    ref = RMVPE(os.path.join(WEIGHTS_IN, "rmvpe.pt"), is_half=False, device="cpu")  # rvc-nano path: torch.stft
    sess = session("rmvpe", "rmvpe")
    for sec in (30.0, 10.31):
        x = vocal16k(sec)
        with torch.no_grad():
            th = ref.mel2hidden(ref.mel_extractor(torch.from_numpy(x)[None], center=True))[0].numpy()
        oh = sess.run(None, {"audio": x[None]})[0][0]
        assert oh.shape == th.shape == (len(x) // 160 + 1, 360), (oh.shape, th.shape)
        tf, of = ref.decode(th), ref.decode(oh)
        both = (tf > 0) & (of > 0)
        cents = np.abs(1200 * np.log2(of[both] / tf[both]))
        report(f"rmvpe {sec:g}s", frames=len(tf), hidden_maxabs=float(np.abs(th - oh).max()),
               voicing_agree=float(((tf > 0) == (of > 0)).mean()), cents_mean=float(cents.mean()),
               cents_max=float(cents.max()), ok=bool(cents.mean() < 5 and ((tf > 0) == (of > 0)).mean() > 0.99))


def check_uvr():
    import export_uvr
    from reference import BandParams, vr_analyze

    net, sess = export_uvr.load_uvr(), session("uvr", "uvr-hp-vocal")
    sep = os.path.join(REFERENCE, "bunnygirl-separate", "input44k.bin")
    if os.path.exists(sep):
        wave = np.fromfile(sep, dtype="<f4").reshape(2, -1)[:, 44100 * 60: 44100 * 75]
        mag = np.abs(vr_analyze(wave, BandParams())[0])
        x = np.stack([mag[:, :, i: i + 512] for i in (0, 512)]) / mag.max()
    else:
        x = np.random.default_rng(0).random((2, 2, 673, 512))
    x = x.astype(np.float32)
    with torch.no_grad():
        tm = net(torch.from_numpy(x)).numpy()
        patched = (tm * x[:, :, :, 128:-128])
        aspp = export_uvr.layers_123821KB.ASPPModule
        aspp.forward = export_uvr.ORIG_ASPP_FORWARD
        try:  # original (unpatched ASPP) predict() = mask * mix
            orig = export_uvr.nets_61968KB.CascadedASPPNet.forward(net, torch.from_numpy(x))[:, :, :, 128:-128].numpy()
        finally:
            aspp.forward = export_uvr._aspp_forward
    om = sess.run(None, {"x": x})[0]
    report("uvr B=2", ort_vs_torch_snr=snr_db(tm, om), maxabs=float(np.abs(tm - om).max()),
           aspp_patch_vs_original_snr=snr_db(orig, patched), ok=snr_db(tm, om) > 40)


def logmel(y, sr):
    m = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    return np.log10(np.maximum(m, 1e-5))


def check_rvc():
    import math
    from export_rvc_templates import CONFIGS, build_net, example_inputs, make_template, template_id

    def run(tid, tpl, f0, upp, T, sd, sr, label):
        sess = load_session(dist("rvc", tid + ".onnx"), dist("rvc", tid + ".manifest.json"), sd)
        x = example_inputs(tid[:2], f0, upp, T=T, seed=T)
        if f0:
            x["pitchf"] = x["pitchf"] * 0 if label.endswith("f0=0") else x["pitchf"]
        with torch.no_grad():
            t = tpl(*x.values()).numpy()
        o = sess.run(None, {k: v.numpy() for k, v in x.items()})[0]
        assert o.shape == t.shape == (1, 1, T * upp), (o.shape, t.shape)
        s, l1 = snr_db(t, o), float(np.abs(logmel(t[0, 0], sr) - logmel(o[0, 0], sr)).mean())
        need = 60 if (not f0 or label.endswith("f0=0")) else None
        report(f"{tid} {label} T={T}", snr=s, logmel_l1=l1, ok=(s > need) if need else l1 < 0.05)

    for (version, sr), cfg in CONFIGS.items():
        for f0 in (1, 0):
            tid = template_id(version, f0, sr)
            torch.manual_seed(0)
            tpl = make_template(build_net(version, f0, cfg), f0)
            upp = math.prod(cfg[12])
            for T in (200, 333):
                for label in (("rand f0", "f0=0") if f0 else ("nof0",)):
                    run(tid, tpl, f0, upp, T, tpl.state_dict(), sr, "random-w " + label)
    for name in ("rvc-kyubey", "rvc-nemu"):  # real user checkpoints through the shared v2-f0-40k template
        cpt = torch.load(os.path.join(WEIGHTS_IN, name + ".pth"), map_location="cpu", weights_only=True)
        tpl = make_template(build_net("v2", 1, cpt["config"], cpt["weight"]), 1)
        for label in ("rand f0", "f0=0"):
            run("v2-f0-40k", tpl, 1, 400, 333, cpt["weight"], 40000, f"{name} {label}")


def check_pth_reader():
    """17.3: public .pth files must pass the browser allowlist reader (spikes/pth-reader.mjs)."""
    reader = os.path.join(ROOT, "docs", "dev", "architecture", "spikes", "pth-reader.mjs")
    if not os.path.exists(reader):
        print("pth-reader.mjs not found, skipped")
        return
    files = [dist(m, n + ".pth") for m, n in (("contentvec", "contentvec"), ("rmvpe", "rmvpe"), ("uvr", "uvr-hp-vocal"))]
    out = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "check_pth.mjs"), reader, *files],
                         capture_output=True, text=True)
    print(out.stdout + out.stderr)
    RESULTS["pth-reader"] = {"ok": out.returncode == 0}


if __name__ == "__main__":
    checks = {"contentvec": check_contentvec, "rmvpe": check_rmvpe, "uvr": check_uvr, "rvc": check_rvc,
              "pth": check_pth_reader}
    for name in sys.argv[1:] or checks:
        checks[name]()
    with open(os.path.join(ROOT, "weights", "verify.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    bad = [k for k, v in RESULTS.items() if not v.get("ok", True)]
    print("FAILED:" if bad else "all ok", bad if bad else "")
