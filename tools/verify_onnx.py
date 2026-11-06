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
        x = example_inputs(tid[:2], f0, upp, sr, T=T, seed=T, f0_zero=label.endswith("f0=0"))
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


def check_rvc_long():
    """Long voiced chunks (>= 30 s) through v2-f0-40k + kyubey: output pitch must follow pitchf (AC-2 >= 0.95)."""
    from export_rvc_templates import build_net, make_template, sine_phase
    from reference import quantize_f0, resample
    from rmvpe import RMVPE

    d = os.path.join(REFERENCE, "bunnygirl-rvc-kyubey")
    case = json.load(open(os.path.join(d, "case.json")))
    arr = lambda k: np.fromfile(os.path.join(d, k + ".bin"), dtype={**case["inputs"], **case["outputs"]}[k]["dtype"])
    cpt = torch.load(os.path.join(WEIGHTS_IN, "rvc-kyubey.pth"), map_location="cpu", weights_only=True)
    net = build_net("v2", 1, cpt["config"], cpt["weight"])
    tpl, sess = make_template(net, 1), load_session(dist("rvc", "v2-f0-40k.onnx"), dist("rvc", "v2-f0-40k.manifest.json"), cpt["weight"])
    rm = RMVPE(os.path.join(WEIGHTS_IN, "rmvpe.pt"), is_half=False, device="cpu")
    feats = arr("feats").reshape(-1, 768)

    ch = case["meta"]["chunks"][0]  # first song chunk (~40 s, as rvc-nano cuts it)
    T = ch["T"]
    f0a = ch["f0Frames"][0]
    real = (np.repeat(feats[: ch["featFrames"]], 2, axis=0)[:T], arr("pitch")[f0a: f0a + T], arr("pitchf")[f0a: f0a + T])
    t = np.arange(4000) / 100  # synthetic 40 s, voiced throughout: 220 Hz glide with vibrato
    pf = (220 * 2 ** (t / 40 + 0.05 * np.sin(2 * np.pi * 5.5 * t))).astype(np.float32)
    synth = (np.repeat(feats[:2000], 2, axis=0), quantize_f0(pf, 0)[0], pf)
    rng = np.random.default_rng(7)
    for label, (phone, pitch, pitchf) in (("song chunk0", real), ("synthetic voiced", synth)):
        T = len(pitchf)
        x = {"phone": phone[None].astype(np.float32), "phone_lengths": np.array([T]), "pitch": pitch[None].astype(np.int64),
             "pitchf": pitchf[None], "g": net.emb_g.weight[0].detach().numpy()[None, :, None],
             "rnd": rng.standard_normal((1, 192, T), dtype=np.float32),
             "noise": rng.standard_normal((1, T * 400, 1), dtype=np.float32),
             "phase": sine_phase(pitchf, 400, 40000)[None, :, None]}
        with torch.no_grad():
            ref = tpl(*[torch.from_numpy(v) for v in x.values()]).numpy()[0, 0]
        out = sess.run(None, x)[0][0, 0]
        corr = {}
        for name, y in (("torch", ref), ("ort", out)):
            got = rm.infer_from_audio(resample(y, 40000, 16000).astype(np.float32))[:T]
            v = (pitchf[: len(got)] > 0) & (got > 0)
            corr[name] = float(np.corrcoef(np.log(pitchf[: len(got)][v]), np.log(got[v]))[0, 1])
        report(f"v2-f0-40k kyubey {label} T={T}", seconds=T / 100, snr=snr_db(ref, out),
               logmel_l1=float(np.abs(logmel(ref, 40000) - logmel(out, 40000)).mean()),
               pitch_corr_torch=corr["torch"], pitch_corr_ort=corr["ort"], ok=corr["ort"] >= 0.95)


def check_pth_reader():
    """17.3: public .pth files must pass the browser's allowlist reader rules (common.check_pth)."""
    from common import check_pth
    for mid, name in (("contentvec", "contentvec"), ("rmvpe", "rmvpe"), ("uvr", "uvr-hp-vocal")):
        errors = check_pth(dist(mid, name + ".pth"), dist(mid, name + ".manifest.json"))
        report(f"pth {name}", errors=len(errors), first=errors[0] if errors else "-", ok=not errors)

def check_contract():
    """weights/dist against src/models/contract.json (the single source, R3 A7), then tests/contract.test.ts."""
    import onnx
    from common import CONTRACT

    c, reg = CONTRACT, json.load(open(dist("registry.json"), encoding="utf-8"))
    dtypes = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT64: "int64"}
    errors = []
    err = lambda cond, msg: errors.append(msg) if not cond else None
    err(list(reg) == c["registryKeys"] and reg["format"] == c["registryFormat"], "registry keys/format")
    err(sorted(reg["models"]) == sorted(c["public"]), f"registry models {sorted(reg['models'])}")

    def meta_ok(rules, meta):
        types = {"number": (int, float), "boolean": bool, "string": str}
        return [k for k, r in rules.items() if not (meta.get(k) in r if isinstance(r, list) else
                                                    isinstance(meta.get(k), types[r]) and not (r == "number" and isinstance(meta.get(k), bool)))]

    def check(base, spec, ins, outs, meta_rules, mid):
        man = json.load(open(dist(base + ".manifest.json"), encoding="utf-8"))
        err(list(man) == c["manifestKeys"] and man["format"] == c["manifestFormat"]
            and man["externalData"] == c["externalData"] and man["id"] == mid, f"{base} manifest header")
        err(not meta_ok(meta_rules, man["meta"]), f"{base} meta {meta_ok(meta_rules, man['meta'])}")
        io = lambda xs: [{"name": x["name"], "dtype": x["dtype"]} for x in xs]
        err(io(man["inputs"]) == ins and io(man["outputs"]) == outs, f"{base} manifest io")
        g = onnx.load(dist(base + ".onnx"), load_external_data=False).graph
        gio = lambda xs: [{"name": x.name, "dtype": dtypes.get(x.type.tensor_type.elem_type)} for x in xs]
        err(gio(g.input) == ins and gio(g.output) == outs, f"{base} graph io")

    for mid, spec in c["public"].items():
        m = reg["models"].get(mid, {})
        want = {"graph": spec["files"] + ".onnx", "weights": spec["files"] + ".pth", "manifest": spec["files"] + ".manifest.json"}
        if "params" in spec:
            want["params"] = spec["params"]
        err(m == want, f"registry models.{mid} {m}")
        check(spec["files"], spec, spec["inputs"], spec["outputs"], spec["meta"], mid)
    t = c["template"]
    for sig, e in reg["rvcTemplates"].items():
        base = t["files"].replace("{id}", e["id"])
        err(e == {"id": e["id"], "graph": base + ".onnx", "manifest": base + ".manifest.json"}, f"rvcTemplates {e}")
        f0 = json.load(open(dist(base + ".manifest.json"), encoding="utf-8"))["meta"]["f0"]
        check(base, t, t["inputs"]["f0" if f0 else "nof0"], t["outputs"], t["meta"], e["id"])
    report("contract (contract.json)", errors=len(errors), first=errors[0] if errors else "-", ok=not errors)
    npx = "npx.cmd" if os.name == "nt" else "npx"
    r = subprocess.run([npx, "vitest", "run", "tests/contract.test.ts"], cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    report("contract (tests/contract.test.ts)", exit=r.returncode, summary=" | ".join(x.strip() for x in tail), ok=r.returncode == 0)

if __name__ == "__main__":
    checks = {"contentvec": check_contentvec, "rmvpe": check_rmvpe, "uvr": check_uvr, "rvc": check_rvc, "rvc-long": check_rvc_long,
              "pth": check_pth_reader, "contract": check_contract}
    for name in sys.argv[1:] or checks:
        checks[name]()
    with open(os.path.join(ROOT, "weights", "verify.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    bad = [k for k, v in RESULTS.items() if not v.get("ok", True)]
    print("FAILED:" if bad else "all ok", bad if bad else "")
