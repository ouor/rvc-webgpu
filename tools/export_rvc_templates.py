"""Export the 12 weightless RVC synthesizer templates: rvc/{v1,v2}-{f0,nof0}-{32k,40k,48k}.{onnx,manifest.json}.

Usage: python tools/export_rvc_templates.py [--only v2-f0-40k ...]
The user's .pth supplies the weights in the browser (ARCHITECTURE.md 17.3), so the module here is randomly
initialised; only parameter names and shapes matter.
"""
import argparse
import math
import os
import types

import numpy as np

import torch
import torch.nn.functional as F

from common import DIST, export_template, io_spec, rvc_signature
from infer_pack import attentions
from infer_pack.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                               SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)


# Stock relative-attention helpers call int()/max() on the length, which tracing bakes into constants
# (the graph then only runs at the traced T). Same math in a trace-friendly form (spike S1).
def _get_rel(self, rel, length):
    w = self.window_size
    return F.pad(rel, [0, 0, length, length])[:, w + 1: w + 2 * length]


def _rel2abs(self, x):
    b, h, l, _ = x.size()
    x = F.pad(x, [0, 1]).view([b, h, l * 2 * l])
    x = F.pad(x, [0, l - 1]).view([b, h, l + 1, 2 * l - 1])
    return x[:, :, :l, l - 1:]


def _abs2rel(self, x):
    b, h, l, _ = x.size()
    x = F.pad(x, [0, l - 1]).view([b, h, l * (2 * l - 1)])
    return F.pad(x, [l, 0]).view([b, h, l, 2 * l])[:, :, :, 1:]


attentions.MultiHeadAttention._get_relative_embeddings = _get_rel
attentions.MultiHeadAttention._relative_position_to_absolute_position = _rel2abs
attentions.MultiHeadAttention._absolute_position_to_relative_position = _abs2rel

# RVC WebUI 2.2.231006 configs/{v1,v2}/*.json as checkpoint config lists (v2 40k uses v1/40k.json).
_COMMON = [192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3]
CONFIGS = {  # (version, sr) -> config
    ("v1", 32000): [513, 40, *_COMMON, [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000],
    ("v1", 40000): [1025, 32, *_COMMON, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    ("v1", 48000): [1025, 24, *_COMMON, [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000],
    ("v2", 32000): [513, 40, *_COMMON, [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000],
    ("v2", 40000): [1025, 32, *_COMMON, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    ("v2", 48000): [1025, 36, *_COMMON, [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000],
}
NETS = {("v1", 1): SynthesizerTrnMs256NSFsid, ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid, ("v2", 0): SynthesizerTrnMs768NSFsid_nono}


def template_id(version, f0, sr):
    return f"{version}-{'f0' if f0 else 'nof0'}-{sr // 1000}k"


def build_net(version, f0, cfg, weight=None):
    """Synthesizer without enc_q, fp32, eval. `weight` = a user .pth 'weight' dict (optional)."""
    cfg = list(cfg)
    if weight is not None:
        cfg[15] = weight["emb_g.weight"].shape[0]
    net = NETS[(version, int(f0))](*cfg, is_half=False)
    del net.enc_q
    if weight is not None:
        net.load_state_dict({k: v.float() for k, v in weight.items()}, strict=False)
    else:  # distinct values: the exporter merges equal-valued initializers (e.g. fresh LayerNorm gammas)
        with torch.no_grad():
            for p in net.parameters():
                p.normal_(0, 0.02)
    return net.eval().float()


def sine_phase(pitchf, upp, sr):
    """NSF sine phase in float64 (ARCHITECTURE 17.3): frac of the inclusive running sum of f0/sr, each frame
    repeated upp times. Mathematically the stock SineGen phase (its cumsum_shift only adds integers), but
    without the fp32 cumsum drift over long chunks. pitchf [T] (Hz) -> float32 [T*upp], restarts at 0 per chunk."""
    rad = np.repeat(np.mod(np.asarray(pitchf, np.float64) / sr, 1.0), upp)
    return np.mod(np.cumsum(rad), 1.0).astype(np.float32)


def _sine_from_phase(self, f0, upp):
    """SineGen.forward (harmonic_num 0, rand_ini 0) with the phase and the noise taken from template inputs."""
    sine = torch.sin(self.phase * (2 * math.pi)) * self.sine_amp
    uv = (f0 > self.voiced_threshold).float()[:, None]  # [B, 1, T]
    uv = F.interpolate(uv, scale_factor=float(upp), mode="nearest").transpose(2, 1)
    noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * self.noise
    return sine * uv + noise, uv, noise


class F0Template(torch.nn.Module):
    """net.infer() with sid -> g, both random sources and the sine phase as inputs (no Random*, no CumSum)."""

    def __init__(self, net):
        super().__init__()
        self.enc_p, self.flow, self.dec = net.enc_p, net.flow, net.dec
        self.sine = self.dec.m_source.l_sin_gen
        self.sine.forward = types.MethodType(_sine_from_phase, self.sine)

    def forward(self, phone, phone_lengths, pitch, pitchf, g, rnd, noise, phase):
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z = self.flow((m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask, x_mask, g=g, reverse=True)
        self.sine.phase, self.sine.noise = phase, noise
        return self.dec(z * x_mask, pitchf, g=g)


class NoF0Template(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.enc_p, self.flow, self.dec = net.enc_p, net.flow, net.dec

    def forward(self, phone, phone_lengths, g, rnd):
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z = self.flow((m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, g=g)


def make_template(net, f0):
    return (F0Template if f0 else NoF0Template)(net).eval()


def example_inputs(version, f0, upp, sr, T=50, seed=0, f0_zero=False):
    g = torch.Generator().manual_seed(seed)
    C = 256 if version == "v1" else 768
    x = {"phone": torch.randn(1, T, C, generator=g), "phone_lengths": torch.tensor([T])}
    if f0:
        x["pitch"] = torch.randint(1, 256, (1, T), generator=g)
        x["pitchf"] = torch.rand(1, T, generator=g) * 400 + 100
        if f0_zero:
            x["pitchf"] = torch.zeros_like(x["pitchf"])
    x["g"] = torch.randn(1, 256, 1, generator=g)
    x["rnd"] = torch.randn(1, 192, T, generator=g)
    if f0:
        x["noise"] = torch.randn(1, T * upp, 1, generator=g)
        x["phase"] = torch.from_numpy(sine_phase(x["pitchf"][0].numpy(), upp, sr))[None, :, None]
    return x


def export_one(version, f0, sr):
    cfg = CONFIGS[(version, sr)]
    upp = math.prod(cfg[12])
    tid = template_id(version, f0, sr)
    C = 256 if version == "v1" else 768
    inputs = [io_spec("phone", "float32", [1, "T", C]), io_spec("phone_lengths", "int64", [1])]
    if f0:
        inputs += [io_spec("pitch", "int64", [1, "T"]), io_spec("pitchf", "float32", [1, "T"])]
    inputs += [io_spec("g", "float32", [1, 256, 1]), io_spec("rnd", "float32", [1, 192, "T"])]
    if f0:
        inputs += [io_spec("noise", "float32", [1, "Tupp", 1]), io_spec("phase", "float32", [1, "Tupp", 1])]
    tpl = make_template(build_net(version, f0, cfg), f0)
    meta = {"version": version, "f0": bool(f0), "sr": sr, "upp": upp, "signature": rvc_signature(version, f0, cfg)}
    if f0:
        meta["phase"] = "float64 acc=0; per frame t: r=(pitchf[t]/sr) mod 1; repeat upp: acc=frac(acc+r); out=acc"
    with torch.no_grad():
        man = export_template(tpl, tuple(example_inputs(version, f0, upp, sr).values()), os.path.join(DIST, "rvc"), tid,
                        id=tid, inputs=inputs, outputs=[io_spec("audio", "float32", [1, 1, "Tupp"])],
                        meta=meta, pth=False)
    assert [t["key"] for t in man["tensors"]] == list(tpl.state_dict()), f"{tid}: initializers != state_dict"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="template ids, e.g. v2-f0-40k")
    a = ap.parse_args()
    for (version, sr) in CONFIGS:
        for f0 in (1, 0):
            if not a.only or template_id(version, f0, sr) in a.only:
                export_one(version, f0, sr)
