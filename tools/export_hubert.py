"""hubert_base.pt (fairseq, ContentVec-compatible) -> contentvec/contentvec.{onnx,pth,manifest.json}.

Usage: python tools/export_hubert.py
Model code: torchaudio 2.8 hubert_base(); key mapping: torchaudio import_fairseq._map_key (ARCHITECTURE 14.2).
Outputs (17.3): feats768 = layer-12 output (v2), feats256 = final_proj(layer-9 output) (v1), as rvc-nano.
"""
import os

import torch
import torchaudio
from torchaudio.models.wav2vec2.utils.import_fairseq import _convert_state_dict

from common import DIST, WEIGHTS_IN, export_template, io_spec


class _Dictionary:  # stands in for fairseq.data.dictionary.Dictionary inside the checkpoint pickle
    pass


def load_fairseq_state_dict(path=os.path.join(WEIGHTS_IN, "hubert_base.pt")):
    torch.serialization.add_safe_globals([(_Dictionary, "fairseq.data.dictionary.Dictionary")])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    m = ckpt["cfg"]["model"]
    assert (m["extractor_mode"], m["encoder_layers"], m["layer_norm_first"], m["conv_pos"]) == ("default", 12, False, 128)
    return ckpt["model"]


class ContentVec(torch.nn.Module):
    def __init__(self, fairseq_sd):
        super().__init__()
        base = torchaudio.models.hubert_base()
        sd = {k: v.float() for k, v in _convert_state_dict(fairseq_sd).items() if k != "label_embs_concat"}
        base.load_state_dict(sd)  # strict: everything else must map
        self.feature_extractor, self.encoder = base.feature_extractor, base.encoder  # keys without a prefix
        self.final_proj = torch.nn.Linear(768, 256)
        self.final_proj.load_state_dict({"weight": fairseq_sd["final_proj.weight"].float(),
                                         "bias": fairseq_sd["final_proj.bias"].float()})
        self.eval()

    def forward(self, source):
        x, _ = self.feature_extractor(source, None)
        layers = self.encoder.extract_features(x, None, num_layers=12)
        return layers[11], self.final_proj(layers[8])


def load_contentvec():
    return ContentVec(load_fairseq_state_dict())


if __name__ == "__main__":
    net = load_contentvec()
    with torch.no_grad():
        man = export_template(
            net, (torch.randn(1, 16000),), os.path.join(DIST, "contentvec"), "contentvec", id="contentvec",
            inputs=[io_spec("source", "float32", [1, "N"])],
            outputs=[io_spec("feats768", "float32", [1, "F", 768]), io_spec("feats256", "float32", [1, "F", 256])],
            meta={"sr": 16000, "hop": 320})
    assert [t["key"] for t in man["tensors"]] == list(net.state_dict()), "initializers != state_dict"
