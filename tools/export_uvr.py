"""uvr-vocal.pth (RVC's VR model, nets_61968KB CascadedASPPNet) -> uvr/uvr-hp-vocal.{onnx,pth,manifest.json}
+ uvr/4band_v2.json.

Usage: python tools/export_uvr.py
Graph = mask head only (17.3): x [B,2,673,512] (X_mag / coef) -> sigmoid instrumental mask [B,2,673,256].
Aggressiveness, mask x mix, mirroring and band STFTs stay outside (JS), as in REFERENCES 4.5.
"""
import os
import shutil

import torch
import torch.nn.functional as F

from common import DIST, VENDOR, WEIGHTS_IN, export_template, io_spec
from uvr import layers_123821KB, nets_61968KB

PARAMS_JSON = os.path.join(VENDOR, "uvr", "4band_v2.json")


# AdaptiveAvgPool2d((1, None)) + bilinear resize back to (h, w) is "mean over frequency, broadcast";
# the original form does not export (dynamic output_size).
def _aspp_forward(self, x):
    f1 = self.conv1[1](x.mean(dim=2, keepdim=True)).expand(-1, -1, x.size(2), -1)
    return self.bottleneck(torch.cat((f1, self.conv2(x), self.conv3(x), self.conv4(x), self.conv5(x)), dim=1))


ORIG_ASPP_FORWARD = layers_123821KB.ASPPModule.forward  # kept for verify_onnx.py
layers_123821KB.ASPPModule.forward = _aspp_forward


class MaskNet(nets_61968KB.CascadedASPPNet):
    """CascadedASPPNet.predict() without aggressiveness and without `* mix`: returns the mask."""

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bw = x.size(2) // 2
        aux1 = torch.cat([self.stg1_low_band_net(x[:, :, :bw]), self.stg1_high_band_net(x[:, :, bw:])], dim=2)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(torch.cat([x, aux1], 1)))
        h = self.stg3_full_band_net(self.stg3_bridge(torch.cat([x, aux1, aux2], 1)))
        mask = torch.sigmoid(self.out(h))
        mask = F.pad(mask, (0, 0, 0, self.output_bin - mask.size(2)), mode="replicate")
        return mask[:, :, :, self.offset: -self.offset]


def load_uvr():
    net = MaskNet(672 * 2)  # bins = 672 (4band_v2)
    sd = torch.load(os.path.join(WEIGHTS_IN, "uvr-vocal.pth"), map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() for k, v in sd.items()})
    return net.eval()


if __name__ == "__main__":
    net = load_uvr()
    out = os.path.join(DIST, "uvr")
    with torch.no_grad():
        man = export_template(
            net, (torch.rand(1, 2, 673, 512),), out, "uvr-hp-vocal", id="uvr-hp-vocal",
            inputs=[io_spec("x", "float32", ["B", 2, 673, 512])], outputs=[io_spec("mask", "float32", ["B", 2, 673, 256])],
            meta={"params": "uvr/4band_v2.json", "outputIs": "instrumental", "window": 512, "offset": 128,
                  "roi": 256, "bins": 673})
    want = [k for k in net.state_dict() if not k.endswith("num_batches_tracked") and not k.startswith(("aux1_out", "aux2_out"))]
    assert [t["key"] for t in man["tensors"]] == want, "initializers != state_dict"
    shutil.copyfile(PARAMS_JSON, os.path.join(out, "4band_v2.json"))
