"""rmvpe.pt -> rmvpe/rmvpe.{onnx,pth,manifest.json}: 16 kHz audio -> log-mel (in graph) -> E2E -> hidden.

Usage: python tools/export_rmvpe.py
The STFT is a strided Conv1d with a windowed DFT basis (no STFT op, WebGPU-friendly), mel basis as in
rvc-nano MelSpectrogram (librosa mel, htk). Both bases are non-persistent buffers, so they stay inline in
the graph and are not part of the manifest (17.2 rule 3). Decode (argmax local average) stays in JS.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

from common import DIST, WEIGHTS_IN, export_template, io_spec
from rmvpe import E2E

N_FFT, HOP, N_MELS, FMIN, FMAX, CLAMP = 1024, 160, 128, 30, 8000, 1e-5
CENTS_OFFSET = 1997.3794084376191  # rvc-nano RMVPE.cents_mapping


class RmvpeGraph(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        e2e = E2E(4, 1, (2, 2))
        e2e.load_state_dict({k: v.float() for k, v in sd.items()})
        self.unet, self.cnn, self.fc = e2e.unet, e2e.cnn, e2e.fc  # same keys as rmvpe.pt
        window = torch.hann_window(N_FFT)  # periodic, = torch.stft default used by rvc-nano
        dft = torch.fft.rfft(torch.eye(N_FFT), dim=1).T  # [513, 1024] complex: bin k of e_n
        basis = torch.cat([dft.real, dft.imag]) * window  # [1026, 1024]
        self.register_buffer("stft_basis", basis[:, None, :].float(), persistent=False)
        mel = librosa_mel(sr=16000, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float(), persistent=False)
        self.eval()

    def forward(self, audio):  # [1, N] -> [1, F, 360], F = N // 160 + 1
        x = F.pad(audio[:, None], (N_FFT // 2, N_FFT // 2), mode="reflect")
        spec = F.conv1d(x, self.stft_basis, stride=HOP)  # [1, 1026, F]
        re, im = spec[:, : N_FFT // 2 + 1], spec[:, N_FFT // 2 + 1:]
        mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, torch.sqrt(re * re + im * im)), min=CLAMP))
        n = mel.shape[-1]
        mel = F.pad(mel, (0, (32 - n % 32) % 32))  # U-Net needs a multiple of 32 frames (RMVPE.mel2hidden)
        h = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2)
        return self.fc(h)[:, :n]


def load_rmvpe():
    return RmvpeGraph(torch.load(os.path.join(WEIGHTS_IN, "rmvpe.pt"), map_location="cpu", weights_only=True))


if __name__ == "__main__":
    net = load_rmvpe()
    with torch.no_grad():
        man = export_template(
            net, (torch.randn(1, 16000),), os.path.join(DIST, "rmvpe"), "rmvpe", id="rmvpe",
            inputs=[io_spec("audio", "float32", [1, "N"])], outputs=[io_spec("hidden", "float32", [1, "F", 360])],
            meta={"sr": 16000, "hop": HOP, "threshold": 0.03, "centsOffset": CENTS_OFFSET})
    assert [t["key"] for t in man["tensors"]] == [k for k in net.state_dict() if not k.endswith("num_batches_tracked")]
