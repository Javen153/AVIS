"""M-v3: harmonic-prior input, depthwise-separable spectrum encoder, slim shared decoder.

Design notes (see docs/REQUIREMENTS_V3.md and docs/DECISIONS.md D-037):
- A fixed subharmonic-summation prior is computed from the mid-band dB channel.
  It is fed as a fourth encoder channel and as a residual pitch logit, so the
  pitch head can use harmonic structure that survives masking noise.
- The frequency encoder uses depthwise-separable blocks instead of dense 5x5
  convolutions; the GRU is 2x208 instead of 2x280.
- One shared transposed-conv decoder replaces four full decoder branches. Each
  head keeps a private 3x3 adapter, which preserves the per-head gradient path.
- The ONNX step contract is unchanged: feat[1,3,192], h[2,1,H] ->
  emb[1,192,20], act/pitch/sal[1,192], h_out[2,1,H].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = 192
EMB = 20
HID = 208
F_MIN = 30.0
F_MAX = 16000.0


class HarmonicPrior(nn.Module):
    """Fixed subharmonic summation on the 192-band log-frequency axis.

    score[b] = sum_h P[b + 21.09*log2(h)] / h, then a level-invariant harmonicity
    map: score is divided by its frame mean and log-compressed, so a flat or noisy
    spectrum maps to ~0 and only harmonic peaks approach 1. The kernel is a
    buffer, so the prior is identical in PyTorch and ONNX and needs no training.
    """

    def __init__(self, bands: int = BANDS, fmin: float = F_MIN, fmax: float = F_MAX,
                 harmonics: int = 5):
        super().__init__()
        bpo = (bands - 1) / math.log2(fmax / fmin)
        shifts = [int(round(bpo * math.log2(h))) for h in range(2, harmonics + 2)]
        size = max(shifts) + 1
        kernel = torch.zeros(1, 1, size)
        kernel[0, 0, 0] = 1.0
        for h, shift in enumerate(shifts, start=2):
            kernel[0, 0, shift] = 1.0 / h
        self.register_buffer("kernel", kernel)
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N,3,192] -> prior [N,192] in [0,1] (0 = flat/noise, 1 = strong peak)."""
        # fixed buffer maths stays fp32 under AMP (small dB values underflow in fp16)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            db = x.float()[:, 0] * 30.0 - 60.0
            p = torch.exp(db * 0.2302585093).clamp(0.0, 1e4)  # ln(10)/10
            zp = F.pad(p.unsqueeze(1), (0, self.size - 1))
            acc = F.conv1d(zp, self.kernel)[:, 0, :]
            # excess over the frame mean: scale-invariant, and a noise floor with no
            # harmonic structure stays close to 1 (prior 0) instead of saturating at 1
            ratio = (acc + 1e-6) / (acc.mean(dim=1, keepdim=True) + 1e-6)
            # log contrast in [0,1]: ratio 3 -> 0 (broadband noise), ratio 36 -> 1
            # (a harmonic peak well above the frame average)
            return ((torch.log(ratio) - math.log(3.0)) / math.log(12.0)).clamp(0.0, 1.0)


class DSBlock(nn.Module):
    """Mobile-style depthwise-separable 1D block (expand -> depthwise -> project)."""

    def __init__(self, cin: int, cout: int, k: int = 5, stride: int = 1, expand: int = 2):
        super().__init__()
        mid = cin * expand
        self.pw1 = nn.Conv1d(cin, mid, 1)
        self.dw = nn.Conv1d(mid, mid, k, stride=stride, padding=k // 2, groups=mid)
        self.pw2 = nn.Conv1d(mid, cout, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw2(self.act(self.dw(self.act(self.pw1(x)))))


def adapter(cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(48, 48, 3, padding=1),
        nn.GELU(),
        nn.Conv1d(48, cout, 1),
    )


class AvisNetV3(nn.Module):
    def __init__(self, emb: int = EMB, hid: int = HID):
        super().__init__()
        self.emb = emb
        self.hid = hid
        self.prior = HarmonicPrior()
        # encoder: 4 input channels (3 FE + prior)
        self.enc1 = nn.Sequential(nn.Conv1d(4, 48, 5, padding=2), nn.GELU(),
                                  DSBlock(48, 48, k=5))
        self.enc2 = DSBlock(48, 64, k=5, stride=2)
        self.enc3 = DSBlock(64, 64, k=5, stride=2)
        self.flat = 64 * 48
        self.ln = nn.LayerNorm(self.flat)
        self.to_gru = nn.Linear(self.flat, hid)
        self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True)
        self.from_gru = nn.Linear(hid, self.flat)
        # one shared decoder, private 3x3 adapter per head
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 48, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(48, 48, 5, padding=2), nn.GELU(),
            nn.ConvTranspose1d(48, 48, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(48, 48, 5, padding=2), nn.GELU(),
        )
        self.skip1 = nn.Conv1d(48, 48, 1)
        self.adapt_emb = adapter(48)
        self.adapt_act = adapter(32)
        self.adapt_pitch = adapter(32)
        self.adapt_sal = adapter(32)
        self.head_emb = nn.Conv1d(48, emb, 1)
        self.head_act = nn.Conv1d(32, 1, 1)
        self.head_pitch = nn.Conv1d(32, 1, 1)
        self.head_sal = nn.Conv1d(32, 1, 1)
        self.prior_gain = nn.Parameter(torch.tensor(0.5))

    def encode(self, x: torch.Tensor):
        """x [N,3,192] -> s1 [N,48,192], e [N,64,48], prior [N,192]."""
        pr = self.prior(x)
        h = torch.cat([x, pr.unsqueeze(1)], dim=1)
        s1 = self.enc1(h)
        s2 = self.enc2(s1)
        e = self.enc3(s2)
        return s1, e, pr

    def decode(self, z: torch.Tensor, s1: torch.Tensor, e: torch.Tensor,
               pr: torch.Tensor) -> dict:
        y = self.from_gru(z).reshape(-1, 64, 48) + e
        y = self.dec(y) + self.skip1(s1)
        return {
            "emb": self.adapt_emb(y),
            "act": self.adapt_act(y),
            "pitch": self.adapt_pitch(y),
            "sal": self.adapt_sal(y),
            "prior": pr,
        }

    def heads(self, d: dict):
        """Logits in float32 (BCE-friendly) plus the ONNX-facing probabilities."""
        emb = F.normalize(self.head_emb(d["emb"]).float(), dim=1)
        act = self.head_act(d["act"]).float().squeeze(1)
        pitch = self.head_pitch(d["pitch"]).float().squeeze(1)
        pitch = pitch + self.prior_gain.float() * (d["prior"] * 6.0 - 3.0)
        sal = self.head_sal(d["sal"]).float().squeeze(1)
        return emb, act, pitch, sal

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        B, T = x.shape[0], x.shape[1]
        s1, e, pr = self.encode(x.reshape(B * T, 3, BANDS))
        z = self.to_gru(self.ln(e.flatten(1))).reshape(B, T, self.hid)
        g, hn = self.gru(z, h0)
        d = self.decode(g.reshape(B * T, self.hid), s1, e, pr)
        emb, act, pitch, sal = self.heads(d)
        return {
            "emb": emb.permute(0, 2, 1).reshape(B, T, BANDS, self.emb),
            "act": torch.sigmoid(act).reshape(B, T, BANDS),
            "pitch": torch.sigmoid(pitch).reshape(B, T, BANDS),
            "sal": torch.sigmoid(sal).reshape(B, T, BANDS),
            "act_logit": act.reshape(B, T, BANDS),
            "pitch_logit": pitch.reshape(B, T, BANDS),
            "sal_logit": sal.reshape(B, T, BANDS),
            "h": hn,
        }

    def step(self, feat: torch.Tensor, h_in: torch.Tensor):
        """One frame: feat [1,3,192], h_in [2,1,hid] -> outputs + h_out."""
        s1, e, pr = self.encode(feat)
        z = self.to_gru(self.ln(e.flatten(1))).unsqueeze(1)
        g, h_out = self.gru(z, h_in)
        d = self.decode(g[:, 0], s1, e, pr)
        emb, act, pitch, sal = self.heads(d)
        return (emb.permute(0, 2, 1), torch.sigmoid(act), torch.sigmoid(pitch),
                torch.sigmoid(sal), h_out)


class StepWrapperV3(nn.Module):
    def __init__(self, net: AvisNetV3):
        super().__init__()
        self.net = net

    def forward(self, feat, h_in):
        return self.net.step(feat, h_in)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
