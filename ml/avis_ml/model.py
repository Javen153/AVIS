"""M-v2: causal band encoder + GRU memory + separate band decoders per head.

The four output heads no longer share the final decoder. Diagnostics showed the
pitch gradient was ~500x smaller than the embedding gradient when both were
produced by one decoder; the embedding task then starved pitch. Each head now
has its own lightweight upsampling branch after the shared encoder/GRU. The
ONNX step contract (feat, h_in -> emb, act, pitch, sal, h_out) is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = 192
EMB = 20
HID = 384


def conv_block(cin, cout, k=5, stride=1):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2),
        nn.GELU(),
    )


def decoder_branch():
    return nn.Sequential(
        nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv1d(64, 64, 5, padding=2),
        nn.GELU(),
        nn.ConvTranspose1d(64, 48, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv1d(48, 48, 5, padding=2),
        nn.GELU(),
    )


class AvisNet(nn.Module):
    def __init__(self, emb=EMB, hid=HID):
        super().__init__()
        self.emb = emb
        self.hid = hid
        # encoder
        self.enc1 = nn.Sequential(conv_block(3, 48), conv_block(48, 48))
        self.enc2 = nn.Sequential(conv_block(48, 64, stride=2), conv_block(64, 64))
        self.enc3 = conv_block(64, 64, stride=2)
        self.flatten_dim = 64 * 48
        self.ln = nn.LayerNorm(self.flatten_dim)
        self.to_gru = nn.Linear(self.flatten_dim, hid)
        self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True)
        self.from_gru = nn.Linear(hid, self.flatten_dim)
        # separate decoders: the shared bottleneck is only encoder + GRU
        self.dec_emb = decoder_branch()
        self.dec_act = decoder_branch()
        self.dec_pitch = decoder_branch()
        self.dec_sal = decoder_branch()
        self.head_emb = nn.Conv1d(48, emb, 1)
        self.head_act = nn.Conv1d(48, 1, 1)
        self.head_pitch = nn.Conv1d(48, 1, 1)
        self.head_sal = nn.Conv1d(48, 1, 1)

    def encode(self, x):
        """x [BT,3,192] -> s1,s2,e"""
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        e = self.enc3(s2)
        return s1, s2, e

    def decode(self, z, s1, s2, e):
        """z [BT,hid] -> private decoder output per head, each [BT,64,192]."""
        y = self.from_gru(z).reshape(-1, 64, 48) + e
        skips = {"emb": s1, "act": s1, "pitch": s1, "sal": s1}
        out = {}
        for name, dec in (("emb", self.dec_emb), ("act", self.dec_act),
                          ("pitch", self.dec_pitch), ("sal", self.dec_sal)):
            out[name] = dec(y) + skips[name]
        return out

    def heads(self, d):
        """Logits in float32 (BCE-friendly) plus the ONNX-facing probabilities."""
        emb = F.normalize(self.head_emb(d["emb"]).float(), dim=1)
        act = self.head_act(d["act"]).float().squeeze(1)
        pitch = self.head_pitch(d["pitch"]).float().squeeze(1)
        sal = self.head_sal(d["sal"]).float().squeeze(1)
        return emb, act, pitch, sal

    def forward(self, x, h0=None):
        """x [B,T,3,192] -> embeddings, band probabilities and logits."""
        B, T = x.shape[0], x.shape[1]
        s1, s2, e = self.encode(x.reshape(B * T, 3, BANDS))
        flat = e.flatten(1)
        z = self.to_gru(self.ln(flat)).reshape(B, T, self.hid)
        g, hn = self.gru(z, h0)
        d = self.decode(g.reshape(B * T, self.hid), s1, s2, e)
        emb, act, pitch, sal = self.heads(d)
        return {
            "emb": emb.permute(0, 2, 1).reshape(B, T, BANDS, self.emb),
            "act": torch.sigmoid(act).reshape(B, T, BANDS),
            "pitch": torch.sigmoid(pitch).reshape(B, T, BANDS),
            "sal": torch.sigmoid(sal).reshape(B, T, BANDS),
            # logits keep the losses away from sigmoid saturation in fp16
            "act_logit": act.reshape(B, T, BANDS),
            "pitch_logit": pitch.reshape(B, T, BANDS),
            "sal_logit": sal.reshape(B, T, BANDS),
            "h": hn,
        }

    def step(self, feat, h_in):
        """One frame: feat [1,3,192], h_in [2,1,hid] -> outputs + h_out."""
        s1, s2, e = self.encode(feat)
        flat = e.flatten(1)
        z = self.to_gru(self.ln(flat)).unsqueeze(1)  # [1,1,hid]
        g, h_out = self.gru(z, h_in)
        d = self.decode(g[:, 0], s1, s2, e)
        emb, act, pitch, sal = self.heads(d)
        return (emb.permute(0, 2, 1), torch.sigmoid(act), torch.sigmoid(pitch),
                torch.sigmoid(sal), h_out)


class StepWrapper(nn.Module):
    """ONNX-friendly wrapper around AvisNet.step."""

    def __init__(self, net: AvisNet):
        super().__init__()
        self.net = net

    def forward(self, feat, h_in):
        emb, act, pitch, sal, h_out = self.net.step(feat, h_in)
        return emb, act, pitch, sal, h_out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
