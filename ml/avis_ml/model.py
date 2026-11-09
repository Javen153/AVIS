"""M-v1: causal band-sequence encoder + GRU memory + decoders."""
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
        # decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(64, 48, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Conv1d(48, 3 + emb, 1)

    def encode(self, x):
        """x [BT,3,192] -> s1,s2,e"""
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        e = self.enc3(s2)
        return s1, s2, e

    def decode(self, z, s1, s2, e):
        """z [BT,hid] -> head outputs [BT,23,192]"""
        y = self.from_gru(z).reshape(-1, 64, 48) + e
        d2 = self.dec2(y) + s2
        d1 = self.dec1(d2) + s1
        return self.head(d1)

    def split(self, h):
        emb = F.normalize(h[:, :self.emb, :], dim=1)  # channel dim
        act = torch.sigmoid(h[:, self.emb, :])
        pitch = torch.sigmoid(h[:, self.emb + 1, :])
        sal = torch.sigmoid(h[:, self.emb + 2, :])
        return emb, act, pitch, sal

    def forward(self, x, h0=None):
        """x [B,T,3,192] -> dict of embeddings and band heads."""
        B, T = x.shape[0], x.shape[1]
        s1, s2, e = self.encode(x.reshape(B * T, 3, BANDS))
        flat = e.flatten(1)
        z = self.to_gru(self.ln(flat)).reshape(B, T, self.hid)
        g, hn = self.gru(z, h0)
        h = self.decode(g.reshape(B * T, self.hid), s1, s2, e)
        emb, act, pitch, sal = self.split(h)
        return {
            "emb": emb.permute(0, 2, 1).reshape(B, T, BANDS, self.emb),
            "act": act.reshape(B, T, BANDS),
            "pitch": pitch.reshape(B, T, BANDS),
            "sal": sal.reshape(B, T, BANDS),
            "h": hn,
        }

    def step(self, feat, h_in):
        """One frame: feat [1,3,192], h_in [2,1,384] -> outputs + h_out."""
        s1, s2, e = self.encode(feat)
        flat = e.flatten(1)
        z = self.to_gru(self.ln(flat)).unsqueeze(1)  # [1,1,hid]
        g, h_out = self.gru(z, h_in)
        h = self.decode(g[:, 0], s1, s2, e)
        emb, act, pitch, sal = self.split(h)
        return emb.permute(0, 2, 1), act, pitch, sal, h_out


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
