"""Training losses for M-v1."""
from __future__ import annotations

import torch


def deep_clustering(emb, dom, active, weights=None):
    """Weighted deep clustering in the efficient form.

    emb [B,T,F,D] (L2-normalized along D), dom [B,T,F] int, active [B,T,F].
    """
    mask = active > 0.5
    if mask.sum() < 8:
        return emb.sum() * 0.0
    V = emb[mask]
    y = dom[mask].long()
    w = (weights[mask] if weights is not None else torch.ones_like(active[mask]))
    K = int(y.max().item()) + 1
    if K < 2:
        return emb.sum() * 0.0
    Y = torch.zeros(V.shape[0], K, device=V.device, dtype=V.dtype)
    Y.scatter_(1, y[:, None], 1.0)
    sw = w.sum().clamp_min(1e-6)
    Vw = V * w[:, None]
    Yw = Y * w[:, None]
    a = V.T @ Vw
    b = V.T @ Yw
    c = Y.T @ Yw
    return ((a * a).sum() - 2 * (b * b).sum() + (c * c).sum()) / (sw * sw)


def compute_losses(out, active, pitch, sal, dom, cfg=None):
    cfg = cfg or {}
    l_dc = deep_clustering(out["emb"], dom, active)
    l_act = torch.nn.functional.binary_cross_entropy(out["act"], active)
    pos = torch.full_like(pitch, 10.0)
    l_pitch = torch.nn.functional.binary_cross_entropy(
        out["pitch"], pitch, weight=pos)
    m = active > 0.5
    if m.sum() > 0:
        l_sal = ((out["sal"] - sal) ** 2)[m].mean()
    else:
        l_sal = out["sal"].sum() * 0.0
    total = l_dc + 0.5 * l_act + 1.0 * l_pitch + 0.5 * l_sal
    return {"loss": total, "l_dc": l_dc, "l_act": l_act, "l_pitch": l_pitch, "l_sal": l_sal}
