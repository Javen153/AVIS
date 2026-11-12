"""Training losses: weighted deep clustering, focal pitch, uncertainty weighting."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def deep_clustering(emb, dom, active, weights=None, cell_mask=None):
    """Weighted deep clustering in the efficient form.

    emb [B,T,F,D] (L2-normalized along D), dom [B,T,F] int, active [B,T,F].
    weights [B,T,F] are band energy * activity; cell_mask keeps only cells where
    the dominant source leads the runner-up by the configured dB margin.
    """
    sel = active > 0.5
    if cell_mask is not None:
        sel = sel & (cell_mask > 0.5)
    if int(sel.sum()) < 8:
        return emb.sum() * 0.0
    V = emb[sel].float()
    y = dom[sel].long()
    w = (weights[sel] if weights is not None else torch.ones_like(active[sel])).float()
    w = w.clamp_min(0)
    K = int(y.max().item()) + 1
    if K < 2 or float(w.sum()) <= 0.0:
        return emb.sum() * 0.0
    Y = torch.zeros(V.shape[0], K, device=V.device, dtype=V.dtype)
    Y.scatter_(1, y[:, None], 1.0)
    sw = w.sum().clamp_min(1e-6)
    # Divide by sw before squaring: the loss is scale-invariant, but the raw
    # (V^T Vw)^2 terms grow as (n_cells * power)^2 and overflow fp16 under AMP
    # (non-finite gradients, D-040).
    Vw = (V * w[:, None]) / sw
    Yw = (Y * w[:, None]) / sw
    a = V.T @ Vw
    b = V.T @ Yw
    c = Y.T @ Yw
    return (a * a).sum() - 2 * (b * b).sum() + (c * c).sum()


def _expand(mask, like):
    m = mask.float()
    if m.dim() == 2:
        m = m[..., None]
    return m.expand_as(like)


def _to_logit(out, name):
    """Prefer the model's float32 logits; fall back to inverting probabilities."""
    z = out.get(f"{name}_logit")
    if z is not None:
        return z
    p = out[name].clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def focal_bce(prob, target, mask, gamma=2.0, alpha=0.25):
    """Sigmoid focal loss on probabilities; supports soft gaussian targets."""
    p = prob.clamp(1e-5, 1.0 - 1e-5)
    t = target.clamp(0.0, 1.0)
    pt = p * t + (1.0 - p) * (1.0 - t)
    at = alpha * t + (1.0 - alpha) * (1.0 - t)
    loss = -at * (1.0 - pt).pow(gamma) * torch.log(pt)
    m = _expand(mask, loss)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def focal_bce_logits(logit, target, mask, gamma=2.0, alpha=0.25):
    """Focal loss from logits: stable even when sigmoid saturates in fp16."""
    t = target.clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * t + (1.0 - p) * (1.0 - t)
    at = alpha * t + (1.0 - alpha) * (1.0 - t)
    loss = at * (1.0 - pt).clamp_min(0.0).pow(gamma) * bce
    m = _expand(mask, loss)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def pos_weight_bce(prob, target, mask, cap=20.0):
    """BCE with pos_weight = neg/pos measured on this batch (clamped)."""
    p = prob.clamp(1e-5, 1.0 - 1e-5)
    m = _expand(mask, target)
    pos = (target * m).sum()
    neg = ((1.0 - target) * m).sum()
    pw = (neg / pos.clamp_min(1e-6)).clamp(0.5, cap)
    bce = F.binary_cross_entropy(p, target, reduction="none")
    w = 1.0 + (pw - 1.0) * target
    return (bce * w * m).sum() / m.sum().clamp_min(1.0)


def pos_weight_bce_logits(logit, target, mask, cap=20.0):
    """Batch-balanced BCE from logits (numerically stable)."""
    m = _expand(mask, target)
    pos = (target * m).sum()
    neg = ((1.0 - target) * m).sum()
    pw = (neg / pos.clamp_min(1e-6)).clamp(0.5, cap)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    w = 1.0 + (pw - 1.0) * target
    return (bce * w * m).sum() / m.sum().clamp_min(1.0)


class UncertaintyWeights(nn.Module):
    """Learned log-variance per loss term (Kendall & Gal).

    loss = sum_i exp(-s_i) * L_i + s_i.  No hand-tuned 0.5/1.0 factors.
    """

    def __init__(self, n=4, init=None):
        super().__init__()
        start = list(init) if init is not None else [0.0] * n
        if len(start) != n:
            raise ValueError(f"log_var_init needs {n} entries")
        self.log_var = nn.Parameter(torch.tensor(start, dtype=torch.float32))

    def forward(self, losses):
        total = 0.0
        for i, l in enumerate(losses):
            total = total + torch.exp(-self.log_var[i]) * l + self.log_var[i]
        return total

    def weights(self):
        return torch.exp(-self.log_var).detach()


def solo_coherence(emb, active, solo):
    """Pull all active bands of a single-source example to one direction.

    S05 (saw to 10 kHz) and S07 (white noise) are solo wideband sources. The
    tracker births a second object once the cosine between the low and high
    band groups drops below MatchThreshold (0.60); measured p50 was 0.29 for
    v3.1 and 0.44-0.49 for v3.2, while v3 (which passes both) sits at 0.93.
    Deep clustering alone is too weak to close that gap, so this term
    optimises the quantity the tracker actually uses (D-041).
    """
    m = (active > 0.5).float() * solo.float()[:, None, None]
    if float(m.sum()) <= 0.0:
        return emb.sum() * 0.0
    v = emb.float()
    den = m.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]
    mean = (v * m[..., None]).sum(dim=(1, 2)) / den
    mean = F.normalize(mean, dim=1)
    cos = (v * mean[:, None, None, :]).sum(dim=-1)
    return ((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0)


def compute_losses(out, labels, cfg=None, uw=None):
    """Return raw terms and the total. `labels` is the dict from make_labels()."""
    cfg = cfg or {}
    l_dc = deep_clustering(out["emb"], labels["dom"], labels["active"],
                           weights=labels.get("power"), cell_mask=labels.get("dc_mask"))
    l_coh = None
    solo = labels.get("solo")
    coh_w = float(cfg.get("solo_coh_w", 0.0))
    if coh_w > 0.0 and solo is not None and float(solo.sum()) > 0.0:
        l_coh = solo_coherence(out["emb"], labels["active"], solo)
    l_act = F.binary_cross_entropy_with_logits(_to_logit(out, "act"), labels["active"],
                                               reduction="mean")
    mode = cfg.get("pitch_loss", "focal")
    pitch_logit = _to_logit(out, "pitch")
    if mode == "focal":
        l_pitch = focal_bce_logits(pitch_logit, labels["pitch"], labels["pitch_mask"],
                                   gamma=float(cfg.get("focal_gamma", 2.0)),
                                   alpha=float(cfg.get("focal_alpha", 0.5)))
    elif mode == "pos_weight":
        l_pitch = pos_weight_bce_logits(pitch_logit, labels["pitch"],
                                        labels["pitch_mask"],
                                        cap=float(cfg.get("pitch_pos_weight_max", 20.0)))
    elif mode == "bce":
        l_pitch = F.binary_cross_entropy_with_logits(
            pitch_logit, labels["pitch"], reduction="none")
        m = _expand(labels["pitch_mask"], l_pitch)
        l_pitch = (l_pitch * m).sum() / m.sum().clamp_min(1.0)
    else:
        raise ValueError(f"unknown pitch_loss {mode}")
    m = labels["active"] > 0.5
    if int(m.sum()) > 0:
        l_sal = ((out["sal"] - labels["sal"]) ** 2)[m].mean()
    else:
        l_sal = out["sal"].sum() * 0.0
    terms = [l_dc, l_act, l_pitch, l_sal]
    if uw is not None:
        total = uw(terms)
        w = uw.weights()
    else:
        w = cfg.get("loss_weights", [1.0, 0.5, 1.0, 0.5])
        total = sum(float(w[i]) * terms[i] for i in range(4))
        w = torch.tensor(w)
    # Outside the uncertainty weighting: log_var[0] could otherwise learn to
    # down-weight the whole DC term to silence the new objective (D-041).
    if l_coh is not None:
        total = total + coh_w * l_coh
    return {"loss": total, "l_dc": l_dc, "l_act": l_act, "l_pitch": l_pitch,
            "l_sal": l_sal, "l_coh": l_coh if l_coh is not None else l_dc * 0.0,
            "w_dc": w[0], "w_act": w[1], "w_pitch": w[2], "w_sal": w[3]}
