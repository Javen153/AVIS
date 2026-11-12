"""Fixed validation sets, composite checkpoint metric and gradients diagnostics."""
from __future__ import annotations

import math

import numpy as np
import torch

from .data_index import data_root
from .losses import compute_losses
from .mixer import make_labels
from .metrics import (act_iou, cell_accuracy, embedding_cosine, pitch_ap,
                      pitch_f1_sweep, pitch_peak_f1, pitch_prob_stats)
from .mixer_v2 import RealMixer
from .real_data import RealSourceSampler, load_pool


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def _to_dev(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_dev(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dev(v, dev) for v in obj]
    return obj


def labels_of(batch, tf, margin_db=3.0):
    lab = make_labels(batch["mix"], batch["dom"], batch["pitch"], batch["sal"], tf,
                      margin=batch.get("margin"), pitch_mask=batch.get("pitch_mask"),
                      margin_db=margin_db)
    lab["solo"] = batch.get("solo")
    return lab


def make_val_sets(cfg, tf, base, workers=0):
    """Build one fixed, seeded validation set for procedural and real data.

    Examples are kept on CPU (4 s stereo each) so a 200+ example set does not
    eat the 4 GB VRAM budget. Both subsets are regenerated identically in every
    process: the mixer generator is seeded and reset afterwards (D-030).
    """
    n = int(cfg.get("val_batch", 8))
    proc_n = int(cfg.get("val_proc_examples", 120))
    real_n = int(cfg.get("val_real_examples", 120))
    seed = int(cfg.get("val_seed", 777))
    sets = {"proc": []}
    if proc_n:
        base.g.manual_seed(seed)
        done = 0
        i = 0
        while done < proc_n:
            bs = min(n, proc_n - done)
            sets["proc"].append(_to_cpu(base.batch(bs, seed=seed + i * 100003)))
            done += bs
            i += 1
    v3 = cfg.get("arch") == "v3"
    stereo = v3 and bool(cfg.get("stereo", True))
    sampler = None
    if cfg.get("real") and (real_n or int(cfg.get("val_hard_examples", 0))):
        base.g.manual_seed(seed)
        pool = load_pool(data_root() / "index.jsonl", ("validation",))
        sampler = RealSourceSampler(pool, batch=int(cfg.get("real_batch", 16)),
                                    workers=workers, seed=seed, stereo=stereo)
    if cfg.get("real") and real_n:
        vmix = _val_mixer(cfg, base, sampler, tf, proc_p=0.0, hard_p=0.0, stereo=stereo)
        sets["real"] = []
        done = 0
        while done < real_n:
            bs = min(n, real_n - done)
            sets["real"].append(_to_cpu(vmix.batch(bs)))
            done += bs
    hard_n = int(cfg.get("val_hard_examples", 0))
    if v3 and hard_n:
        hseed = seed + 12345
        base.g.manual_seed(hseed)
        hmix = _val_mixer(cfg, base, None, tf, proc_p=0.0, hard_p=1.0, stereo=stereo)
        sets["hard"] = []
        done = 0
        while done < hard_n:
            bs = min(n, hard_n - done)
            sets["hard"].append(_to_cpu(hmix.batch(bs)))
            done += bs
    base.g.manual_seed(int(cfg.get("seed", 0)))
    return sets


def _val_mixer(cfg, base, sampler, tf, proc_p, hard_p, stereo):
    if cfg.get("arch") == "v3":
        from .mixer_v3 import RealMixerV3
        return RealMixerV3(base, sampler, tf, proc_p=proc_p,
                           distort_p=float(cfg.get("mix_distort_p", 0.5)),
                           hard_p=hard_p, stereo=stereo)
    return RealMixer(base, sampler, tf, proc_p=proc_p,
                     distort_p=float(cfg.get("mix_distort_p", 0.5)))


def _set_batches(sets):
    for name in sorted(sets):
        for b in sets[name]:
            yield name, b


def evaluate(net, tf, sets, device, cfg=None, uw=None, sweep=False, cell_acc=True,
             max_emb=50000, max_cell_frames=1200):
    """Raw losses + pitch AP/F1 + act IoU + cluster metrics on a fixed set."""
    cfg = cfg or {}
    net.eval()
    keys = ["loss", "l_dc", "l_act", "l_pitch", "l_sal"]
    acc = {k: 0.0 for k in keys}
    acc_by = {}
    n_ex = 0
    pitch_p, pitch_y = [], []
    sel_pitch_p, sel_pitch_y = [], []
    sweep_p, sweep_y, sweep_m = [], [], []
    act_tp = act_fp = act_fn = 0
    sel_tp = sel_fp = sel_fn = 0
    sel_dc_sum = 0.0
    sel_n = 0
    batch_ap, batch_iou = [], []
    emb_parts, dom_parts, active_parts = [], [], []
    margin_db = float(cfg.get("dc_margin_db", 3.0))
    with torch.no_grad():
        for name, b in _set_batches(sets):
            b = _to_dev(b, device)
            lab = labels_of(b, tf, margin_db)
            out = net(lab["feat"])
            ls = compute_losses(out, lab, cfg, uw)
            bs = lab["active"].shape[0]
            n_ex += bs
            for k in keys:
                acc[k] += float(ls[k]) * bs
            acc_by.setdefault(name, {k: 0.0 for k in keys})
            acc_by[name].setdefault("n", 0)
            for k in keys:
                acc_by[name][k] += float(ls[k]) * bs
            acc_by[name]["n"] += bs
            p = out["pitch"].float().cpu()
            y = lab["pitch"].float().cpu()
            m = lab["pitch_mask"].float().cpu()
            if m.dim() == 2:
                m = m[..., None].expand_as(p)
            pm = m > 0.5
            pitch_p.append(p[pm])
            pitch_y.append(y[pm])
            # "hard" is a diagnostic set: keep it out of the selection metrics so
            # the composite stays comparable with v1/v2.1 (proc + real only)
            sel = name != "hard"
            if sel:
                sel_pitch_p.append(p[pm])
                sel_pitch_y.append(y[pm])
                sel_dc_sum += float(ls["l_dc"]) * bs
                sel_n += bs
            if pm.any():
                batch_ap.append(pitch_ap(p, y, m)[0])
            a = act_iou(out["act"].float().cpu(), lab["active"].float().cpu())
            batch_iou.append(a["iou"])
            act_tp += a["tp"]; act_fp += a["fp"]; act_fn += a["fn"]
            if sel:
                sel_tp += a["tp"]; sel_fp += a["fp"]; sel_fn += a["fn"]
            if len(sweep_p) < 64:
                # every example contributes up to 4 frames, so the sweep covers
                # the whole fixed set (>=200 examples), not just the first batches
                k = min(4, p.shape[1])
                idx = torch.linspace(0, p.shape[1] - 1, k).long()
                sweep_p.append(p[:, idx].reshape(-1, p.shape[-1]))
                sweep_y.append(y[:, idx].reshape(-1, y.shape[-1]))
                sweep_m.append(m[:, idx].reshape(-1, m.shape[-1]))
            if cell_acc and len(emb_parts) < 24:
                emb_parts.append(out["emb"].float().cpu()[:2])
                dom_parts.append(lab["dom"].cpu()[:2])
                active_parts.append(lab["active"].float().cpu()[:2])
    res = {k: acc[k] / max(n_ex, 1) for k in keys}
    for name, d in acc_by.items():
        res[f"loss_{name}"] = d["loss"] / max(d["n"], 1)
    if pitch_p:
        P = torch.cat(pitch_p, 0)
        Y = torch.cat(pitch_y, 0)
        # AP over the fixed set; cells are strided uniformly when the set is huge
        if P.numel() > 4_000_000:
            stride = int(math.ceil(P.numel() / 4_000_000))
            P, Y = P[::stride], Y[::stride]
        M = torch.ones_like(P)
        ap, npos, ncells = pitch_ap(P, Y, M)
        res.update({"pitch_ap": ap, "pitch_pos": npos, "pitch_cells": ncells,
                    "pitch_stats": pitch_prob_stats(P, M)})
    else:
        res.update({"pitch_ap": float("nan"), "pitch_pos": 0, "pitch_cells": 0,
                    "pitch_stats": {"mean": 0.0, "max": 0.0, "frac_gt_0.5": 0.0, "q99": 0.0}})
    if sel_pitch_p:
        SP, SY = torch.cat(sel_pitch_p, 0), torch.cat(sel_pitch_y, 0)
        if SP.numel() > 4_000_000:
            stride = int(math.ceil(SP.numel() / 4_000_000))
            SP, SY = SP[::stride], SY[::stride]
        res["sel_pitch_ap"] = pitch_ap(SP, SY, torch.ones_like(SP))[0]
        res["sel_n_examples"] = sel_n
        res["sel_l_dc"] = sel_dc_sum / max(sel_n, 1)
        res["sel_act_iou"] = sel_tp / max(sel_tp + sel_fp + sel_fn, 1)
    res["act_iou"] = act_tp / max(act_tp + act_fp + act_fn, 1)
    res["act_tp"] = act_tp; res["act_fp"] = act_fp; res["act_fn"] = act_fn
    if cell_acc and emb_parts:
        E = torch.cat(emb_parts, 0)
        D = torch.cat(dom_parts, 0)
        A = torch.cat(active_parts, 0)
        F, Dd = E.shape[-2], E.shape[-1]
        res["cell_acc"] = cell_accuracy(E.reshape(-1, F, Dd), D.reshape(-1, F),
                                        A.reshape(-1, F), max_frames=max_cell_frames)
        res["emb_cos"] = embedding_cosine(E, A, max_samples=max_emb)
    else:
        res["cell_acc"] = float("nan")
        res["emb_cos"] = float("nan")
    if sweep_p:
        SP = torch.cat(sweep_p, 0).reshape(-1, 192)
        SY = torch.cat(sweep_y, 0).reshape(-1, 192)
        SM = torch.cat(sweep_m, 0).reshape(-1, 192)
        rows, best = pitch_f1_sweep(SP, SY, SM)
        res["pitch_f1"] = pitch_peak_f1(SP, SY, SM, 0.5)["f1"]
        res["pitch_f1_best"] = best["f1"]
        res["pitch_thr_best"] = best["threshold"]
        if sweep:
            res["pitch_sweep"] = rows
    else:
        res["pitch_f1"] = 0.0; res["pitch_f1_best"] = 0.0; res["pitch_thr_best"] = 0.5
    res["batch_noise"] = {
        "pitch_ap_std": float(np.nanstd(batch_ap)) if batch_ap else 0.0,
        "act_iou_std": float(np.std(batch_iou)) if batch_iou else 0.0,
    }
    res["n_examples"] = n_ex
    net.train()
    return res


def composite(val):
    """Checkpoint score: pitch AP + act IoU - 0.5 * l_dc (higher is better).

    Uses the selection metrics (proc + real) when they are present, so the
    diagnostic hard-case set cannot drive checkpoint choice.
    """
    ap = val.get("sel_pitch_ap", val.get("pitch_ap", 0.0))
    if ap is None or not math.isfinite(float(ap)):
        ap = 0.0
    dc = val.get("sel_l_dc", val.get("l_dc", 1.0))
    iou = val.get("sel_act_iou", val.get("act_iou", 0.0))
    return float(ap + iou - 0.5 * float(dc))


def head_grad_norms(net):
    """Gradient norm at each private head output (same input scale per head)."""
    grads = getattr(net, "_head_grads", None) or {}
    out = {}
    for name in ("emb", "act", "pitch", "sal"):
        t = grads.get(name)
        g = getattr(t, "grad", None) if t is not None else None
        out[f"g_{name}"] = float(g.norm()) if g is not None else 0.0
    return out
