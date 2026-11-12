"""Evaluation: oracle-K cell accuracy (Hungarian) and pitch F1 on procedural data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .frontend import TorchFrontend
from .mixer import MixConfig, ProceduralMixer, make_labels
from .model import AvisNet

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def kmeans(x, k, iters=25):
    if k <= 1:
        return np.zeros(len(x), dtype=np.int64)
    rng = np.random.default_rng(0)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    lab = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab


def eval_batch(out, active, dom, pitch, pred_pitch, pred_act, pitch_mask=None):
    emb = out["emb"].cpu().numpy()
    act = active.cpu().numpy()
    dom = dom.cpu().numpy()
    B, T, F, D = emb.shape
    if pitch_mask is None:
        pitch_mask = torch.ones(B, T, device=active.device)
    pm = pitch_mask.cpu().numpy() > 0.5
    correct = total = 0
    p_tp = p_fp = p_fn = 0
    act_tp = act_fp = act_fn = 0
    for b in range(B):
        for t in range(T):
            m = act[b, t] > 0.5
            if m.sum() < 2:
                continue
            x = emb[b, t, m]
            y = dom[b, t, m]
            k = len(np.unique(y))
            if k < 2:
                continue
            lab = kmeans(x, k)
            # Hungarian: clusters -> source ids
            C = np.zeros((k, int(y.max()) + 1), dtype=np.int64)
            for i in range(k):
                for s in range(C.shape[1]):
                    C[i, s] = np.sum((lab == i) & (y == s))
            r, c = linear_sum_assignment(-C)
            mp = {int(rr): int(cc) for rr, cc in zip(r, c)}
            pred = np.array([mp.get(int(l), -1) for l in lab])
            correct += int((pred == y).sum())
            total += len(y)
            # pitch: local maxima above 0.5 vs gaussian truth
            if pm[b, t]:
                pp = pred_pitch[b, t].cpu().numpy()
                tp = pitch[b, t].cpu().numpy()
                pk = [i for i in range(1, F - 1) if pp[i] > 0.5 and pp[i] >= pp[i - 1] and pp[i] > pp[i + 1]]
                truek = [i for i in range(1, F - 1) if tp[i] > 0.5 and tp[i] >= tp[i - 1] and tp[i] > tp[i + 1]]
                used = set()
                for i in pk:
                    hit = [j for j in truek if abs(i - j) <= 1 and j not in used]
                    if hit:
                        p_tp += 1; used.add(hit[0])
                    else:
                        p_fp += 1
                p_fn += len(truek) - len(used)
            pa = pred_act[b, t].cpu().numpy() > 0.5
            act_tp += int((pa & m).sum()); act_fp += int((pa & ~m).sum()); act_fn += int((~pa & m).sum())
    cell = correct / max(total, 1)
    prec = p_tp / max(p_tp + p_fp, 1)
    rec = p_tp / max(p_tp + p_fn, 1)
    pf = 2 * prec * rec / max(prec + rec, 1e-9)
    aiou = act_tp / max(act_tp + act_fp + act_fn, 1)
    return {"cell_acc": cell, "pitch_precision": prec, "pitch_recall": rec,
            "pitch_f1": pf, "act_iou": aiou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hid = int(ck.get("config", {}).get("hid", 384))
    net = AvisNet(hid=hid); net.load_state_dict(ck["model"]); net.eval().to(dev)
    tf = TorchFrontend(REPO / "shared" / "frontend" / "v1" / "weights.bin", dev)
    mixer = ProceduralMixer(MixConfig(seconds=float(ck.get("config", {}).get("seconds", 4.0))), dev)
    acc = {}
    with torch.no_grad():
        for i in range(a.batches):
            b = mixer.batch(2, seed=500 + i)
            lab = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], tf,
                              margin=b.get("margin"), pitch_mask=b.get("pitch_mask"))
            out = net(lab["feat"])
            r = eval_batch(out, lab["active"], lab["dom"], lab["pitch"], out["pitch"],
                           out["act"], lab["pitch_mask"])
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
