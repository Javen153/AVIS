"""Validation metrics: pitch AP / threshold sweep, activity IoU, cluster accuracy.

These are deliberately independent of the loss functions: training selects a
checkpoint by the composite metric built here, not by the sum of losses.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


def _flat(prob, target, mask):
    p = prob.detach().float().cpu().numpy() if hasattr(prob, "detach") else np.asarray(prob)
    y = target.detach().float().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
    m = mask.detach().float().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    m = m > 0.5
    return p[m], y[m]


def pitch_ap(prob, target, mask):
    """Average precision of band pitch probabilities over supervised cells.

    Positive = target gaussian >= 0.5. Returns (ap, n_pos, n_cells); ap is NaN
    when a split has no positive or no negative cells.
    """
    p, y = _flat(prob, target, mask)
    yb = (y >= 0.5).astype(np.int8)
    npos = int(yb.sum())
    if npos == 0 or npos == len(yb):
        return float("nan"), npos, int(len(yb))
    return float(average_precision_score(yb, p)), npos, int(len(yb))


def pitch_prob_stats(prob, mask):
    """Collapse diagnostics for the pitch head: fraction above 0.5 and maximum."""
    p, _ = _flat(prob, prob, mask)
    if p.size == 0:
        return {"mean": 0.0, "max": 0.0, "frac_gt_0.5": 0.0, "q99": 0.0}
    return {"mean": float(p.mean()), "max": float(p.max()),
            "frac_gt_0.5": float((p > 0.5).mean()), "q99": float(np.quantile(p, 0.99))}


def _peaks(row, thr):
    """Local maxima strictly above thr, endpoints included when they stand out."""
    out = []
    n = len(row)
    for i in range(n):
        v = row[i]
        if v < thr:
            continue
        left = row[i - 1] if i > 0 else -1.0
        right = row[i + 1] if i + 1 < n else -1.0
        if v >= left and v > right:
            out.append(i)
    return out


def pitch_peak_f1(prob, target, mask, threshold=0.5):
    """Peak-matching F1 (same definition as eval.py): +-1 band tolerance."""
    P = prob.detach().float().cpu().numpy() if hasattr(prob, "detach") else np.asarray(prob)
    Y = target.detach().float().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
    M = mask.detach().float().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    M = M > 0.5
    tp = fp = fn = 0
    for t in range(P.shape[0]):
        if not M[t].any():
            continue
        true = _peaks(np.where(M[t], Y[t], 0.0), 0.5)
        pred = _peaks(np.where(M[t], P[t], 0.0), float(threshold))
        used = set()
        for i in pred:
            hit = [j for j in true if abs(i - j) <= 1 and j not in used]
            if hit:
                tp += 1
                used.add(hit[0])
            else:
                fp += 1
        fn += len(true) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"f1": 2 * prec * rec / max(prec + rec, 1e-9), "precision": prec,
            "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def pitch_f1_sweep(prob, target, mask, thresholds=None):
    th = thresholds if thresholds is not None else np.arange(0.05, 0.91, 0.05)
    rows = []
    for t in th:
        r = pitch_peak_f1(prob, target, mask, float(t))
        rows.append({"threshold": float(t), **r})
    best = max(rows, key=lambda r: r["f1"])
    return rows, best


def act_iou(pred, active):
    p = (pred.detach().float().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)) > 0.5
    a = (active.detach().float().cpu().numpy() if hasattr(active, "detach") else np.asarray(active)) > 0.5
    tp = int((p & a).sum())
    fp = int((p & ~a).sum())
    fn = int((~p & a).sum())
    return {"iou": tp / max(tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def kmeans(x, k, iters=20):
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


def cell_accuracy(emb, dom, active, max_frames=1500):
    """Oracle-K Hungarian cell accuracy; frames are subsampled deterministically."""
    E = emb.detach().float().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
    D = dom.detach().cpu().numpy() if hasattr(dom, "detach") else np.asarray(dom)
    A = active.detach().float().cpu().numpy() if hasattr(active, "detach") else np.asarray(active)
    correct = total = 0
    idx = np.linspace(0, max(0, E.shape[0] - 1), min(max_frames, E.shape[0])).astype(int)
    for t in idx:
        m = A[t] > 0.5
        if m.sum() < 2:
            continue
        x = E[t, m]
        y = D[t, m].astype(np.int64)
        k = len(np.unique(y))
        if k < 2:
            continue
        lab = kmeans(x, k)
        C = np.zeros((k, int(y.max()) + 1), dtype=np.int64)
        for i in range(k):
            for s in range(C.shape[1]):
                C[i, s] = np.sum((lab == i) & (y == s))
        r, c = linear_sum_assignment(-C)
        mp = {int(rr): int(cc) for rr, cc in zip(r, c)}
        pred = np.array([mp.get(int(l), -1) for l in lab])
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)


def embedding_cosine(emb, active, max_samples=50000, seed=0):
    """Mean pairwise cosine similarity of active-band embeddings.

    A value rising toward 1 means representation collapse (all bands equal).
    """
    E = emb.detach().float().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
    A = active.detach().float().cpu().numpy() if hasattr(active, "detach") else np.asarray(active)
    flat = E.reshape(-1, E.shape[-1])[A.reshape(-1) > 0.5]
    if flat.shape[0] < 4:
        return float("nan")
    if flat.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.shape[0], max_samples, replace=False)]
    n = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
    sim = (flat / n) @ (flat / n).T
    k = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / max(k * (k - 1), 1))
