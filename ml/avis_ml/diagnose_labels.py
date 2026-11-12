"""Label audit: compare Slakh MIDI fundamentals with the spectral peak through FE-v1.

This is diagnostic only. It does not change training labels; it answers whether
bass octave ambiguity comes from the dataset labels or from the timbre.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .data_index import data_root
from .frontend import HOP, SR, TorchFrontend, band_centers
from .real_eval import index_rows
from .real_data import read_chunk

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _peak_freq(logp: np.ndarray, lo: int, hi: int) -> float:
    """Parabolic interpolation of the band-power peak in [lo,hi]."""
    fc = band_centers()
    seg = logp[lo:hi]
    i = int(np.argmax(seg)) + lo
    if i <= 0 or i >= len(logp) - 1:
        return float(fc[i])
    a, b, c = logp[i - 1], logp[i], logp[i + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    d = float(np.clip(d, -0.5, 0.5))
    logf = np.log(fc[i]) + d * (np.log(fc[i + 1]) - np.log(fc[i - 1])) / 2.0
    return float(math.exp(logf))


def _summary(rows):
    if not rows:
        return {"n": 0}
    c = np.array([r["cents"] for r in rows], dtype=np.float64)
    return {"n": int(len(c)), "median_cents": float(np.median(c)),
            "median_abs_cents": float(np.median(np.abs(c))),
            "frac_within_50c": float((np.abs(c) <= 50).mean()),
            "frac_octave": float(((np.abs(c) >= 1000) & (np.abs(c) <= 1400)).mean()),
            "frac_sub_octave": float((c <= -1000).mean()),
            "frac_super_octave": float((c >= 1000).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-cat", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "diag" / "labels_slakh.json"))
    a = ap.parse_args()
    tf = TorchFrontend(REPO / "shared" / "frontend" / "v1" / "weights.bin", a.device)
    rows = index_rows(a.split, "slakh")
    by_cat = {}
    for r in rows:
        if r["active_segments"]:
            by_cat.setdefault(r["category"], []).append(r)
    out = {"split": a.split, "categories": {}}
    fc = band_centers()
    for cat, items in sorted(by_cat.items()):
        recs = []
        for r in items[: a.per_cat]:
            start = float(r["active_segments"][0]) + 0.5
            audio, f0 = read_chunk(r, start)
            y = resample_poly(audio, 480, 441).astype(np.float32)
            n = 4 * SR
            if len(y) < n:
                y = np.pad(y, (0, n - len(y)))
            x = torch.from_numpy(np.stack([y[:n], y[:n]], axis=1))[None].to(a.device)
            with torch.no_grad():
                _, p = tf.features(x)
            db = (10 * torch.log10(p["M"][0] + 1e-12)).cpu().numpy()
            for t in range(min(400, f0.shape[0])):
                note = float(f0[t, 0])
                if note <= 0:
                    continue
                expected = 1200.0 * math.log2(float(fc[0]) / note)
                band = int(np.argmin(np.abs(np.log(fc) - np.log(note))))
                lo = max(0, band - 4)
                hi = min(len(fc), band + 5)
                local = _peak_freq(db[t], lo, hi)
                # global peak below 2 kHz, which is where bass fundamentals live
                ghi = int(np.searchsorted(fc, 2000.0))
                glob = _peak_freq(db[t], 0, max(2, ghi))
                cents_local = 1200.0 * math.log2(local / note)
                cents_glob = 1200.0 * math.log2(glob / note)
                recs.append({"cents": cents_local, "cents_global": cents_glob,
                             "note_hz": note, "band": band,
                             "local_hz": local, "global_hz": glob})
        if recs:
            out["categories"][cat] = {
                "rows": len(recs),
                "local": _summary(recs),
                "global": _summary([{"cents": r["cents_global"]} for r in recs]),
            }
            print(cat, json.dumps(out["categories"][cat]["local"]), flush=True)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
