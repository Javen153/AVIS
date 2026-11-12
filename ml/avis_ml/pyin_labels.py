"""Pitch labels for MUSDB vocals/bass via librosa.pyin (SPEC F: fmin 40, fmax 1600).

pyin runs at ~1.2x realtime, so a whole MUSDB corpus would need ~17 h. Instead each stem
gets a fixed set of 4 s windows drawn from its active seconds; the sampler only reads
those windows (D-026). Cache: labels/musdb/<track>.<role>.npz with
  starts [W] (s), f0 [W,400] (Hz, 0 = unvoiced), prob [W,400].
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .data_index import data_root

FMIN, FMAX = 40.0, 1600.0
HOP = 441
SEC = 4.0
FRAMES = 400


def pick_windows(active: list[int], duration: float, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    segs = [s for s in active if s <= duration - SEC]
    if not segs:
        return []
    take = min(n, len(segs))
    pick = rng.choice(segs, size=take, replace=False)
    return sorted(float(s) + float(rng.random()) for s in pick)


def job(args) -> str:
    path, label, starts, cat = args
    if Path(label).exists():
        return "skip"
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    f0s, probs = [], []
    for s in starts:
        i = int(round(s * sr))
        seg = mono[i:i + int(SEC * sr)]
        if len(seg) < int(SEC * sr):
            seg = np.pad(seg, (0, int(SEC * sr) - len(seg)))
        f0, voiced, prob = librosa.pyin(seg, fmin=FMIN, fmax=FMAX, sr=sr,
                                        frame_length=2048, hop_length=HOP)
        f0 = np.where(voiced & np.isfinite(f0), f0, 0.0).astype(np.float32)
        f0s.append(f0[:FRAMES] if len(f0) >= FRAMES else np.pad(f0, (0, FRAMES - len(f0))))
        probs.append(np.asarray(prob[:FRAMES], dtype=np.float32))
    out = Path(label)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, starts=np.asarray(starts, np.float32), f0=np.stack(f0s),
             prob=np.stack(probs))
    return f"{Path(path).name} windows={len(starts)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="", help="test|train (default: all)")
    args = ap.parse_args()
    root = data_root()
    index = root / "index.jsonl"
    labels = root / "labels"
    rows = []
    with open(index, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "musdb" or r["stem_id"] == "mix":
                continue
            if args.split and r["split"] != args.split:
                continue
            seed = zlib.crc32(f"{r['track_id']}.{r['stem_id']}".encode()) & 0x7FFFFFFF
            starts = pick_windows(r["active_segments"], r["duration"], args.windows, seed=seed)
            if not starts:
                continue
            lab = labels / "musdb" / f"{r['track_id']}.{r['stem_id']}.npz"
            rows.append((r["path"], str(lab), starts, r["category"]))
    # test first: evaluation labels are needed before training finishes
    rows.sort(key=lambda t: ("\\test\\" not in t[0] and "/test/" not in t[0], t[0]))
    print(f"files={len(rows)} windows_each={args.windows} workers={args.workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, msg in enumerate(ex.map(job, rows)):
            if i % 10 == 0:
                print(f"[{i+1}/{len(rows)}] {msg} min={(time.time()-t0)/60:.1f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
