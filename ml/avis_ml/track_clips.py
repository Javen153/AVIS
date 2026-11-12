"""Real held-out tracking clips: sums of 2-3 isolated stems with a known source count.

Clipping is 10 s so ID switches and re-identification have room. The expected object
count is the number of audible stems (+-1 tolerance in the metric); this is a proxy
because a real drum stem can legitimately split into several objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .real_eval import SEC as EVAL_SEC  # noqa: F401  (keeps the 4 s eval contract visible)
from .real_eval import MUSDB, MUSDB_STREAMS, _decode_mp4, index_rows, slakh_track_dirs, tracks

CLIP_SEC = 10.0
SR = 44100


def _stem_db(x: np.ndarray) -> float:
    return 20 * float(np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def _read(path: Path, start: float, dur: float) -> np.ndarray:
    sr = sf.info(str(path)).samplerate
    x, _ = sf.read(str(path), start=int(round(start * sr)), frames=int(round(dur * sr)),
                   dtype="float32", always_2d=True)
    if len(x) < int(dur * sr):
        x = np.pad(x, ((0, int(dur * sr) - len(x)), (0, 0)))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak > 0.98:
        x = x / peak * 0.98
    sf.write(str(path), x.astype(np.float32), 48000, subtype="PCM_16")


def _resample(x: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    y = resample_poly(x, 480, 441, axis=0).astype(np.float32)
    n = int(CLIP_SEC * 48000)
    if len(y) < n:
        y = np.pad(y, ((0, n - len(y)), (0, 0)))
    return y[:n]


def musdb_clips(n: int, out_dir: Path) -> list[dict]:
    import os

    raw = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "raw" / "musdb" / "test"
    rows = tracks(index_rows("test", "musdb"), "mix")
    clips = []
    for i, name in enumerate(sorted(rows)[:n]):
        start = 30.0 + 20.0 * (i % 3)
        stems = {}
        for role in ("vocals", "bass", "drums"):
            f = raw / f"{name}.{role}.flac"
            if f.exists():
                stems[role] = _read(f, start, CLIP_SEC)
            else:
                stems[role] = _decode_mp4(str(MUSDB / "test" / f"{name}.stem.mp4"),
                                          MUSDB_STREAMS[role], start, CLIP_SEC)
        keep = [k for k in ("vocals", "bass") if _stem_db(stems[k]) > -45]
        if len(keep) < 2:
            continue
        x = sum(stems[k] for k in keep)
        p = out_dir / f"musdb_{i:02d}.wav"
        write_wav(p, _resample(x))
        clips.append({"dataset": "musdb", "track": name, "start": start, "wav": str(p),
                      "sources": len(keep)})
    return clips


def slakh_clips(n: int, out_dir: Path) -> list[dict]:
    import yaml

    dirs = slakh_track_dirs("test")
    clips = []
    k = 0
    for name in sorted(dirs):
        if k >= n:
            break
        tdir = dirs[name]
        with open(tdir / "metadata.yaml", encoding="utf-8") as f:
            md = yaml.safe_load(f)
        start = 30.0 + 10.0 * (k % 2)
        stems = {}
        for sid, s in md["stems"].items():
            p = tdir / "stems" / f"{sid}.flac"
            if p.exists() and not s.get("is_drum"):
                x = _read(p, start, CLIP_SEC)
                if _stem_db(x) > -45:
                    stems[sid] = x
        if len(stems) < 2:
            continue
        keep = sorted(stems)[:3]
        x = sum(stems[s] for s in keep)
        p = out_dir / f"slakh_{k:02d}.wav"
        write_wav(p, _resample(x))
        clips.append({"dataset": "slakh", "track": name, "start": start, "wav": str(p),
                      "sources": len(keep), "stems": keep})
        k += 1
    return clips
