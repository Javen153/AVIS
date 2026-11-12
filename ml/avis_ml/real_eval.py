"""Reference targets for MUSDB-test and Slakh-test clips, built from the isolated stems.

A clip is a real mixture; every source stem is measured through FE-v1 band power, so the
dominant source, salience and pitch targets use the same definitions as SPEC F. Drums are
dominant-capable but never pitch-labelled. Nothing here is used for training.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from .categories import MUSDB_STREAMS, slakh_category

SEC = 4.0  # matches the 4 s pyin windows (D-026)
FRAMES = int(SEC * 100)
SLOTS = 4
F_ROOT = Path(r"F:\Datasets")
SLAKH = F_ROOT / "slakh2100_flac_redux"
MUSDB = F_ROOT / "musdb18"


def resample_48(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 48 kHz and force exactly SEC seconds (clips must stack)."""
    from scipy.signal import resample_poly

    y = x.astype(np.float32, copy=False) if sr == 48000 else \
        resample_poly(x, 480, 441, axis=0).astype(np.float32)
    n = int(SEC * 48000)
    if len(y) < n:
        y = np.pad(y, ((0, n - len(y)), (0, 0)))
    y = y[:n]
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:, :2]


def _decode_mp4(mp4: str, idx: int, start: float, dur: float) -> np.ndarray:
    import av

    with av.open(mp4) as c:
        st = c.streams.audio[idx]
        c.seek(int(start / st.time_base), stream=st)
        chunks, got = [], 0
        for pkt in c.demux(st):
            for fr in pkt.decode():
                chunks.append(fr.to_ndarray())
                got += chunks[-1].shape[1]
                if got >= int(dur * st.rate):
                    break
            if got >= int(dur * st.rate):
                break
        arr = np.concatenate(chunks, axis=1).T[:, : int(dur * st.rate)]
        return arr.astype(np.float32)


def _f0_midi(label: Path, start: float) -> np.ndarray:
    d = np.load(label)
    out = np.zeros((FRAMES, SLOTS), dtype=np.float32)
    cnt = np.zeros(FRAMES, dtype=np.int32)
    for s, e, p in zip(d["start"], d["end"], d["pitch"]):
        a = max(0, int(np.ceil((float(s) - start) * 100)))
        b = min(FRAMES, int(np.ceil((float(e) - start) * 100)))
        if b <= a:
            continue
        rows = np.arange(a, b)
        rows = rows[cnt[rows] < SLOTS]
        if rows.size:
            out[rows, cnt[rows]] = 440.0 * 2.0 ** ((float(p) - 69) / 12.0)
            cnt[rows] += 1
    return out


def _f0_pyin(label: Path, start: float) -> np.ndarray:
    """Windowed pyin cache (D-026): nearest labelled 4 s window to `start`."""
    d = np.load(label)
    starts, f0s = d["starts"], d["f0"]
    out = np.zeros((FRAMES, SLOTS), dtype=np.float32)
    if starts.size == 0:
        return out
    i = int(np.argmin(np.abs(starts - start)))
    seg = f0s[i]
    out[: len(seg), 0] = seg
    return out


def pyin_window(track: str, role: str, start: float) -> tuple[float, np.ndarray]:
    """Snap a requested start to a cached pyin window; returns (start, f0)."""
    lab = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "labels" / "musdb" / f"{track}.{role}.npz"
    if not lab.exists():
        return start, None
    d = np.load(lab)
    if d["starts"].size == 0:
        return start, None
    i = int(np.argmin(np.abs(d["starts"] - start)))
    s = float(d["starts"][i])
    out = np.zeros((FRAMES, SLOTS), dtype=np.float32)
    out[:, 0] = d["f0"][i]
    return s, out


def musdb_clip(track: str, start: float):
    """Return {stem_name: (audio48 stereo, f0)} for a MUSDB test track."""
    raw = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "raw" / "musdb" / "test"
    start, _ = pyin_window(track, "vocals", start)
    out = {}
    for role, idx in MUSDB_STREAMS.items():
        if role == "mix":
            continue
        f = raw / f"{track}.{role}.flac"
        if f.exists():
            x, sr = sf.read(str(f), start=int(start * 44100), frames=int(SEC * 44100),
                            dtype="float32", always_2d=True)
        else:
            x = _decode_mp4(str(MUSDB / "test" / f"{track}.stem.mp4"), idx, start, SEC)
            sr = 44100
        if len(x) < int(SEC * 44100):
            x = np.pad(x, ((0, int(SEC * 44100) - len(x)), (0, 0)))
        f0 = None
        if role in ("vocals", "bass"):
            lab = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "labels" / "musdb" / f"{track}.{role}.npz"
            if lab.exists():
                f0 = _f0_pyin(lab, start)
        out[role] = (resample_48(x, sr), x, f0)
    return out


def slakh_clip(track_dir: Path, start: float):
    import yaml

    with open(track_dir / "metadata.yaml", encoding="utf-8") as f:
        md = yaml.safe_load(f)
    out = {}
    for sid, s in md["stems"].items():
        p = track_dir / "stems" / f"{sid}.flac"
        if not p.exists():
            continue
        x, sr = sf.read(str(p), start=int(start * 44100), frames=int(SEC * 44100),
                        dtype="float32", always_2d=True)
        if len(x) < int(SEC * 44100):
            x = np.pad(x, ((0, int(SEC * 44100) - len(x)), (0, 0)))
        f0 = None
        lab = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "labels" / "slakh" / track_dir.name / f"{sid}.npz"
        if not s.get("is_drum") and lab.exists():
            f0 = _f0_midi(lab, start)
        cat = "drums" if s.get("is_drum") else slakh_category(s.get("inst_class", ""), int(s.get("program_num", -1)))
        out[sid] = (resample_48(x, sr), x, f0, cat)
    return out


def clip_starts(entry: dict, n: int, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    segs = [s for s in entry["active_segments"] if s <= int(entry["duration"] - SEC)]
    if not segs:
        return []
    pick = rng.choice(segs, size=min(n, len(segs)), replace=False)
    return [float(s) for s in sorted(pick)]


def index_rows(split: str, dataset: str) -> list[dict]:
    p = Path(os.environ.get("AVIS_DATA", r"E:\avis_data")) / "index.jsonl"
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split and r["dataset"] == dataset:
                rows.append(r)
    return rows


def tracks(rows: list[dict], stem_id: str) -> dict[str, dict]:
    out = {}
    for r in rows:
        if r["stem_id"] == stem_id:
            out[r["track_id"]] = r
    return out


def slakh_track_dirs(split: str) -> dict[str, Path]:
    return {Path(d).name: Path(d) for d in sorted(glob.glob(str(SLAKH / split / "Track*")))}


def mixdown(stems: dict, keys=None) -> np.ndarray:
    keys = keys or list(stems)
    mix = np.zeros((int(SEC * 48000), 2), dtype=np.float32)
    for k in keys:
        a = stems[k][0]
        mix += a
    return mix


def targets(stems: dict, tf, keys=None, cat_of=None):
    """Measured dominant/salience/pitch targets from the isolated stems (SPEC F)."""
    import torch

    keys = keys or list(stems)
    aud = torch.from_numpy(np.stack([stems[k][0] for k in keys])).to(tf.device)
    P = tf.band_power(aud)["M"]
    tot = P.sum(0) + 1e-8
    pdom = P.max(0).values.clamp_min(1e-9)
    rest = (tot - pdom).clamp_min(1e-9)
    dom = P.argmax(0).long()
    sal = torch.sigmoid((10 * torch.log10(pdom) - 10 * torch.log10(rest)) / 6.0)
    f0s = [stems[k][2] for k in keys]
    lfc = torch.log(tf.fc)
    pitch = torch.zeros(1, FRAMES, 192, device=tf.device)
    for f0 in f0s:
        if f0 is None:
            continue
        ft = torch.from_numpy(f0).to(tf.device)
        g = torch.exp(-0.5 * ((lfc[None, None, :] - torch.log(ft.clamp_min(1.0))[..., None]) / 0.03) ** 2)
        g = g * (ft > 0).float()[..., None]
        pitch = torch.maximum(pitch, g.amax(dim=1)[None])
    return dom[None], pitch.clamp(0, 1), sal[None]
