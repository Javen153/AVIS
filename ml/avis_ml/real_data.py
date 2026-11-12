"""Real-source pool: index-driven 4 s chunks read with soundfile seek in DataLoader workers.

Each chunk is mono 44.1 kHz audio plus a pitch track f0 [400, 4] (Hz, 0 = none) on the
10 ms grid. Pitch comes from pyin (MUSDB vocals/bass) or from MIDI (Slakh); drums are
never pitch-labelled. Resampling to 48 kHz happens on the GPU in mixer_v2.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .categories import PITCHED

SEC = 4.0
FRAMES = 400  # 10 ms grid over SEC
SLOTS = 4  # simultaneous pitches kept per frame

# Real stems are used for pitch only when the dataset guarantees a note (D-035).
# Sound Effects, Percussive and other_tonal stems still carry MIDI notes in Slakh,
# but those notes do not describe a reliable fundamental, so they are excluded
# from the real pool instead of being trained as negative pitch.
PITCH_LABELLED = PITCHED - {"fx", "other_tonal"}

# Category draw weights for the loader. Vocal/bass get the most draws because the
# mixer needs them in every non-procedural example (SPEC F: all vocal/bass examples real).
DEFAULT_WEIGHTS = {
    "vocal_lead": 0.2, "bass": 0.2, "guitar": 0.07, "piano": 0.07, "keys": 0.05,
    "synth_lead": 0.05, "synth_pad": 0.05, "strings": 0.07, "brass": 0.05,
    "winds": 0.05, "organ": 0.05,
}


def load_pool(index_path: Path, splits: tuple[str, ...]) -> dict[str, list[dict]]:
    """Group usable index rows by category (labels must exist; only active stems)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] not in splits or r["category"] not in PITCH_LABELLED:
                continue
            if not r["active_segments"] or not Path(r["label"]).exists():
                continue
            by_cat[r["category"]].append(r)
    return dict(by_cat)


def _read_mono(path: str, sr: int, start: float) -> np.ndarray:
    x, _ = sf.read(path, start=int(round(start * sr)), frames=int(round(SEC * sr)),
                   dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    n = int(round(SEC * sr))
    if len(mono) < n:
        mono = np.pad(mono, (0, n - len(mono)))
    return mono


def _read_stereo(path: str, sr: int, start: float) -> np.ndarray:
    """Stereo chunk [n,2]: real L/R content is preserved for M-v3 training."""
    x, _ = sf.read(path, start=int(round(start * sr)), frames=int(round(SEC * sr)),
                   dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    n = int(round(SEC * sr))
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    return x


def _f0_from_pyin(label: str, start: float) -> np.ndarray:
    d = np.load(label)
    starts, f0s = d["starts"], d["f0"]
    out = np.zeros((FRAMES, SLOTS), dtype=np.float32)
    if starts.size == 0:
        return out
    i = int(np.argmin(np.abs(starts - start)))
    if abs(float(starts[i]) - start) > 0.25:
        return out
    out[:, 0] = f0s[i]
    return out


def pyin_starts(label: str) -> np.ndarray:
    """Windows with cached pyin labels (MUSDB). Empty when the cache is missing."""
    try:
        return np.load(label)["starts"]
    except (FileNotFoundError, KeyError):
        return np.zeros(0, dtype=np.float32)


def _f0_from_midi(label: str, start: float) -> np.ndarray:
    d = np.load(label)
    ns, ne, pit = d["start"], d["end"], d["pitch"]
    out = np.zeros((FRAMES, SLOTS), dtype=np.float32)
    cnt = np.zeros(FRAMES, dtype=np.int32)
    keep = (ne > start) & (ns < start + SEC)
    for s, e, p in zip(ns[keep], ne[keep], pit[keep]):
        a = max(0, int(np.ceil((s - start) * 100)))
        b = min(FRAMES, int(np.ceil((e - start) * 100)))
        if b <= a:
            continue
        rows = np.arange(a, b)
        ok = cnt[rows] < SLOTS
        rows = rows[ok]
        if rows.size == 0:
            continue
        out[rows, cnt[rows]] = 440.0 * 2.0 ** ((p - 69) / 12.0)
        cnt[rows] += 1
    return out


def read_chunk(entry: dict, start: float) -> tuple[np.ndarray, np.ndarray]:
    audio = _read_mono(entry["path"], int(entry["sr"]), start)
    if entry["dataset"] == "musdb":
        f0 = _f0_from_pyin(entry["label"], start)
    else:
        f0 = _f0_from_midi(entry["label"], start)
    return audio, f0


def read_chunk_stereo(entry: dict, start: float) -> tuple[np.ndarray, np.ndarray]:
    """Same as read_chunk but keeps [n,2]; used by the M-v3 stereo mixer."""
    audio = _read_stereo(entry["path"], int(entry["sr"]), start)
    if entry["dataset"] == "musdb":
        f0 = _f0_from_pyin(entry["label"], start)
    else:
        f0 = _f0_from_midi(entry["label"], start)
    return audio, f0


class ChunkDataset(IterableDataset):
    """Infinite stream of (audio, f0, cat) chunks; category drawn by DEFAULT_WEIGHTS."""

    def __init__(self, pool: dict[str, list[dict]], cats: list[str], weights: list[float],
                 seed: int = 0, stereo: bool = False):
        self.pool = pool
        self.cats = cats
        self.weights = weights
        self.seed = seed
        self.stereo = stereo

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(self.seed * 1000 + wid)
        while True:
            ci = rng.choices(range(len(self.cats)), weights=self.weights)[0]
            cat = self.cats[ci]
            e = self._pick(rng, cat)
            start = self._start(rng, e)
            if self.stereo:
                audio, f0 = read_chunk_stereo(e, start)
            else:
                audio, f0 = read_chunk(e, start)
            yield {"audio": torch.from_numpy(audio), "f0": torch.from_numpy(f0),
                   "cat": torch.tensor(ci, dtype=torch.long)}

    def _start(self, rng, e):
        if e["dataset"] == "musdb":
            starts = pyin_starts(e["label"])
            if starts.size:
                return float(rng.choice(starts.tolist()))
        s = rng.choice(e["active_segments"]) + rng.random()
        return min(max(0.0, s), max(0.0, e["duration"] - SEC))

    def _pick(self, rng, cat):
        rows = self.pool[cat]
        if cat == "bass":
            # half MUSDB, half Slakh: MUSDB has only 100 train bass stems
            musdb = [r for r in rows if r["dataset"] == "musdb"]
            slakh = [r for r in rows if r["dataset"] == "slakh"]
            if musdb and (rng.random() < 0.5 or not slakh):
                return rng.choice(musdb)
            return rng.choice(slakh)
        return rng.choice(rows)


class RealSourceSampler:
    """Hands out real chunks per category; refills from a persistent DataLoader."""

    def __init__(self, pool, weights=None, batch: int = 16, workers: int = 4, seed: int = 0,
                 cap: int = 64, stereo: bool = False):
        cats = [c for c in (weights or DEFAULT_WEIGHTS) if c in pool]
        w = [(weights or DEFAULT_WEIGHTS)[c] for c in cats]
        self.cats = cats
        self.queues: dict[str, deque] = {c: deque() for c in cats}
        self.cap = cap
        self.stereo = stereo
        ds = ChunkDataset(pool, cats, w, seed, stereo=stereo)
        self.loader = DataLoader(ds, batch_size=batch, num_workers=workers,
                                 persistent_workers=workers > 0, prefetch_factor=2 if workers else None)
        self._it = iter(self.loader)
        self.served = defaultdict(int)

    def has(self, cat: str) -> bool:
        return cat in self.queues

    def take(self, cat: str):
        q = self.queues[cat]
        while not q:
            b = next(self._it)
            for i in range(b["cat"].shape[0]):
                c = self.cats[int(b["cat"][i])]
                if len(self.queues[c]) < self.cap:
                    self.queues[c].append((b["audio"][i].numpy(), b["f0"][i].numpy()))
        self.served[cat] += 1
        return q.popleft()
