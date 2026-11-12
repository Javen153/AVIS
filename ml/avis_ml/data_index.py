"""Build the AVIS_DATA index from Slakh2100-redux and MUSDB18-HQ (sources are read-only).

Writes <AVIS_DATA>/index.jsonl (one stem or mix per line), a MIDI note cache for Slakh
(labels/slakh/<track>/<stem>.npz), and FLAC copies of the MUSDB stems we use
(raw/musdb/<split>/*.flac, 24-bit). Nothing is written under F:.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .categories import MUSDB_STREAMS, musdb_category, slakh_category

F_ROOT = Path(r"F:\Datasets")
SLAKH = F_ROOT / "slakh2100_flac_redux"
MUSDB = F_ROOT / "musdb18"
ACTIVE_DB = -50.0
SLAKH_SPLITS = ["train", "validation", "test"]


def data_root() -> Path:
    return Path(os.environ.get("AVIS_DATA", r"E:\avis_data"))


def active_seconds(x: np.ndarray, sr: int) -> list[int]:
    """Indices of 1-second windows whose RMS exceeds -50 dBFS (x mono or [n,2])."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = len(mono) // sr
    if n == 0:
        return []
    rms = np.sqrt(np.mean(mono[: n * sr].reshape(n, sr).astype(np.float64) ** 2, axis=1))
    return [int(i) for i in np.nonzero(rms > 10 ** (ACTIVE_DB / 20))[0]]


def save_midi_notes(midi_path: Path, out: Path) -> int:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    start = np.array([n.start for n in notes], dtype=np.float32)
    end = np.array([n.end for n in notes], dtype=np.float32)
    pitch = np.array([n.pitch for n in notes], dtype=np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, start=start, end=end, pitch=pitch)
    return len(notes)


def _slakh_track(tpath: Path, split: str, labels_dir: Path) -> list[dict]:
    track = tpath.name
    with open(tpath / "metadata.yaml", encoding="utf-8") as f:
        md = yaml.safe_load(f)
    rows = []
    for sid, s in md["stems"].items():
        if s.get("is_drum"):
            continue
        stem_path = tpath / "stems" / f"{sid}.flac"
        if not stem_path.exists():
            continue
        prog = int(s.get("program_num", -1))
        cat = slakh_category(s.get("inst_class", ""), prog)
        info = sf.info(str(stem_path))
        x, sr = sf.read(str(stem_path), dtype="float32")
        lab = labels_dir / "slakh" / track / f"{sid}.npz"
        n_notes = save_midi_notes(tpath / "MIDI" / f"{sid}.mid", lab)
        rows.append({
            "path": str(stem_path), "dataset": "slakh", "track_id": track, "stem_id": sid,
            "split": split, "category": cat, "program": prog, "sr": info.samplerate,
            "channels": info.channels, "duration": round(info.frames / info.samplerate, 3),
            "active_segments": active_seconds(x, sr), "label": str(lab), "n_notes": n_notes,
        })
    return rows


def slakh_entries(split: str, labels_dir: Path, limit: int = 0, threads: int = 4):
    from concurrent.futures import ThreadPoolExecutor

    tracks = sorted(glob.glob(str(SLAKH / split / "Track*")))
    if limit:
        tracks = tracks[:limit]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for rows in ex.map(lambda d: _slakh_track(Path(d), split, labels_dir), tracks):
            yield from rows


def _decode_stream(mp4: str, idx: int) -> np.ndarray:
    import av

    chunks = []
    with av.open(mp4) as c:
        st = c.streams.audio[idx]
        sr = st.rate
        for pkt in c.demux(st):
            for fr in pkt.decode():
                chunks.append(fr.to_ndarray())
    arr = np.concatenate(chunks, axis=1).T
    return arr.astype(np.float32), sr


def musdb_entries(split: str, roles: list[str], raw_dir: Path):
    for mp4 in sorted(glob.glob(str(MUSDB / split / "*.stem.mp4"))):
        name = Path(mp4).name[: -len(".stem.mp4")]
        for role in roles:
            cat = "mix" if role == "mix" else musdb_category(role)
            if cat is None:
                continue
            out = raw_dir / split / f"{name}.{role}.flac"
            if not out.exists():
                x, sr = _decode_stream(mp4, MUSDB_STREAMS[role])
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out), x, sr, subtype="PCM_24")
            x, sr = sf.read(str(out), dtype="float32")
            info = sf.info(str(out))
            yield {
                "path": str(out), "dataset": "musdb", "track_id": name, "stem_id": role,
                "split": split, "category": cat, "program": -1, "sr": info.samplerate,
                "channels": info.channels, "duration": round(info.frames / info.samplerate, 3),
                "active_segments": active_seconds(x, sr), "source_mp4": mp4,
                "label": str(data_root() / "labels" / "musdb" / f"{name}.{role}.npz"),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-musdb", action="store_true")
    ap.add_argument("--skip-slakh", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Slakh tracks per split (0 = all)")
    args = ap.parse_args()
    root = data_root()
    labels_dir = root / "labels"
    raw_dir = root / "raw" / "musdb"
    out_path = Path(args.out) if args.out else root / "index.jsonl"
    tmp = out_path.with_suffix(".tmp")
    n = 0
    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        if not args.skip_musdb:
            for split, roles in [("train", ["vocals", "bass"]), ("test", ["mix", "vocals", "bass"])]:
                for e in musdb_entries(split, roles, raw_dir):
                    f.write(json.dumps(e) + "\n")
                    n += 1
                    print(f"musdb {split} {e['track_id']} {e['stem_id']} dur={e['duration']}", flush=True)
        if not args.skip_slakh:
            for split in SLAKH_SPLITS:
                for e in slakh_entries(split, labels_dir, args.limit):
                    f.write(json.dumps(e) + "\n")
                    n += 1
                    if n % 200 == 0:
                        print(f"entries={n} elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    tmp.replace(out_path)
    print(f"done entries={n} -> {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
