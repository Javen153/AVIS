"""Real+procedural mixer: shapes, no NaN, measured dominance on a tiny synthetic pool."""
import numpy as np
import pytest
import soundfile as sf
import torch

from avis_ml.frontend import TorchFrontend
from avis_ml.mixer import MixConfig, ProceduralMixer, make_labels
from avis_ml.mixer_v2 import RealMixer
from avis_ml.real_data import FRAMES, SLOTS, RealSourceSampler, load_pool

SR = 44100
SEC = 6.0

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for speed")


def _mk(tmp_path):
    rng = np.random.default_rng(0)
    idx = tmp_path / "index.jsonl"
    rows = []
    for cat, hz in [("vocal_lead", 220.0), ("bass", 55.0)]:
        p = tmp_path / f"{cat}.flac"
        t = np.arange(int(SR * SEC)) / SR
        x = (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        sf.write(str(p), x, SR, subtype="PCM_16")
        lab = tmp_path / f"{cat}.npz"
        np.savez(lab, starts=np.array([0.0, 1.0, 2.0], np.float32),
                 f0=np.full((3, int(4 * 100)), hz, dtype=np.float32),
                 prob=np.ones((3, int(4 * 100)), dtype=np.float32))
        rows.append({"path": str(p), "dataset": "musdb", "track_id": cat, "stem_id": cat,
                     "split": "train", "category": cat, "sr": SR, "channels": 1,
                     "duration": SEC, "active_segments": [0, 1, 2], "label": str(lab)})
    idx.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")
    return idx


def test_real_mixer_shapes_and_labels(tmp_path):
    idx = _mk(tmp_path)
    pool = load_pool(idx, ("train",))
    assert set(pool) == {"vocal_lead", "bass"}
    tf = TorchFrontend(r"C:\OpenCode\AVIS\shared\frontend\v1\weights.bin", device="cuda")
    base = ProceduralMixer(MixConfig(seconds=4.0), device="cuda")
    sampler = RealSourceSampler(pool, batch=4, workers=0, seed=1)
    m = RealMixer(base, sampler, tf)
    b = m.batch(2)
    assert b["mix"].shape == (2, 192000, 2)
    for k in ("dom", "pitch", "sal"):
        assert b[k].shape == (2, 400, 192)
        assert torch.isfinite(b[k]).all()
    assert b["dom"].min() >= 0 and b["dom"].max() <= 7
    lab = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], tf)
    feat, active = lab["feat"], lab["active"]
    assert feat.shape == (2, 400, 3, 192) and torch.isfinite(feat).all()
    assert 0 < float(active.mean()) < 1
    assert lab["pitch_mask"].shape == (2, 400)
    assert m.stats["real_sources"] > 0


def test_chunk_reader_alignment(tmp_path):
    from avis_ml.real_data import read_chunk

    p = tmp_path / "a.flac"
    t = np.arange(int(SR * 3)) / SR
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32), SR, subtype="PCM_16")
    lab = tmp_path / "a.npz"
    np.savez(lab, start=np.array([0.5, 1.5], np.float32), end=np.array([1.0, 2.0], np.float32),
             pitch=np.array([57, 60], np.int16))
    entry = {"path": str(p), "dataset": "slakh", "sr": SR, "label": str(lab)}
    x, f0 = read_chunk(entry, 0.0)
    assert x.shape == (int(4 * SR),)
    assert f0.shape == (FRAMES, SLOTS)
    # A3 (57) sounds in [0.5, 1.0) s -> frames 50..99, no frame 120
    assert abs(float(f0[60, 0]) - 220.0) < 1 and f0[120, 0] == 0
    assert abs(float(f0[160, 0]) - 261.63) < 0.1
