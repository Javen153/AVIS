"""M-v3 mixer and stereo chunk reader. CPU only: the suite must run while a GPU
training job owns the device."""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from avis_ml.frontend import TorchFrontend
from avis_ml.mixer import MixConfig, ProceduralMixer, make_labels
from avis_ml.mixer_v3 import RealMixerV3
from avis_ml.real_data import FRAMES, SLOTS, ChunkDataset, load_pool, read_chunk_stereo

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "shared" / "frontend" / "v1" / "weights.bin"
SR = 44100
SEC = 6.0
N = 192000


def _mixer(seed: int) -> RealMixerV3:
    """Hard-case-only v3 mixer without real stems (sampler=None)."""
    tf = TorchFrontend(WEIGHTS, device="cpu")
    base = ProceduralMixer(MixConfig(seconds=4.0), device="cpu")
    m = RealMixerV3(base, None, tf, proc_p=0.0, hard_p=1.0)
    base.g.manual_seed(seed)
    return m


def _write_pool(tmp_path: Path) -> Path:
    idx = tmp_path / "index.jsonl"
    rows = []
    for cat, hz in (("vocal_lead", 220.0), ("bass", 55.0)):
        p = tmp_path / f"{cat}.flac"
        t = np.arange(int(SR * SEC)) / SR
        sf.write(str(p), (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32), SR,
                 subtype="PCM_16")
        lab = tmp_path / f"{cat}.npz"
        np.savez(lab, starts=np.array([0.0, 1.0], np.float32),
                 f0=np.full((2, FRAMES), hz, np.float32),
                 prob=np.ones((2, FRAMES), np.float32))
        rows.append({"path": str(p), "dataset": "musdb", "track_id": cat,
                     "stem_id": cat, "split": "train", "category": cat, "sr": SR,
                     "channels": 1, "duration": SEC, "active_segments": [0, 1],
                     "label": str(lab)})
    idx.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return idx


def test_hard_cases_shapes_and_labels():
    m = _mixer(1234)
    b = m.batch(2)
    assert b["mix"].shape == (2, N, 2)
    assert torch.isfinite(b["mix"]).all()
    assert float(b["mix"].abs().max()) > 0.05  # audible, not a silent draw
    for k in ("dom", "pitch", "sal", "margin"):
        assert b[k].shape == (2, FRAMES, 192), k
        assert torch.isfinite(b[k]).all(), k
    assert b["pitch_mask"].shape == (2, FRAMES)
    assert torch.isfinite(b["pitch_mask"]).all()
    assert b["solo"].shape == (2,)
    assert float(b["dom"].min()) >= 0
    for k in ("pitch", "sal"):
        assert float(b[k].min()) >= 0.0 and float(b[k].max()) <= 1.0, k
    assert float(b["margin"].min()) >= 0.0  # dB gap between top-1 and top-2
    mask = b["pitch_mask"]
    assert torch.equal(mask, (mask > 0.5).float())
    # every example is a hard case and the five family counters add up
    assert m.stats["hard_examples"] == 2
    fam = (m.stats["close_pair"] + m.stats["triad"] + m.stats["pitch_noise"]
           + m.stats["drum_bed"] + m.stats["solo"])
    assert fam == m.stats["hard_examples"]
    lab = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], m.tf,
                      margin=b["margin"], pitch_mask=b["pitch_mask"])
    assert lab["feat"].shape == (2, FRAMES, 3, 192)
    assert torch.isfinite(lab["feat"]).all()
    assert 0.0 < float(lab["active"].mean()) < 1.0


def test_hard_batch_is_deterministic_per_seed():
    a = _mixer(1234).batch(2)
    b = _mixer(1234).batch(2)  # fresh mixer instance: no hidden global RNG state
    c = _mixer(1235).batch(2)
    keys = ("mix", "dom", "pitch", "sal", "margin", "pitch_mask")
    for k in keys:
        assert torch.equal(a[k], b[k]), k
    assert any(not torch.equal(a[k], c[k]) for k in keys)


def test_pitch_noise_realised_snr():
    """The tone must sit BELOW the noise: the pre-fix sign gave +4..+18 dB.

    `_hard_pitch_noise` draws a target in [-18, -3] dB, augments both sources
    (distortion/reverb change each by up to ~15 dB) and only then scales the
    noise, so the realised ratio has to match the target.
    """
    m = _mixer(0)
    got = []
    for seed in range(12):
        torch.manual_seed(seed)
        r = m._hard_pitch_noise()
        if r is None:
            continue
        src = r[1]
        tone = float(src[0].pow(2).mean().sqrt())
        noise = float(src[1].pow(2).mean().sqrt())
        assert tone > 0.0 and noise > 0.0
        got.append(20.0 * math.log10(tone / noise))
    assert len(got) >= 10, len(got)
    assert all(-19.5 <= x <= -1.5 for x in got), [round(x, 2) for x in got]


def test_combine_relevel_false_keeps_relative_levels():
    """relevel=False must keep the levels a hard-case builder designed."""
    m = _mixer(0)

    def src(rms, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(N, 2, generator=g)
        return x / x.pow(2).mean().sqrt() * rms

    srcs = [(src(0.4, 0), 0.0, torch.zeros(FRAMES, SLOTS), True),
            (src(0.1, 1), 0.0, torch.zeros(FRAMES, SLOTS), True)]
    out = m._combine(srcs, relevel=False)
    assert out is not None
    rms = out[1].pow(2).mean(dim=(1, 2)).sqrt()
    ratio = float(rms[0] / rms[1])
    assert abs(ratio / 4.0 - 1.0) <= 0.05, ratio  # designed 4:1 survives
    # and relevel=True does re-draw the balance, otherwise the flag is dead code
    ratios = []
    for seed in (21, 22, 23):
        m.base.g.manual_seed(seed)
        r = m._combine(srcs, relevel=True)[1].pow(2).mean(dim=(1, 2)).sqrt()
        ratios.append(float(r[0] / r[1]))
    assert max(abs(x / 4.0 - 1.0) for x in ratios) > 0.2, ratios


def _stub_rand(m: RealMixerV3, values) -> None:
    """Force the draws inside _augment_stereo: speed, distortion gate, width, rot."""
    it = iter(values)
    m.base._rand = lambda *a, **k: next(it)


def test_stereo_augment_identity_roundtrip():
    """width=1, rot=0 must return the input unchanged.

    Regression for the mid/side mix-up: the old code returned (mid, side) as
    (L, R), so any L == R source lost its right channel entirely (all Slakh
    stems are mono duplicated to stereo).
    """
    m = _mixer(5)
    left, right = torch.randn(N), torch.randn(N)
    x = torch.stack([left, right], dim=1)
    _stub_rand(m, [0.0, 0.9, 1.0, 0.0])  # speed 0 -> 1.0, no distortion, width 1, rot 0
    y, speed, pan = m._augment_stereo(x)
    assert speed == 1.0 and pan == 0.0
    assert float((y - x).abs().max()) <= 1e-6, float((y - x).abs().max())
    mono = x[:, :1].repeat(1, 2)
    _stub_rand(m, [0.0, 0.9, 1.0, 0.0])
    y2, _, _ = m._augment_stereo(mono)
    assert float((y2[:, 0] - y2[:, 1]).abs().max()) <= 1e-6


def test_stereo_augment_keeps_mono_audible():
    """Rotation may pan a duplicated mono source, but must not silence a channel."""
    m = _mixer(5)
    torch.manual_seed(0)
    x = torch.randn(N, 1).repeat(1, 2)
    rms_in = float(x[:, 0].pow(2).mean().sqrt())
    energy_in = float(x.pow(2).sum())
    for _ in range(50):
        y, _, _ = m._augment_stereo(x)
        rms = y.pow(2).mean(dim=0).sqrt()
        # measured min per-channel ratio 0.61 (rot 0.25 pans by ~4.5 dB); the bug
        # gave 0.0 on the right channel
        assert float(rms.min()) > 0.2 * rms_in, float(rms.min()) / rms_in
        assert abs(float(y.pow(2).sum()) / energy_in - 1.0) < 0.7
    ones = torch.ones(N, 2)
    y, _, _ = m._augment_stereo(ones)
    assert float(y.min()) > 1e-3 and float(y.max()) < 3.0


def test_stereo_augment_keeps_level():
    m = _mixer(5)
    torch.manual_seed(0)
    x = torch.randn(N, 1).repeat(1, 2)
    rms_in = float(x[:, 0].pow(2).mean().sqrt())
    ones = torch.ones(N, 2)
    for _ in range(20):
        y, speed, pan = m._augment_stereo(x)
        assert pan == 0.0 and 0.8 < speed < 1.3
        assert float(y[:, 0].pow(2).mean().sqrt()) > 0.5 * rms_in
        assert torch.isfinite(y).all()
        assert float(m._augment_stereo(ones)[0].abs().min()) > 1e-3  # not silenced


def test_read_chunk_stereo_keeps_channels_and_duplicates_mono(tmp_path):
    t = np.arange(int(SR * 3)) / SR
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 110 * t),
                       0.2 * np.cos(2 * np.pi * 110 * t)], axis=1)
    mono = 0.3 * np.sin(2 * np.pi * 220 * t)
    lab = tmp_path / "a.npz"
    np.savez(lab, start=np.array([0.5], np.float32), end=np.array([1.0], np.float32),
             pitch=np.array([57], np.int16))  # A3 = 220 Hz

    sp = tmp_path / "st.flac"
    sf.write(str(sp), stereo.astype(np.float32), SR, subtype="PCM_16")
    mp = tmp_path / "mono.flac"
    sf.write(str(mp), mono.astype(np.float32), SR, subtype="PCM_16")
    base = {"dataset": "slakh", "sr": SR, "label": str(lab)}

    a, f0 = read_chunk_stereo({**base, "path": str(sp)}, 0.0)
    assert a.shape == (int(4 * SR), 2) and a.dtype == np.float32
    assert f0.shape == (FRAMES, SLOTS)
    # real stereo content is preserved, not downmixed
    assert not np.array_equal(a[:, 0], a[:, 1])
    # short (3 s) file is zero-padded to the 4 s chunk
    assert float(np.abs(a[-SR // 2:]).max()) == 0.0

    b, _ = read_chunk_stereo({**base, "path": str(mp)}, 0.0)
    assert b.shape == (int(4 * SR), 2)
    assert np.array_equal(b[:, 0], b[:, 1])  # mono duplicated to both channels


def test_read_chunk_stereo_f0_alignment(tmp_path):
    t = np.arange(int(SR * 3)) / SR
    p = tmp_path / "m.flac"
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR,
             subtype="PCM_16")
    lab = tmp_path / "m.npz"
    np.savez(lab, start=np.array([0.5, 1.5], np.float32),
             end=np.array([1.0, 2.0], np.float32), pitch=np.array([57, 60], np.int16))
    entry = {"path": str(p), "dataset": "slakh", "sr": SR, "label": str(lab)}
    _, f0 = read_chunk_stereo(entry, 0.0)
    assert f0.shape == (FRAMES, SLOTS)
    # A3 sounds in [0.5, 1.0) s -> frames 50..99; C4 in [1.5, 2.0) s -> 150..199
    assert abs(float(f0[60, 0]) - 220.0) < 1.0
    assert abs(float(f0[160, 0]) - 261.63) < 0.1
    assert float(f0[120, 0]) == 0.0
    assert float(f0[:, 1:].max()) == 0.0


def test_chunk_dataset_streams_stereo(tmp_path):
    pool = load_pool(_write_pool(tmp_path), ("train",))
    assert set(pool) == {"vocal_lead", "bass"}
    ds = ChunkDataset(pool, sorted(pool), [1.0] * len(pool), seed=0, stereo=True)
    item = next(iter(ds))
    assert item["audio"].shape == (int(4 * SR), 2)
    assert item["f0"].shape == (FRAMES, SLOTS)
    assert item["cat"].dim() == 0


def _side_rms(mix: torch.Tensor) -> float:
    """RMS of the side channel: 0 for a mix whose sources share one pan."""
    side = 0.5 * (mix[:, 0] - mix[:, 1])
    return float(side.pow(2).mean().sqrt())


def test_close_pair_includes_colocated_variants():
    """Some hard pairs must have no side cue, otherwise the embedding learns pan
    as a shortcut and collapses on mono scenario audio (D-038)."""
    m = _mixer(2024)
    zeros = 0
    for _ in range(40):
        out = m._hard_close_pair()
        assert out is not None
        if _side_rms(out[0]) < 1e-6:
            zeros += 1
    # target share is 55 %; the bound is loose enough for a fixed seed
    assert zeros >= 8, f"only {zeros}/40 co-located pairs"


def test_solo_family_has_one_source_and_wide_bands():
    """S05/S07 are single sources; every other family has K>=2 sources (D-040).

    A solo build must produce exactly one source and a mix with real energy
    well above 2 kHz, otherwise it cannot teach that the high harmonics of one
    source belong to the same object.
    """
    m = _mixer(2026)
    wide = 0
    for _ in range(20):
        out = m._hard_solo()
        assert out is not None
        mix, src, f0s, known = out
        assert mix.shape == (N, 2)
        assert torch.isfinite(mix).all()
        assert src.shape[0] == 1, src.shape
        assert f0s.shape == (1, FRAMES, SLOTS)
        mono = mix.mean(dim=1)
        spec = torch.fft.rfft(mono).abs().pow(2)
        fr = torch.arange(spec.shape[0]) * SR / float(N)
        hi = float(spec[(fr > 2000.0) & (fr < 11000.0)].sum())
        share = hi / (float(spec.sum()) + 1e-12)
        if share > 0.01:
            wide += 1
    assert wide >= 12, f"only {wide}/20 solos carry energy above 2 kHz"


def test_batch_marks_solo_examples():
    """Solo examples are flagged so the coherence term can see them (D-041)."""
    m = _mixer(2027)
    b = m.batch(40)
    n = int(b["solo"].sum())
    assert n == m.stats["solo"]
    assert 3 <= n <= 15, n
    assert set(b["solo"].tolist()) <= {0.0, 1.0}


def test_triad_is_colocated_and_keeps_pitch_labels():
    m = _mixer(2025)
    colocated = 0
    for _ in range(30):
        out = m._hard_triad()
        assert out is not None
        mix, src, f0s, known = out
        assert mix.shape == (N, 2)
        assert torch.isfinite(mix).all()
        assert f0s.shape == (src.shape[0], FRAMES, SLOTS)
        # each tonal source owns one pitch slot with a distinct frequency
        mean_f0 = f0s[:, :, 0].mean(dim=1)
        tones = sorted({round(float(v), 1) for v in mean_f0 if float(v) > 0})
        assert 2 <= len(tones) <= 3, tones
        assert all(b > a for a, b in zip(tones, tones[1:]))
        if _side_rms(mix) < 1e-6:
            colocated += 1
    assert colocated >= 5, f"only {colocated}/30 co-located triads"
