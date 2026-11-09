"""Generate FE-v1 golden vectors (input signal + expected features)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import frontend as fe
from . import synth as sy

SR = 48000


def golden_signal() -> np.ndarray:
    dur = 3.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    # steady sines with different pan
    L += 0.25 * np.sin(2 * np.pi * 55 * t)
    R += 0.25 * np.sin(2 * np.pi * 55 * t + 0.2)
    L += 0.20 * sy.sine(440, dur, 1.0)
    R += 0.20 * sy.sine(440, dur, 1.0, phase=0.4)
    L += 0.10 * sy.sine(3000, dur, 1.0)
    R += 0.05 * sy.sine(3000, dur, 1.0, phase=0.7)
    # log sweep 300 -> 6000 in 1.5..2.5 s
    sw = sy.log_sweep(300, 6000, 1.0, 0.25)
    i = int(1.5 * SR)
    L[i:i + len(sw)] += sw
    R[i:i + len(sw)] += 0.6 * sw
    # pink noise -30 dBFS
    pn = sy.pink_noise(dur, 1.0, seed=99)
    pn = sy.normalize_rms(pn, -30.0)
    L += pn
    R += 0.8 * pn
    # clicks
    rng = np.random.default_rng(3)
    for _ in range(12):
        j = int(rng.uniform(0.1, 2.9) * SR)
        L[j:j + 40] += 0.5 * np.exp(-np.arange(40) / 5.0)
        R[j:j + 40] += 0.3 * np.exp(-np.arange(40) / 5.0)
    x = np.stack([L, R], axis=1).astype(np.float32)
    return x


def generate(out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wdir = out.parents[1] / "shared" / "frontend" / "v1"
    if not (wdir / "weights.bin").exists():
        fe.save_weights(wdir)
    x = golden_signal()
    nf = fe.NumpyFrontend(wdir / "weights.bin")
    frames = nf.push(x.astype(np.float64))
    feat = np.stack([f["features"] for f in frames]).astype(np.float32)
    x.astype("<f4").tofile(out / "frontend_input.f32")
    feat.astype("<f4").tofile(out / "frontend_expected.f32")
    meta = {"sr": SR, "hop": 480, "frames": int(feat.shape[0]), "bands": 192,
            "input": "frontend_input.f32 interleaved stereo float32",
            "output": "frontend_expected.f32 [T,3,192] float32"}
    (out / "frontend_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(generate(root / "shared" / "golden"))
