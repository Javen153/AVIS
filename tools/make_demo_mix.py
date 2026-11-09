"""Render one procedural training-style mix to a WAV for visual checks.

Usage (from the repository root):
    uv run --project ml python tools/make_demo_mix.py --out testdata/demo/mix.wav --seconds 12 --seed 7

The mixer is the same code that produced the training data, so the clip is a
reasonable stand-in for music when no real dataset is available (D-001).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/demo/mix.wav")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--sources", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ml"))
    import soundfile as sf
    from avis_ml.mixer import MixConfig, ProceduralMixer

    mixer = ProceduralMixer(
        MixConfig(seconds=a.seconds, max_sources=a.sources, seed=a.seed), device="cpu"
    )
    mixer.g.manual_seed(a.seed)
    mix, _scores, _f0s, kinds = mixer._one()
    x = mix.detach().cpu().numpy().astype(np.float32)
    out = Path(a.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, x, 48000, subtype="FLOAT")
    print(f"wrote {out} shape={x.shape} kinds={kinds} peak={float(np.abs(x).max()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
