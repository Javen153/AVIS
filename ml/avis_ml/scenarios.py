"""Deterministic scenario generator S01-S13 (48 kHz stereo) + expect files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import synth as sy

SR = 48000
FREQ = {"C4": 261.6256, "E4": 329.6276, "G4": 392.0000}


def _canvas(dur):
    return np.zeros((int(round(dur * SR)), 2), dtype=np.float64)


def gen_s01():
    x = _canvas(5.0)
    sy.place(x, sy.sine(440, 5.0, sy.db_to_amp(-12)), 0.0)
    return x, {"window": [0.5, 4.5], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.95},
        {"type": "pitch", "hz": 440, "tol_cents": 20, "frac": 0.9},
    ]}


def gen_s02():
    x = _canvas(5.0)
    sy.place(x, sy.log_sweep(200, 2000, 5.0, sy.db_to_amp(-12)), 0.0)
    return x, {"window": [0.3, 4.7], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.95},
        {"type": "id_stable", "frac": 0.95},
        {"type": "pitch_sweep", "f0": 200, "f1": 2000, "t0": 0.0, "t1": 5.0,
         "tol_cents": 50, "median": True, "frac": 0.8},
    ]}


def gen_s03():
    x = _canvas(5.5)
    sy.place(x, sy.vibrato(440, 50, 5.5, 5.5, sy.db_to_amp(-12)), 0.0)
    return x, {"window": [0.5, 5.0], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.95},
        {"type": "id_stable", "frac": 0.95},
        {"type": "pitch", "hz": 440, "tol_cents": 60, "frac": 0.8},
    ]}


def gen_s04():
    x = _canvas(5.0)
    sy.place(x, sy.sine(300, 5.0, sy.db_to_amp(-12)), 0.0)
    sy.place(x, sy.sine(1700, 4.0, sy.db_to_amp(-12)), 1.0)
    return x, {"window": [1.3, 5.0], "checks": [
        {"type": "count", "min": 2, "max": 2, "frac": 0.9},
        {"type": "object_near_pitch", "hz": 300, "tol_cents": 30, "frac": 0.8},
        {"type": "object_near_pitch", "hz": 1700, "tol_cents": 30, "frac": 0.8},
        {"type": "distinct_ids", "min": 2, "max": 2},
    ]}


def gen_s05():
    x = _canvas(5.0)
    sy.place(x, sy.saw_additive(220, 5.0, sy.db_to_amp(-12), max_hz=10000), 0.0)
    return x, {"window": [0.5, 4.5], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.9},
        {"type": "pitch", "hz": 220, "tol_cents": 30, "frac": 0.8},
    ]}


def gen_s06():
    x = _canvas(5.0)
    for f in (FREQ["C4"], FREQ["E4"], FREQ["G4"]):
        sy.place(x, sy.harmonic_tone(f, 5.0, sy.db_to_amp(-15), a=1.0), 0.0)
    return x, {"window": [0.5, 4.5], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.8},
        {"type": "pitches_present", "hz": [FREQ["C4"], FREQ["E4"], FREQ["G4"]],
         "tol_cents": 50, "frac": 0.7},
    ]}


def gen_s07():
    x = _canvas(5.0)
    sy.place(x, sy.white_noise(5.0, 1.0, seed=21), 0.0)
    from .synth import normalize_rms
    x = normalize_rms(x, -30.0)
    return x, {"window": [0.5, 4.5], "checks": [
        {"type": "count", "min": 1, "max": 1, "frac": 0.8},
        {"type": "width_frac_min", "value": 0.8, "frac": 0.7},
        {"type": "tonality_max", "value": 0.3, "frac": 0.7},
    ]}


def gen_s08():
    x = _canvas(6.0)
    sy.place(x, sy.white_noise(3.0, 1.0, seed=22), 3.0)
    x = sy.normalize_rms(x, -95.0) if np.any(x) else x
    return x, {"window": [0.0, 6.0], "checks": [
        {"type": "count", "min": 0, "max": 0, "frac": 1.0},
    ]}


def _drum_hits(seed=30):
    hits = []
    # 120 BPM, 8 s, 4 bars
    for bar in range(4):
        b = bar * 2.0
        for beat, name in ((0.0, "kick"), (0.5, "snare"), (1.0, "kick"), (1.5, "snare")):
            hits.append((b + beat, name))
        for i in range(8):
            hits.append((b + i * 0.25, "hihat"))
    return sorted(hits)


def gen_s09():
    x = _canvas(8.0)
    hits = _drum_hits()
    rng = np.random.default_rng(30)
    for t, name in hits:
        g = sy.db_to_amp(-14 + rng.uniform(-2, 2))
        if name == "kick":
            m = sy.kick(seed=31 + int(t * 10))
        elif name == "snare":
            m = sy.snare(seed=41 + int(t * 10))
        else:
            m = sy.hihat(seed=51 + int(t * 10))
        sy.place(x, m, t, pan={"kick": 0.0, "snare": 0.05, "hihat": 0.3}[name], gain=g)
    expect_hits = [{"t": round(t, 3), "type": n} for t, n in hits]
    return x, {"window": [0.2, 7.8], "checks": [
        {"type": "distinct_ids", "min": 3, "max": 5},
        {"type": "drum_ids", "hits": expect_hits, "frac": 0.85},
    ]}


def gen_s10():
    x = _canvas(6.0)
    t = sy.t_axis(6.0)
    f0 = 150.0 * (220.0 / 150.0) ** (t / 6.0)
    f0 = f0 * 2.0 ** (30.0 * np.sin(2 * np.pi * 5.0 * t) / 1200.0)
    voice = sy.formant_voice(f0, 6.0, sy.db_to_amp(-12))
    sy.place(x, voice, 0.0, pan=-0.1)
    noise = sy.pink_noise(6.0, 1.0, seed=61)
    noise = sy.normalize_rms(noise, -20.0)
    sy.place(x, noise, 0.0, pan=0.2)
    return x, {"window": [1.0, 5.5], "checks": [
        {"type": "count", "min": 2, "max": 3, "frac": 0.8},
        {"type": "voice_pitch", "curve": {"f0": 150, "f1": 220, "t0": 0.0, "t1": 6.0},
         "tol_cents": 80, "frac": 0.8},
    ]}


def gen_s11():
    x = _canvas(18.0)
    A = sy.saw_additive(330, 4.0, sy.db_to_amp(-12), max_hz=6000)
    B = sy.saw_additive(550, 10.0, sy.db_to_amp(-12), max_hz=9000, tilt=0.7)
    sy.place(x, A, 0.0)
    sy.place(x, B, 4.0)
    sy.place(x, A, 14.0)
    return x, {"window": [0.5, 17.5], "checks": [
        {"type": "same_id_after_gap", "before": [0.5, 3.5], "gap": [4.0, 14.0],
         "after": [14.5, 17.5], "frac": 0.9},
        {"type": "count", "min": 1, "max": 3, "frac": 0.7},
    ]}


def gen_s12():
    x = _canvas(6.0)
    lead = sy.harmonic_tone(220, 6.0, sy.db_to_amp(-12), a=1.1)
    back = sy.harmonic_tone(220 * 2 ** (4 / 12), 5.0, sy.db_to_amp(-15), a=1.1)
    sy.place(x, lead, 0.0, pan=0.0)
    sy.place(x, back, 0.5, pan=0.7)
    return x, {"window": [1.0, 5.5], "checks": [
        {"type": "count", "min": 2, "max": 3, "frac": 0.8},
        {"type": "object_near_pitch", "hz": 220, "tol_cents": 60, "frac": 0.7},
        {"type": "object_near_pitch", "hz": 220 * 2 ** (4 / 12), "tol_cents": 60, "frac": 0.7},
        {"type": "distinct_ids", "min": 2, "max": 4},
    ]}


def gen_s13():
    x = _canvas(6.0)
    noise = sy.pink_noise(6.0, 1.0, seed=71)
    noise = np.tanh(noise * 8.0)
    noise = sy.normalize_rms(noise, -10.0)
    sy.place(x, noise, 0.0, pan=0.0)
    sy.place(x, sy.saw_additive(330, 6.0, sy.db_to_amp(-24), max_hz=5000), 0.0, pan=0.0)
    return x, {"window": [1.0, 5.5], "checks": [
        {"type": "object_near_pitch", "hz": 330, "tol_cents": 40, "frac": 0.8},
    ]}


GENERATORS = {
    "S01": gen_s01, "S02": gen_s02, "S03": gen_s03, "S04": gen_s04, "S05": gen_s05,
    "S06": gen_s06, "S07": gen_s07, "S08": gen_s08, "S09": gen_s09, "S10": gen_s10,
    "S11": gen_s11, "S12": gen_s12, "S13": gen_s13,
}


def generate(out_dir: str | Path, only=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in GENERATORS.items():
        if only and name not in only:
            continue
        x, expect = fn()
        x = x - x.mean(axis=0, keepdims=True)  # remove any DC offset
        sy.write_stereo(out / f"{name}.wav", x, SR)
        (out / f"{name}.expect.json").write_text(
            json.dumps(expect, ensure_ascii=False, indent=1), encoding="utf-8")
        made.append(name)
    return made


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    names = generate(root / "testdata" / "scenarios")
    print("generated:", ",".join(names))
