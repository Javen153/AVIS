"""Procedural audio synthesis used by scenarios and the mixer (numpy, deterministic)."""
from __future__ import annotations

import numpy as np

SR = 48000


def t_axis(dur: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(round(dur * sr)), dtype=np.float64) / sr


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def sine(freq, dur, amp=0.25, sr=SR, phase=0.0) -> np.ndarray:
    t = t_axis(dur, sr)
    if np.isscalar(freq):
        ph = 2 * np.pi * freq * t
    else:
        ph = 2 * np.pi * np.cumsum(np.asarray(freq)[: len(t)]) / sr
    return amp * np.sin(ph + phase)


def log_sweep(f0, f1, dur, amp=0.25, sr=SR) -> np.ndarray:
    t = t_axis(dur, sr)
    k = np.log(f1 / f0) / dur
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    return amp * np.sin(ph)


def vibrato(f0, cents, rate, dur, amp=0.25, sr=SR) -> np.ndarray:
    t = t_axis(dur, sr)
    f = f0 * 2.0 ** (cents * np.sin(2 * np.pi * rate * t) / 1200.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    return amp * np.sin(ph)


def saw_additive(f0, dur, amp=0.25, sr=SR, max_hz=10000.0, tilt=1.0,
                 vibrato_cents=0.0, vibrato_rate=5.0) -> np.ndarray:
    t = t_axis(dur, sr)
    f = np.full_like(t, float(f0))
    if vibrato_cents:
        f = f0 * 2.0 ** (vibrato_cents * np.sin(2 * np.pi * vibrato_rate * t) / 1200.0)
    phase = np.cumsum(f) / sr  # cycles
    out = np.zeros_like(t)
    nh = max(1, int(min(64, max_hz / max(f0, 1e-6))))
    for n in range(1, nh + 1):
        if f0 * n > max_hz:
            break
        out += np.sin(2 * np.pi * n * phase) / (n ** tilt)
    return amp * out / (np.max(np.abs(out)) + 1e-12)


def adsr(n: int, a=0.01, d=0.1, s=0.7, r=0.2, sr=SR) -> np.ndarray:
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    ns = max(0, n - na - nd - nr)
    env = np.concatenate([
        np.linspace(0, 1, max(na, 1), endpoint=False),
        np.linspace(1, s, max(nd, 1), endpoint=False),
        np.full(ns, s),
        np.linspace(s, 0, max(nr, 1)),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def harmonic_tone(f0, dur, amp=0.25, a=1.0, sr=SR) -> np.ndarray:
    """Harmonic tone with 1/n^a amplitudes, random phase-free deterministic."""
    t = t_axis(dur, sr)
    out = np.zeros_like(t)
    nh = int(min(40, 12000 / max(f0, 1.0)))
    for n in range(1, nh + 1):
        out += np.sin(2 * np.pi * f0 * n * t) / n ** a
    return amp * out / (np.max(np.abs(out)) + 1e-12)


def fm_tone(fc, ratio, index, dur, amp=0.25, sr=SR) -> np.ndarray:
    t = t_axis(dur, sr)
    mod = index * np.sin(2 * np.pi * fc * ratio * t)
    return amp * np.sin(2 * np.pi * fc * t + mod)


def white_noise(dur, amp=0.1, sr=SR, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(dur * sr))


def pink_noise(dur, amp=0.1, sr=SR, seed=2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    X[0] = 0.0  # no DC
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return amp * y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def brown_noise(dur, amp=0.1, sr=SR, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    X[0] = 0.0  # no DC
    X /= f
    y = np.fft.irfft(X, n)
    return amp * y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def band_noise(dur, lo, hi, amp=0.1, sr=SR, seed=4) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    x = white_noise(dur, 1.0, sr, seed)
    sos = butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], btype="band", output="sos")
    y = sosfilt(sos, x)
    return amp * y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def kick(dur=0.35, amp=1.0, sr=SR, seed=10) -> np.ndarray:
    t = t_axis(dur, sr)
    f = 150.0 * np.exp(-t * 18.0) + 45.0
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 7.0)
    click = white_noise(dur, 0.35, sr, seed) * np.exp(-t * 90.0)
    y = body + click
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def snare(dur=0.25, amp=1.0, sr=SR, seed=11) -> np.ndarray:
    t = t_axis(dur, sr)
    tone = 210.0 * np.sin(2 * np.pi * 210 * t) * np.exp(-t * 25.0)
    noise = band_noise(dur, 1000, 9000, 1.0, sr, seed) * np.exp(-t * 16.0)
    y = 0.6 * tone + noise
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def hihat(dur=0.08, amp=1.0, sr=SR, seed=12) -> np.ndarray:
    t = t_axis(dur, sr)
    y = band_noise(dur, 7000, 14000, 1.0, sr, seed) * np.exp(-t * 55.0)
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def cymbal(dur=1.2, amp=1.0, sr=SR, seed=13) -> np.ndarray:
    t = t_axis(dur, sr)
    y = band_noise(dur, 3000, 16000, 1.0, sr, seed) * np.exp(-t * 3.2)
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def tom(f0=120, dur=0.4, amp=1.0, sr=SR) -> np.ndarray:
    t = t_axis(dur, sr)
    f = f0 * (1.0 + 0.5 * np.exp(-t * 20.0))
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 8.0)
    return amp * y / (np.max(np.abs(y)) + 1e-12)


def formant_voice(f0_curve, dur, amp=0.3, sr=SR, formants=(700, 1220, 2600)) -> np.ndarray:
    """Glottal pulse train shaped by three resonant formant filters."""
    from scipy.signal import lfilter
    t = t_axis(dur, sr)
    f0 = np.asarray(f0_curve)
    if f0.ndim == 0:
        f0 = np.full(len(t), float(f0))
    f0 = f0[: len(t)]
    phase = np.cumsum(f0) / sr
    pulses = np.diff(np.floor(phase), prepend=0.0)
    y = pulses
    out = np.zeros_like(y)
    for bw, fc in zip((90.0, 110.0, 160.0), formants):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        b = [1.0 - r]
        out += lfilter(b, a, y)
    out = out / (np.max(np.abs(out)) + 1e-12)
    return amp * out


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan, pan in [-1,1]."""
    pan = float(np.clip(pan, -1, 1))
    ang = (pan + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=1)


def normalize_rms(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + 1e-20)
    return x * (db_to_amp(dbfs) / rms)


def place(canvas: np.ndarray, mono: np.ndarray, start_sec: float, pan=0.0, gain=1.0):
    sr = SR
    i = int(round(start_sec * sr))
    if i >= canvas.shape[0]:
        return
    n = min(len(mono), canvas.shape[0] - i)
    canvas[i:i + n] += pan_stereo(mono[:n] * gain, pan)


def write_stereo(path, x: np.ndarray, sr=SR):
    import soundfile as sf
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(str(path), x, sr, subtype="FLOAT")
