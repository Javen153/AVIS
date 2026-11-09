"""GPU procedural mixer and per-band labels for M-v1 training.

Each example is 4 s stereo at 48 kHz. Sources are synthesized in torch and
mixed with per-source gain/pan/start/augmentations. Labels are computed on the
same 192-band FE-v1 grid (analytic source scores + mixed-band activity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import frontend as fe

SR = 48000
CATEGORIES = [
    "vocal_lead", "vocal_back", "bass", "guitar", "keys", "piano", "synth_lead",
    "synth_pad", "strings", "brass", "winds", "organ", "kick", "snare", "hihat",
    "cymbal", "tom", "perc", "other_tonal", "fx",
]
TONAL = ["vocal_lead", "vocal_back", "bass", "guitar", "keys", "piano",
         "synth_lead", "synth_pad", "strings", "brass", "winds", "organ",
         "other_tonal", "fx"]
DRUMS = ["kick", "snare", "hihat", "cymbal", "tom", "perc"]

# f0 range, harmonic tilt, harmonic count, vibrato cents, attack
TIMBRE = {
    "vocal_lead": (90, 520, 1.35, 14, 25, 0.03),
    "vocal_back": (90, 520, 1.35, 14, 25, 0.03),
    "bass": (40, 260, 1.0, 9, 3, 0.005),
    "guitar": (80, 800, 0.85, 14, 5, 0.003),
    "keys": (70, 1200, 1.2, 10, 4, 0.005),
    "piano": (60, 1500, 1.15, 12, 3, 0.002),
    "synth_lead": (180, 2200, 0.65, 8, 20, 0.01),
    "synth_pad": (90, 1100, 1.0, 12, 6, 0.25),
    "strings": (100, 1400, 1.1, 14, 8, 0.12),
    "brass": (80, 900, 0.7, 12, 4, 0.03),
    "winds": (180, 1600, 1.35, 8, 6, 0.04),
    "organ": (90, 1100, 0.9, 10, 2, 0.01),
    "other_tonal": (70, 1400, 1.0, 10, 5, 0.01),
    "fx": (120, 2000, 0.8, 8, 10, 0.02),
}


@dataclass
class MixConfig:
    seconds: float = 4.0
    max_sources: int = 8
    sr: int = SR
    seed: int = 0


class ProceduralMixer:
    def __init__(self, cfg: MixConfig, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n = int(cfg.seconds * cfg.sr)
        self.T = self.n // fe.HOP
        self.fc = torch.tensor(fe.band_centers(), dtype=torch.float32, device=self.device)
        self.logfc = torch.log(self.fc)
        self.time = torch.arange(self.n, device=self.device, dtype=torch.float32) / cfg.sr
        self.g = torch.Generator(device=self.device)
        self.alpha_band = 0.055  # ~1 band gaussian width in log-frequency

    # ------------------------------------------------------------- utilities
    def _rand(self, shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(shape, device=self.device, generator=self.g)

    def _env(self, gain, attack, dur, release):
        n = self.n
        t = self.time
        atk = torch.clamp(t / max(attack, 1e-3), 0, 1)
        rel = torch.clamp((dur - t) / max(release, 1e-3), 0, 1)
        return gain * atk * rel

    def _tone(self, f0, timbre, dur, gain, attack):
        lo, hi, tilt, nh, vib, _ = TIMBRE.get(timbre, TIMBRE["other_tonal"])
        t = self.time
        vibc = float(self._rand((), 0, vib)) * (1 if timbre in ("vocal_lead", "vocal_back", "synth_lead", "strings") else 0.3)
        rate = float(self._rand((), 3.5, 6.5))
        f = f0 * torch.pow(torch.tensor(2.0, device=self.device), vibc * torch.sin(2 * math.pi * rate * t) / 1200.0)
        phase = 2 * math.pi * torch.cumsum(f, 0) / self.cfg.sr
        out = torch.zeros_like(t)
        nmax = 1
        while nmax < nh and f0 * (nmax + 1) < 12000:
            nmax += 1
        for k in range(1, nmax + 1):
            out += torch.sin(phase * k) / (k ** tilt)
        out = out / (out.abs().max() + 1e-9)
        return out * self._env(gain, attack, dur, 0.05)

    def _chord(self, notes, timbre, dur, gain, attack):
        out = torch.zeros(self.n, device=self.device)
        for f0 in notes:
            out += self._tone(f0, timbre, dur, 1.0, attack)
        return out / (out.abs().max() + 1e-9) * self._env(gain, attack, dur, 0.05)

    def _noise(self, color, gain, dur, start):
        w = torch.randn(self.n, device=self.device, generator=self.g)
        if color == "pink" or color == "brown":
            X = torch.fft.rfft(w)
            f = torch.arange(X.shape[0], device=self.device, dtype=torch.float32)
            f[0] = 1
            X = X / torch.sqrt(f) if color == "pink" else X / f
            w = torch.fft.irfft(X, self.n)
        elif color == "band":
            X = torch.fft.rfft(w)
            fr = torch.arange(X.shape[0], device=self.device, dtype=torch.float32) * SR / self.n
            lo = float(self._rand((), 300, 3000))
            hi = min(lo * float(self._rand((), 2.0, 8.0)), 18000)
            mask = ((fr > lo) & (fr < hi)).float()
            w = torch.fft.irfft(X * mask, self.n)
        w = w / (w.std() + 1e-9)
        t = self.time
        env = ((t >= start) & (t < start + dur)).float()
        env = env * torch.clamp((t - start) / 0.01, 0, 1) * torch.clamp((start + dur - t) / 0.1, 0, 1)
        return w * env * gain

    def _place(self, x, start, gain):
        out = torch.zeros(self.n, device=self.device)
        i = int(start * self.cfg.sr)
        if i >= self.n:
            return out
        L = min(len(x), self.n - i)
        out[i:i + L] = x[:L] * gain
        return out

    def _drum_track(self, kind, tempo, dur, gain):
        out = torch.zeros(self.n, device=self.device)
        beat = 60.0 / tempo
        step = beat / (2 if kind == "hihat" else 1)
        t = float(self._rand((), 0, step))
        while t < dur:
            out += self._place(self._drum(kind), t, gain * float(self._rand((), 0.7, 1.0)))
            t += step * float(self._rand((), 0.9, 1.1))
        return out

    def _drum(self, kind):
        t = self.time
        if kind == "kick":
            f = 150 * torch.exp(-t * 18) + 45
            y = torch.sin(2 * math.pi * torch.cumsum(f, 0) / self.cfg.sr) * torch.exp(-t * 7)
            y += torch.randn(self.n, device=self.device, generator=self.g) * torch.exp(-t * 90) * 0.3
            return y / (y.abs().max() + 1e-9)
        if kind == "snare":
            tone = torch.sin(2 * math.pi * 210 * t) * torch.exp(-t * 25)
            noise = torch.randn(self.n, device=self.device, generator=self.g) * torch.exp(-t * 16)
            X = torch.fft.rfft(noise); fr = torch.arange(X.shape[0], device=self.device) * SR / self.n
            noise = torch.fft.irfft(X * ((fr > 1000) & (fr < 9000)).float(), self.n)
            y = 0.6 * tone + noise / (noise.abs().max() + 1e-9)
            return y / (y.abs().max() + 1e-9)
        if kind == "hihat":
            noise = torch.randn(self.n, device=self.device, generator=self.g) * torch.exp(-t * 55)
            X = torch.fft.rfft(noise); fr = torch.arange(X.shape[0], device=self.device) * SR / self.n
            y = torch.fft.irfft(X * ((fr > 7000) & (fr < 15000)).float(), self.n)
            return y / (y.abs().max() + 1e-9)
        if kind == "cymbal":
            noise = torch.randn(self.n, device=self.device, generator=self.g) * torch.exp(-t * 3.2)
            X = torch.fft.rfft(noise); fr = torch.arange(X.shape[0], device=self.device) * SR / self.n
            y = torch.fft.irfft(X * ((fr > 3000) & (fr < 16000)).float(), self.n)
            return y / (y.abs().max() + 1e-9)
        if kind == "tom":
            f = 120 * (1 + 0.5 * torch.exp(-t * 20))
            y = torch.sin(2 * math.pi * torch.cumsum(f, 0) / self.cfg.sr) * torch.exp(-t * 8)
            return y / (y.abs().max() + 1e-9)
        # perc: short filtered click
        noise = torch.randn(self.n, device=self.device, generator=self.g) * torch.exp(-t * 40)
        return noise / (noise.abs().max() + 1e-9)

    def _augment(self, x, pan):
        # distortion
        if float(self._rand(())) < 0.3:
            drive = float(self._rand((), 1, 30))
            x = torch.tanh(x * drive) / math.tanh(drive)
        # EQ: simple one-pole tilt via FFT (p=0.3)
        if float(self._rand(())) < 0.3:
            X = torch.fft.rfft(x)
            fr = torch.arange(X.shape[0], device=self.device) * SR / self.n
            shelf = torch.where(fr < 500, torch.tensor(float(self._rand((), 0.5, 2.0)), device=self.device),
                                torch.tensor(1.0, device=self.device))
            x = torch.fft.irfft(X * shelf, self.n)
        # chorus / delay
        if float(self._rand(())) < 0.2:
            d = int(float(self._rand((), 0.01, 0.06)) * SR)
            x = x + 0.35 * torch.roll(x, d)
        # reverb (FFT convolution with synthetic decaying IR)
        if float(self._rand(())) < 0.4:
            rt60 = float(self._rand((), 0.3, 3.0))
            ir_n = min(int(rt60 * SR), self.n)
            tt = torch.arange(ir_n, device=self.device) / SR
            ir = torch.randn(ir_n, device=self.device, generator=self.g) * torch.exp(-6.9 * tt / rt60)
            ir = ir / (ir.abs().sum() + 1e-9)
            wet = float(self._rand((), 0.1, 0.6))
            X = torch.fft.rfft(x, n=self.n)
            H = torch.fft.rfft(ir, n=self.n)
            x = (1 - wet) * x + wet * torch.fft.irfft(X * H, self.n)
        # speed +-2 semitones (labels handled by caller)
        speed = float(2.0 ** (float(self._rand((), -2, 2)) / 12.0))
        if abs(speed - 1.0) > 0.01:
            idx = torch.arange(self.n, device=self.device, dtype=torch.float32) * speed
            idx = torch.clamp(idx, 0, self.n - 1)
            i0 = idx.floor().long(); i1 = torch.clamp(i0 + 1, 0, self.n - 1)
            a = idx - i0
            x = x[i0] * (1 - a) + x[i1] * a
        return x, speed, pan

    # ------------------------------------------------------------- one batch
    def _one(self):
        dev = self.device
        K = int(torch.randint(2, self.cfg.max_sources + 1, (1,), device=dev, generator=self.g))
        mix = torch.zeros(self.n, 2, device=dev)
        src_scores = []
        src_env = []
        src_f0 = []
        src_kind = []
        for s in range(K):
            r = float(self._rand(()))
            if r < 0.6:
                kind = "tonal"
            elif r < 0.72:
                kind = "noise"
            else:
                kind = "drum"
            if kind == "drum":
                dkind = DRUMS[int(torch.randint(0, len(DRUMS), (1,), device=dev, generator=self.g))]
                tempo = float(self._rand((), 80, 160))
                dur = float(self._rand((), 1.0, self.cfg.seconds))
                gain = 10 ** (float(self._rand((), -24, -6)) / 20)
                x = self._drum_track(dkind, tempo, dur, gain)
                pan = float(self._rand((), -0.8, 0.8))
                f0s = []
                score = self._score_drum(dkind, tempo, dur, gain)
            elif kind == "noise":
                color = ["white", "pink", "brown", "band"][int(torch.randint(0, 4, (1,), device=dev, generator=self.g))]
                dur = float(self._rand((), 1.5, self.cfg.seconds))
                start = float(self._rand((), 0, max(0.0, self.cfg.seconds - dur)))
                gain = 10 ** (float(self._rand((), -30, -8)) / 20)
                x = self._noise(color, gain, dur, start)
                pan = float(self._rand((), -0.8, 0.8))
                f0s = []
                score = self._score_noise(start, dur, gain)
            else:
                timbre = TONAL[int(torch.randint(0, len(TONAL), (1,), device=dev, generator=self.g))]
                lo, hi = TIMBRE[timbre][0], TIMBRE[timbre][1]
                n_notes = 1 if float(self._rand(())) < 0.72 else int(torch.randint(2, 4, (1,), device=dev, generator=self.g))
                base = float(2 ** self._rand((), math.log2(lo), math.log2(hi)))
                f0s = [base]
                for j in range(n_notes - 1):
                    f0s.append(base * 2 ** (int(torch.randint(0, 12, (1,), device=dev, generator=self.g)) / 12.0))
                dur = float(self._rand((), 1.5, self.cfg.seconds))
                start = 0.0 if float(self._rand(())) < 0.7 else float(self._rand((), 0, max(0.0, self.cfg.seconds - dur)))
                gain = 10 ** (float(self._rand((), -30, 0)) / 20)
                attack = TIMBRE[timbre][5]
                core = self._chord(f0s, timbre, dur, 1.0, attack)
                x = self._place(core, start, gain)
                pan = float(self._rand((), -0.1, 0.1)) if timbre == "vocal_lead" else float(self._rand((), -1, 1))
                score = self._score_tonal(f0s, timbre, start, dur, gain, attack)
            x, speed, pan = self._augment(x, pan)
            if kind == "tonal":
                f0s = [f * speed for f in f0s]
                score = score  # score is band-domain; speed shift applied below
                if abs(speed - 1.0) > 0.01:
                    score = self._shift_score(score, speed)
            ang = (pan + 1) * math.pi / 4
            mix[:, 0] += x * math.cos(ang)
            mix[:, 1] += x * math.sin(ang)
            src_scores.append(score)
            src_env.append(torch.clamp(x.abs().mean(dim=0) if x.dim() > 1 else x.abs(), 0, 1))
            src_f0.append(f0s)
            src_kind.append(kind)
        # master: normalize RMS -20..-8 dBFS, soft limiter
        rms = mix.pow(2).mean().sqrt() + 1e-9
        target = 10 ** (float(self._rand((), -20, -8)) / 20)
        mix = mix * (target / rms)
        if float(self._rand(())) < 0.5:
            mix = torch.tanh(mix * 1.5) / 1.5
        return mix, src_scores, src_f0, src_kind

    def _score_tonal(self, f0s, timbre, start, dur, gain, attack):
        # [T,192] analytic dominant-source score
        lo, hi, tilt, nh, vib, _ = TIMBRE.get(timbre, TIMBRE["other_tonal"])
        t = torch.arange(self.T, device=self.device, dtype=torch.float32) * fe.HOP / SR
        active = ((t >= start) & (t < start + dur)).float()
        env = torch.clamp((t - start) / max(attack, 1e-3), 0, 1) * torch.clamp((start + dur - t) / 0.05, 0, 1)
        score = torch.zeros(self.T, fe.BANDS, device=self.device)
        for f0 in f0s:
            nmax = 1
            while nmax < nh and f0 * (nmax + 1) < 12000:
                nmax += 1
            for k in range(1, nmax + 1):
                mu = math.log(f0 * k)
                width = self.alpha_band * (1.0 + 0.12 * k)
                score += (1.0 / k ** (tilt * 0.7)) * torch.exp(
                    -0.5 * ((self.logfc - mu) / width) ** 2) / k ** 0.3
        return score * (active * env * gain)[:, None]

    def _score_noise(self, start, dur, gain):
        t = torch.arange(self.T, device=self.device, dtype=torch.float32) * fe.HOP / SR
        env = ((t >= start) & (t < start + dur)).float()
        return torch.ones(self.T, fe.BANDS, device=self.device) * (gain * env)[:, None]

    def _score_drum(self, kind, tempo, dur, gain):
        t = torch.arange(self.T, device=self.device, dtype=torch.float32) * fe.HOP / SR
        score = torch.zeros(self.T, fe.BANDS, device=self.device)
        beat = 60.0 / tempo
        step = beat / (2 if kind == "hihat" else 1)
        tt = 0.0
        weights = torch.ones(fe.BANDS, device=self.device)
        if kind in ("hihat", "cymbal"):
            weights = torch.linspace(0.2, 1.0, fe.BANDS, device=self.device)
        while tt < dur:
            score += torch.exp(-0.5 * ((t - tt) / 0.02) ** 2)[:, None] * weights[None, :]
            tt += step
        return score * gain

    def _shift_score(self, score, speed):
        # shift band axis by log2(speed)*bands/octave (approx 192/log2(16000/30)=36.8 bands/oct)
        shift = int(round(math.log2(speed) * (fe.BANDS - 1) / math.log2(fe.F_MAX / fe.F_MIN)))
        if shift == 0:
            return score
        out = torch.zeros_like(score)
        if shift > 0:
            out[:, shift:] = score[:, :-shift]
        else:
            out[:, :shift] = score[:, -shift:]
        return out

    def batch(self, batch_size: int, seed: int | None = None):
        if seed is not None:
            self.g.manual_seed(seed)
        mixes, doms, pitches, acts, sals, metas = [], [], [], [], [], []
        for b in range(batch_size):
            mix, scores, f0s, kinds = self._one()
            S = torch.stack(scores)  # [K,T,192]
            T = self.T
            total = S.sum(0) + 1e-8
            dom = S.argmax(0).long()
            pdom = S.max(0).values
            p = pdom.clamp_min(1e-9)
            rest = (total - pdom).clamp_min(1e-9)
            sal = torch.sigmoid((10 * torch.log10(p) - 10 * torch.log10(rest)) / 6.0)
            pitch = torch.zeros(T, fe.BANDS, device=self.device)
            for s, notes in enumerate(f0s):
                for f0 in notes:
                    mu = math.log(f0)
                    pitch = torch.maximum(pitch, torch.exp(-0.5 * ((self.logfc - mu) / 0.03) ** 2)[None, :])
            mixes.append(mix)
            doms.append(dom)
            pitches.append(pitch)
            sals.append(sal)
            metas.append({"kinds": kinds, "f0": f0s})
        mix = torch.stack(mixes)  # [B,n,2]
        # activity from mixed band power, computed by caller's frontend for exactness
        dom = torch.stack(doms)
        pitch = torch.stack(pitches).clamp(0, 1)
        sal = torch.stack(sals).clamp(0, 1)
        return {"mix": mix, "dom": dom, "pitch": pitch, "sal": sal, "meta": metas}


def make_labels(mix, dom, pitch, sal, tf: fe.TorchFrontend):
    """Compute activity from the mix through the exact FE-v1 frontend."""
    feat, p = tf.features(mix)
    dbm = 10 * torch.log10(p["M"] + 1e-12)
    mx = dbm.max(dim=2, keepdim=True).values
    thr = torch.maximum(torch.full_like(mx, -80.0), mx - 60.0)
    active = (dbm > thr).float()
    # weighted salience target only on active bands
    sal = sal * active
    return feat, active, pitch, sal, dom
