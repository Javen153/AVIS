"""v2 mixer: real stems (MUSDB, Slakh) mixed with procedural sources (SPEC F).

25% of examples stay fully procedural and use the analytic v1 labels. Other examples
take 2..8 sources by role (vocal/bass/drum/noise/tonal). Vocal and bass are always real
when the pool has them; tonal sources are real with p=0.7. Labels for mixed examples are
measured: P_M of every source through the FE-v1 band power, so dominant = argmax_s P_M^s
and sal = sigmoid((dB(P_dom) - dB(sum of others)) / 6), as in SPEC F.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import frontend as fe
from .mixer import DRUMS, TIMBRE, TONAL, ProceduralMixer
from .real_data import FRAMES, PITCH_LABELLED, SLOTS, RealSourceSampler

PROC_P = 0.25
ROLES = ["vocal", "bass", "drum", "noise", "tonal"]
ROLE_W = [0.15, 0.15, 0.25, 0.10, 0.35]
REAL_TONAL_P = 0.7
MAXK = 8
SEC_OUT = 192000
MIN_DB = -55.0      # a real chunk quieter than this is re-drawn (silence, muted stem)
GAIN_DB = (-40.0, 20.0)  # per-source gain clamp, so near-silent chunks stay inaudible
MIN_MIX_DB = -60.0  # an all-but-silent mixture is re-drawn instead of normalised up
RETRIES = 4


def rms_db(x: np.ndarray) -> float:
    """RMS level of a mono chunk in dBFS (used to reject silent real stems)."""
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12)


class RealMixer:
    def __init__(self, base: ProceduralMixer, sampler: RealSourceSampler | None, tf: fe.TorchFrontend,
                 proc_p: float = PROC_P, distort_p: float = 0.5):
        self.base = base
        self.sampler = sampler
        self.tf = tf
        self.proc_p = proc_p
        self.distort_p = distort_p
        self.dev = base.device
        self.n = base.n
        self.T = base.T
        self.logfc = base.logfc
        self.tonal_real = [c for c in (sampler.cats if sampler else []) if c not in ("vocal_lead", "bass")]
        self.stats = {"proc_examples": 0, "mixed_examples": 0, "real_sources": 0, "proc_sources": 0}

    # ------------------------------------------------------------ helpers
    def _pick(self, weights):
        r = float(self.base._rand(()))
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r < acc:
                return i
        return len(weights) - 1

    def _resample(self, x44: torch.Tensor) -> torch.Tensor:
        """44.1 -> 48 kHz by zero-padding the spectrum (exact for band-limited content)."""
        n_in = x44.shape[0]
        n_out = SEC_OUT
        X = torch.fft.rfft(x44)
        Y = torch.zeros(n_out // 2 + 1, dtype=X.dtype, device=X.device)
        m = min(X.shape[0], Y.shape[0])
        Y[:m] = X[:m]
        return torch.fft.irfft(Y, n_out) * (n_out / n_in)

    def _pad_frames(self, f0: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.T, SLOTS, device=self.dev)
        m = min(self.T, f0.shape[0])
        out[:m] = f0[:m]
        return out

    @staticmethod
    def _warp_f0(f0: torch.Tensor, speed: float) -> torch.Tensor:
        """Warp the f0 time axis with `_augment`'s speed read (audio is played at t*speed).

        A note at chunk time t0 in the original audio sounds at t0/speed after the speed
        change, so the pitch label at frame t must be f0(t*speed) (D-034).
        """
        if abs(speed - 1.0) <= 0.01:
            return f0
        t = torch.arange(f0.shape[0], device=f0.device, dtype=torch.float32) * speed
        idx = t.round().long().clamp_(0, f0.shape[0] - 1)
        return f0[idx]

    # ------------------------------------------------------------ sources
    def _real(self, cat: str):
        b = self.base
        audio, f0 = self.sampler.take(cat)
        for _ in range(RETRIES):
            if rms_db(audio) > MIN_DB:
                break
            audio, f0 = self.sampler.take(cat)
        x = self._resample(torch.from_numpy(audio).to(self.dev))
        f0 = self._pad_frames(torch.from_numpy(f0).to(self.dev))
        self.stats["real_sources"] += 1
        pan = float(b._rand((), -0.1, 0.1)) if cat == "vocal_lead" else float(b._rand((), -1, 1))
        if float(b._rand(())) < 0.3:
            d = int(float(b._rand((), 0.0, 2.0)) * fe.HOP * 100)
            x = torch.cat([torch.zeros(d, device=self.dev), x[: self.n - d]])
            k = d // fe.HOP
            f0 = torch.cat([torch.zeros(k, SLOTS, device=self.dev), f0[: self.T - k]])
        x, speed, pan = b._augment(x, pan)
        f0 = self._warp_f0(f0, speed) * speed
        return x, pan, f0, cat in PITCH_LABELLED

    def _proc_drum(self):
        b = self.base
        dkind = DRUMS[int(torch.randint(0, len(DRUMS), (1,), device=self.dev, generator=b.g))]
        tempo = float(b._rand((), 80, 160))
        dur = float(b._rand((), 1.0, b.cfg.seconds))
        x = b._drum_track(dkind, tempo, dur, 1.0)
        x, _, pan = b._augment(x, float(b._rand((), -0.8, 0.8)))
        return x, pan, torch.zeros(self.T, SLOTS, device=self.dev), True

    def _proc_noise(self):
        b = self.base
        color = ["white", "pink", "brown", "band"][int(torch.randint(0, 4, (1,), device=self.dev, generator=b.g))]
        dur = float(b._rand((), 1.5, b.cfg.seconds))
        start = float(b._rand((), 0, max(0.0, b.cfg.seconds - dur)))
        x = b._noise(color, 1.0, dur, start)
        x, _, pan = b._augment(x, float(b._rand((), -0.8, 0.8)))
        return x, pan, torch.zeros(self.T, SLOTS, device=self.dev), True

    def _proc_tonal(self, timbre: str | None = None):
        b = self.base
        if timbre is None:
            timbre = TONAL[int(torch.randint(0, len(TONAL), (1,), device=self.dev, generator=b.g))]
        lo, hi = TIMBRE[timbre][0], TIMBRE[timbre][1]
        n_notes = 1 if float(b._rand(())) < 0.72 else int(torch.randint(2, 4, (1,), device=self.dev, generator=b.g))
        base_f = float(2 ** b._rand((), math.log2(lo), math.log2(hi)))
        f0s = [base_f]
        for _ in range(n_notes - 1):
            f0s.append(base_f * 2 ** (int(torch.randint(0, 12, (1,), device=self.dev, generator=b.g)) / 12.0))
        dur = float(b._rand((), 1.5, b.cfg.seconds))
        start = 0.0 if float(b._rand(())) < 0.7 else float(b._rand((), 0, max(0.0, b.cfg.seconds - dur)))
        attack = TIMBRE[timbre][5]
        x = b._place(b._chord(f0s, timbre, dur, 1.0, attack), start, 1.0)
        pan = float(b._rand((), -0.1, 0.1)) if timbre == "vocal_lead" else float(b._rand((), -1, 1))
        x, speed, pan = b._augment(x, pan)
        t = torch.arange(self.T, device=self.dev, dtype=torch.float32) * fe.HOP / 48000.0
        on = ((t >= start) & (t < start + dur)).float()
        f0 = torch.zeros(self.T, SLOTS, device=self.dev)
        for j, f in enumerate(f0s[:SLOTS]):
            f0[:, j] = on * f
        f0 = self._warp_f0(f0, speed) * speed
        self.stats["proc_sources"] += 1
        return x, pan, f0, True

    def _source(self, role: str):
        s = self.sampler
        if role == "vocal":
            if s is not None and s.has("vocal_lead"):
                return self._real("vocal_lead")
            return self._proc_tonal("vocal_lead")
        if role == "bass":
            if s is not None and s.has("bass"):
                return self._real("bass")
            return self._proc_tonal("bass")
        if role == "drum":
            return self._proc_drum()
        if role == "noise":
            return self._proc_noise()
        if s is not None and self.tonal_real and float(self.base._rand(())) < REAL_TONAL_P:
            cat = self.tonal_real[int(float(self.base._rand(())) * len(self.tonal_real)) % len(self.tonal_real)]
            return self._real(cat)
        return self._proc_tonal(None)

    # ------------------------------------------------------------ examples
    def _mixed(self):
        b = self.base
        K = int(torch.randint(2, MAXK + 1, (1,), device=self.dev, generator=b.g))
        srcs = []
        for _ in range(K):
            role = ROLES[self._pick(ROLE_W)]
            srcs.append(self._source(role))
        rms = torch.stack([x.pow(2).mean().sqrt() + 1e-9 for x, _, _, _ in srcs])
        db = 20 * torch.log10(rms)
        loud = float(db.max())
        target = [loud - float(b._rand((), 0, 30)) for _ in srcs]
        if float(b._rand(())) < 0.3 and K >= 2:
            k = int(torch.randint(0, K, (1,), device=self.dev, generator=b.g))
            target[k] = loud - float(b._rand((), 18, 26))
        stereo, f0s, known = [], [], []
        lo, hi = 10 ** (GAIN_DB[0] / 20.0), 10 ** (GAIN_DB[1] / 20.0)
        for (x, pan, f0, ok), tgt, cur in zip(srcs, target, db.tolist()):
            # Clamp the gain: a digital-silence or very quiet chunk otherwise needs >100 dB
            # of gain and turns its own noise floor into a loud, mislabelled source.
            g = min(max(10 ** ((tgt - cur) / 20.0), lo), hi)
            ang = (pan + 1.0) * math.pi / 4.0
            stereo.append(torch.stack([x * g * math.cos(ang), x * g * math.sin(ang)], dim=1))
            f0s.append(f0)
            known.append(bool(ok))
        src = torch.stack(stereo)  # [K,n,2]
        mix = src.sum(0)
        rms_m = mix.pow(2).mean().sqrt() + 1e-9
        if float(20 * torch.log10(rms_m)) < MIN_MIX_DB:
            return None  # every source was effectively silent: draw another example
        self.stats["mixed_examples"] += 1
        mix = mix * (10 ** (float(b._rand((), -20, -8)) / 20) / rms_m)
        if float(b._rand(())) < self.distort_p:
            mix = torch.tanh(mix * 1.5) / 1.5
        return mix, src, torch.stack(f0s), known

    def _labels(self, src, f0s, known):
        """Measured dominance/salience (FE-v1 band power per source) and pitch labels.

        `known` is False for sources whose notes are not a reliable fundamental
        (fx/other_tonal/percussive). Frames where such a source is audible get
        pitch_mask=0 instead of a invented zero label (D-035).
        """
        P = self.tf.band_power(src)["M"]  # [K,T,192]
        tot = P.sum(0) + 1e-8
        dom = P.argmax(0).long()
        pdom = P.max(0).values.clamp_min(1e-9)
        rest = (tot - pdom).clamp_min(1e-9)
        sal = torch.sigmoid((10 * torch.log10(pdom) - 10 * torch.log10(rest)) / 6.0)
        if P.shape[0] >= 2:
            top2 = torch.topk(P, 2, dim=0).values
            margin = (10 * torch.log10((top2[0] + 1e-9) / (top2[1] + 1e-9))).clamp(max=60.0)
        else:
            # a solo source has no runner-up: every active cell is unambiguous
            margin = torch.full_like(pdom, 60.0)
        lf = torch.log(f0s.clamp_min(1.0))
        g = torch.exp(-0.5 * ((self.logfc[None, None, None, :] - lf[..., None]) / 0.03) ** 2)
        g = g * (f0s > 0).float()[..., None]
        pitch = g.amax(dim=(0, 2))
        # frame presence from per-source energy on the FE hop grid
        e = src.detach().float().pow(2).mean(dim=-1).reshape(src.shape[0], self.T, -1).mean(-1)
        present = e > (e.amax(dim=1, keepdim=True) * 1e-3).clamp_min(1e-9)
        ok = torch.tensor(known, device=self.dev, dtype=torch.bool)
        unknown = present & ~ok[:, None]
        pitch_mask = (present.any(0) & ~unknown.any(0)).float()
        return dom, pitch, sal, margin, pitch_mask

    def _analytic(self, scores, f0s):
        """v1 labels for fully procedural examples (unchanged content, plus margin)."""
        S = torch.stack(scores)
        total = S.sum(0) + 1e-8
        dom = S.argmax(0).long()
        pdom = S.max(0).values
        p = pdom.clamp_min(1e-9)
        rest = (total - pdom).clamp_min(1e-9)
        sal = torch.sigmoid((10 * torch.log10(p) - 10 * torch.log10(rest)) / 6.0)
        top2 = torch.topk(S, 2, dim=0).values
        margin = (10 * torch.log10((top2[0] + 1e-9) / (top2[1] + 1e-9))).clamp(max=60.0)
        pitch = torch.zeros(self.T, fe.BANDS, device=self.dev)
        for notes in f0s:
            for f0 in notes:
                mu = math.log(f0)
                pitch = torch.maximum(pitch, torch.exp(-0.5 * ((self.logfc - mu) / 0.03) ** 2)[None, :])
        return dom, pitch.clamp(0, 1), sal, margin

    def batch(self, B: int):
        mixes, doms, pitches, sals, margins, pmasks = [], [], [], [], [], []
        for _ in range(B):
            if float(self.base._rand(())) < self.proc_p or self.sampler is None:
                mix, scores, f0s, _ = self.base._one()
                dom, pitch, sal, margin = self._analytic(scores, f0s)
                pmask = torch.ones(self.T, device=self.dev)
                self.stats["proc_examples"] += 1
            else:
                mixed = None
                for _ in range(RETRIES):
                    mixed = self._mixed()
                    if mixed is not None:
                        break
                if mixed is None:
                    mix, scores, f0s, _ = self.base._one()
                    dom, pitch, sal, margin = self._analytic(scores, f0s)
                    pmask = torch.ones(self.T, device=self.dev)
                    self.stats["proc_examples"] += 1
                else:
                    mix, src, f0s, known = mixed
                    dom, pitch, sal, margin, pmask = self._labels(src, f0s, known)
            mixes.append(mix)
            doms.append(dom)
            pitches.append(pitch)
            sals.append(sal.clamp(0, 1))
            margins.append(margin)
            pmasks.append(pmask)
        return {"mix": torch.stack(mixes), "dom": torch.stack(doms),
                "pitch": torch.stack(pitches).clamp(0, 1), "sal": torch.stack(sals),
                "margin": torch.stack(margins), "pitch_mask": torch.stack(pmasks),
                "meta": None}
