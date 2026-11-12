"""M-v3 mixer: stereo real stems + targeted hard cases for S04/S09/S13.

Extends the v2 real/procedural mixer without changing it. Real chunks are read
as stereo, so the side channel at training time matches the inference/eval
domain. A configurable share of examples is drawn from four hard-case families:

- close_pair: two independent tones with controlled f0 (1..36 semitones apart),
  different timbres/levels. Half of the pairs are co-located (same pan, often
  both centred): a pair panned hard left/right teaches the embedding to use the
  side channel as a shortcut, which then collapses on mono scenario audio
  (S04/S06/S10, D-038).
- triad: 2-3 simultaneous tonal sources at one pan, one of them often entering
  later; direct training signal for the S04 (300/1700 Hz) and S06 (C-E-G)
  layouts, where several pitches share a single mid channel.
- pitch_noise: one tonal source at -18..-3 dB SNR under white/pink/band noise;
  the pitch target stays the clean f0, so the head learns harmonic structure
  below the noise floor (S10, S13).
- drum_bed: kick+snare+hihat at the same tempo with overlapping decays, so the
  noise part of the snare does not collapse into the hihat cluster (S09).
- solo: one source alone, either a bright additive series up to ~10 kHz or
  coloured noise (S05/S07). Every other family has K>=2 sources, so without
  this one the model never learns that all bands of a wideband source belong
  to a single object (D-040).

Labels remain measured through FE-v1 (as in v2): dominance from per-source P_M,
salience from the dominant/rest dB ratio, pitch from f0 gaussians.
"""
from __future__ import annotations

import math

import torch

from . import frontend as fe
from .mixer import DRUMS, TIMBRE, TONAL, ProceduralMixer
from .mixer_v2 import (GAIN_DB, MAXK, MIN_DB, MIN_MIX_DB, PROC_P, RETRIES,
                       ROLE_W, ROLES, RealMixer, rms_db)
from .real_data import PITCH_LABELLED, SLOTS

# A pure sine timbre for S01/S04/S13. Not added to TONAL, so the v2 procedural
# mixer keeps drawing exactly the same timbres as before.
TIMBRE["sine"] = (40.0, 4000.0, 1.0, 1, 0, 0.005)

# Additive 1/n series up to 12 kHz (the S05/S11 stimulus, D-040). `nh` is the
# harmonic cap inside ProceduralMixer._tone; no existing timbre reaches past
# ~2.2 kHz, so high bands of a single source were never part of a training mix.
TIMBRE["saw_bright"] = (60.0, 2000.0, 1.0, 64, 0, 0.004)


class RealMixerV3(RealMixer):
    def __init__(self, base: ProceduralMixer, sampler, tf: fe.TorchFrontend,
                 proc_p: float = PROC_P, distort_p: float = 0.5, hard_p: float = 0.35,
                 stereo: bool = True):
        super().__init__(base, sampler, tf, proc_p=proc_p, distort_p=distort_p)
        self.hard_p = hard_p
        self.stereo = stereo
        self.stats.update({"hard_examples": 0, "hard_miss": 0, "close_pair": 0,
                           "triad": 0, "pitch_noise": 0, "drum_bed": 0, "solo": 0,
                           "stereo_sources": 0})

    # ------------------------------------------------------------ real stems
    def _real(self, cat: str):
        b = self.base
        audio, f0 = self.sampler.take(cat)
        for _ in range(RETRIES):
            if rms_db(audio) > MIN_DB:
                break
            audio, f0 = self.sampler.take(cat)
        xt = torch.from_numpy(audio).to(self.dev)
        if xt.dim() == 2:
            x = torch.stack([self._resample(xt[:, c]) for c in range(xt.shape[1])], dim=1)
            self.stats["stereo_sources"] += 1
        else:
            x = self._resample(xt)
        f0 = self._pad_frames(torch.from_numpy(f0).to(self.dev))
        if float(b._rand(())) < 0.3:
            d = int(float(b._rand((), 0.0, 2.0)) * fe.HOP * 100)
            x = torch.cat([torch.zeros((d,) + x.shape[1:], device=self.dev), x[: self.n - d]])
            k = d // fe.HOP
            f0 = torch.cat([torch.zeros(k, SLOTS, device=self.dev), f0[: self.T - k]])
        if x.dim() == 1:
            x, speed, pan = b._augment(x, float(b._rand((), -1, 1)))
        else:
            x, speed, pan = self._augment_stereo(x)
        f0 = self._warp_f0(f0, speed) * speed
        self.stats["real_sources"] += 1
        return x, pan, f0, cat in PITCH_LABELLED

    def _augment_stereo(self, x: torch.Tensor):
        """Same speed/gain/width family for both channels (keeps the stereo image)."""
        b = self.base
        speed = float(2.0 ** (float(b._rand((), -2, 2)) / 12.0))
        if abs(speed - 1.0) > 0.01:
            idx = torch.arange(self.n, device=self.dev, dtype=torch.float32) * speed
            idx = torch.clamp(idx, 0, self.n - 1)
            i0 = idx.floor().long()
            i1 = torch.clamp(i0 + 1, 0, self.n - 1)
            a = (idx - i0)[:, None]
            x = x[i0] * (1 - a) + x[i1] * a
        if float(b._rand(())) < 0.25:
            drive = float(b._rand((), 1, 20))
            x = torch.tanh(x * drive) / math.tanh(drive)
        width = float(b._rand((), 0.4, 1.6))
        rot = float(b._rand((), -0.25, 0.25))
        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.5 * (x[:, 0] - x[:, 1]) * width
        left, right = mid + side, mid - side
        c, s = math.cos(rot), math.sin(rot)
        return torch.stack([c * left + s * right, -s * left + c * right], dim=1), speed, 0.0

    # ------------------------------------------------------------ combination
    def _combine(self, srcs, relevel: bool = True):
        """Sum per-source stereo signals and normalise the mix level.

        ``relevel=False`` keeps the levels that a hard-case builder designed
        (SNR under noise, relative tone levels), so the curriculum is not
        overwritten by the generic random-gain draw.
        """
        b = self.base
        if not srcs:
            return None
        rms = torch.stack([x.pow(2).mean().sqrt() + 1e-9 for x, _, _, _ in srcs])
        db = 20 * torch.log10(rms)
        loud = float(db.max())
        if relevel:
            target = [loud - float(b._rand((), 0, 30)) for _ in srcs]
            if float(b._rand(())) < 0.3 and len(srcs) >= 2:
                k = int(torch.randint(0, len(srcs), (1,), device=self.dev, generator=b.g))
                target[k] = loud - float(b._rand((), 18, 26))
        else:
            target = db.tolist()
        stereo, f0s, known = [], [], []
        lo, hi = 10 ** (GAIN_DB[0] / 20.0), 10 ** (GAIN_DB[1] / 20.0)
        for (x, pan, f0, ok), tgt, cur in zip(srcs, target, db.tolist()):
            g = min(max(10 ** ((tgt - cur) / 20.0), lo), hi)
            if x.dim() == 1:
                ang = (pan + 1.0) * math.pi / 4.0
                s = torch.stack([x * g * math.cos(ang), x * g * math.sin(ang)], dim=1)
            else:
                s = x * g
            stereo.append(s)
            f0s.append(f0)
            known.append(bool(ok))
        src = torch.stack(stereo)  # [K,n,2]
        mix = src.sum(0)
        rms_m = mix.pow(2).mean().sqrt() + 1e-9
        if float(20 * torch.log10(rms_m)) < MIN_MIX_DB:
            return None
        mix = mix * (10 ** (float(b._rand((), -20, -8)) / 20) / rms_m)
        if float(b._rand(())) < self.distort_p:
            mix = torch.tanh(mix * 1.5) / 1.5
        return mix, src, torch.stack(f0s), known

    def _mixed(self):
        b = self.base
        K = int(torch.randint(2, MAXK + 1, (1,), device=self.dev, generator=b.g))
        srcs = []
        for _ in range(K):
            role = ROLES[self._pick(ROLE_W)]
            srcs.append(self._source(role))
        out = self._combine(srcs)
        if out is not None:
            self.stats["mixed_examples"] += 1
        return out

    # ------------------------------------------------------------ hard cases
    def _tone_src(self, f0: float, timbre: str, dur: float, start: float,
                  gain: float, pan: float):
        b = self.base
        # build at unit level, augment, then apply the designed gain: distortion and
        # reverb inside _augment would otherwise change the intended relative levels
        x = b._place(b._tone(f0, timbre, dur, 1.0, TIMBRE[timbre][5]), start, 1.0)
        x, speed, pan = b._augment(x, pan)
        x = x * gain
        f0w = f0 * speed
        t = torch.arange(self.T, device=self.dev, dtype=torch.float32) * fe.HOP / 48000.0
        tt = t * speed
        on = ((tt >= start) & (tt < start + dur)).float()
        f0t = torch.zeros(self.T, SLOTS, device=self.dev)
        f0t[:, 0] = on * f0w
        return x, pan, f0t, True

    def _hard_close_pair(self):
        b = self.base
        if float(b._rand(())) < 0.6:
            timbre = "sine"
            lo, hi = 60.0, 3000.0
        else:
            timbre = TONAL[int(torch.randint(0, len(TONAL), (1,), device=self.dev, generator=b.g))]
            lo, hi = float(TIMBRE[timbre][0]), float(TIMBRE[timbre][1])
        f1 = 2 ** float(b._rand((), math.log2(lo), math.log2(hi)))
        semis = int(torch.randint(0, 2, (1,), device=self.dev, generator=b.g))
        step = [1, 2, 3, 4, 5, 7, 12, 19, 24, 31, 36][int(b._rand((), 0, 11))]
        semis = step if semis == 1 else -step
        f2 = min(max(f1 * 2 ** (semis / 12.0), 30.0), 8000.0)
        dur = float(b._rand((), 2.5, b.cfg.seconds))
        start1 = 0.0 if float(b._rand(())) < 0.7 else float(b._rand((), 0, 0.8))
        start2 = float(b._rand((), 0, 1.2))
        g = 10 ** (-15 / 20)
        g1 = g * 10 ** (float(b._rand((), -1.5, 1.5)) / 20)
        g2 = g * 10 ** (float(b._rand((), -1.5, 1.5)) / 20)
        if float(b._rand(())) < 0.55:
            # co-located pair: no side-channel cue, separation must come from
            # the spectrum (the S04/S06/S10 setting, D-038)
            if float(b._rand(())) < 0.6:
                p1 = p2 = 0.0
            else:
                p1 = p2 = float(b._rand((), -0.3, 0.3))
        else:
            p1 = float(b._rand((), -0.9, -0.2))
            p2 = float(b._rand((), 0.2, 0.9))
        srcs = [self._tone_src(f1, timbre, dur, start1, g1, p1),
                self._tone_src(f2, timbre, dur, start2, g2, p2)]
        if float(b._rand(())) < 0.25:
            n = b._noise("pink", 10 ** (-32 / 20), dur, 0.0)
            srcs.append((n, p1, torch.zeros(self.T, SLOTS, device=self.dev), True))
        self.stats["close_pair"] += 1
        return self._combine(srcs, relevel=False)

    def _hard_triad(self):
        """Two or three simultaneous tones sharing one pan position.

        S04, S06 and S10 put all sources in the mid channel, so the embedding
        has to separate them by spectral structure alone. The root and the
        interval set cover both a major triad (4 and 7 semitones) and the wide
        S04 pair (30 semitones); one source may start later, as in S04.
        """
        b = self.base
        k = 3 if float(b._rand(())) < 0.65 else 2
        if float(b._rand(())) < 0.5:
            timbre = "sine"
        else:
            timbre = TONAL[int(torch.randint(0, len(TONAL), (1,), device=self.dev,
                                              generator=b.g))]
        pan = 0.0 if float(b._rand(())) < 0.7 else float(b._rand((), -0.3, 0.3))
        root = 2 ** float(b._rand((), math.log2(150.0), math.log2(600.0)))
        steps = [0, 4, 7, 12, 3, 8, 5, 19, 24, 30]
        picks = [0]
        while len(picks) < k:
            s = steps[int(torch.randint(0, len(steps), (1,), device=self.dev,
                                        generator=b.g))]
            if all(abs(s - q) >= 3 for q in picks):
                picks.append(s)
        srcs = []
        for i, s in enumerate(picks):
            f0 = min(max(root * 2 ** (s / 12.0), 30.0), 6000.0)
            dur = float(b._rand((), 2.5, b.cfg.seconds))
            start = 0.0 if i == 0 or float(b._rand(())) < 0.5 else \
                float(b._rand((), 0.2, 1.2))
            gain = 10 ** (-15 / 20) * 10 ** (float(b._rand((), -1.5, 1.5)) / 20)
            srcs.append(self._tone_src(f0, timbre, dur, start, gain, pan))
        if float(b._rand(())) < 0.3:
            n = b._noise("pink", 10 ** (-34 / 20), b.cfg.seconds, 0.0)
            srcs.append((n, pan, torch.zeros(self.T, SLOTS, device=self.dev), True))
        self.stats["triad"] += 1
        return self._combine(srcs, relevel=False)

    def _hard_pitch_noise(self):
        b = self.base
        timbre = ["synth_lead", "strings", "vocal_lead"][
            int(torch.randint(0, 3, (1,), device=self.dev, generator=b.g))]
        f0 = float(2 ** b._rand((), math.log2(70.0), math.log2(1200.0)))
        dur = float(b._rand((), 2.5, b.cfg.seconds))
        start = float(b._rand((), 0, 0.6))
        x = b._place(b._tone(f0, timbre, dur, 1.0, TIMBRE[timbre][5]), start, 1.0)
        color = ["white", "pink", "brown", "band"][
            int(torch.randint(0, 4, (1,), device=self.dev, generator=b.g))]
        noise = b._noise(color, 1.0, dur, start)
        # augment first, then set the noise level: distortion/reverb change each
        # source's gain by up to ~15 dB, so scaling before augmentation loses the
        # realised SNR (D-037)
        x, speed, pan = b._augment(x, float(b._rand((), -0.4, 0.4)))
        noise, _, npan = b._augment(noise, float(b._rand((), -0.6, 0.6)))
        rt = float(x.pow(2).mean().sqrt()) + 1e-9
        rn = float(noise.pow(2).mean().sqrt()) + 1e-9
        snr = float(b._rand((), -18, -3))
        # noise is scaled to the tone RMS and then to the target signal-to-noise
        # ratio, which is negative: the tone sits below the noise floor
        noise = noise * (rt / rn) * 10 ** (-snr / 20.0)
        f0w = f0 * speed
        t = torch.arange(self.T, device=self.dev, dtype=torch.float32) * fe.HOP / 48000.0
        on = (((t * speed) >= start) & ((t * speed) < start + dur)).float()
        f0t = torch.zeros(self.T, SLOTS, device=self.dev)
        f0t[:, 0] = on * f0w
        self.stats["pitch_noise"] += 1
        return self._combine([(x, pan, f0t, True),
                              (noise, npan, torch.zeros(self.T, SLOTS, device=self.dev), True)],
                             relevel=False)

    def _hard_drum_bed(self):
        b = self.base
        tempo = float(b._rand((), 90, 150))
        dur = float(b._rand((), 3.0, b.cfg.seconds))
        srcs = []
        for kind, pan in (("kick", -0.3), ("snare", 0.1), ("hihat", 0.5)):
            g = 10 ** (float(b._rand((), -17, -10)) / 20)
            x = b._drum_track(kind, tempo, dur, g)
            srcs.append((x, pan, torch.zeros(self.T, SLOTS, device=self.dev), True))
        if float(b._rand(())) < 0.4:
            f0 = float(2 ** b._rand((), math.log2(55.0), math.log2(220.0)))
            srcs.append(self._tone_src(f0, "bass", dur, 0.0,
                                       10 ** (-20 / 20), 0.0))
        self.stats["drum_bed"] += 1
        return self._combine(srcs, relevel=False)

    def _hard_solo(self):
        """One source alone: the wideband case every other family is missing.

        S05 (additive saw to 10 kHz) and S07 (white noise) are single sources,
        but `_mixed`, `_one` and every hard family build K>=2 sources. The
        embedding therefore treats a bright high-band region as a second
        source (D-040). A bright 1/n series up to 12 kHz and coloured noise
        teach act/embedding continuity across the whole band span.
        """
        b = self.base
        dur = float(b._rand((), 2.5, b.cfg.seconds))
        start = 0.0 if float(b._rand(())) < 0.8 else float(b._rand((), 0, 0.6))
        pan = float(b._rand((), -0.5, 0.5))
        if float(b._rand(())) < 0.6:
            f0 = float(2 ** b._rand((), math.log2(70.0), math.log2(900.0)))
            src = self._tone_src(f0, "saw_bright", dur, start, 10 ** (-12 / 20), pan)
        else:
            color = ["white", "pink", "brown", "band"][
                int(torch.randint(0, 4, (1,), device=self.dev, generator=b.g))]
            x = b._noise(color, 10 ** (-16 / 20), dur, start)
            x, _, pan = b._augment(x, pan)
            src = (x, pan, torch.zeros(self.T, SLOTS, device=self.dev), True)
        self.stats["solo"] += 1
        return self._combine([src], relevel=False)

    def _hard_case(self):
        r = float(self.base._rand(()))
        if r < 0.35:
            out = self._hard_close_pair()
        elif r < 0.45:
            out = self._hard_triad()
        elif r < 0.65:
            out = self._hard_pitch_noise()
        elif r < 0.85:
            out = self._hard_solo()
        else:
            out = self._hard_drum_bed()
        if out is not None:
            self.stats["hard_examples"] += 1
        return out

    # ------------------------------------------------------------ batch
    def batch(self, B: int):
        mixes, doms, pitches, sals, margins, pmasks, solos = [], [], [], [], [], [], []
        for _ in range(B):
            want_hard = float(self.base._rand(())) < self.hard_p
            hard = self._hard_case() if want_hard else None
            if hard is not None:
                mix, src, f0s, known = hard
                dom, pitch, sal, margin, pmask = self._labels(src, f0s, known)
                solos.append(1.0 if src.shape[0] == 1 else 0.0)
            else:
                solos.append(0.0)
                if want_hard:
                    # builder rejected the example (too quiet / mix under floor):
                    # count it so the effective hard share is visible
                    self.stats["hard_miss"] += 1
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
            sals.append(sal)
            margins.append(margin)
            pmasks.append(pmask)
        return {"mix": torch.stack(mixes), "dom": torch.stack(doms),
                "pitch": torch.stack(pitches).clamp(0, 1), "sal": torch.stack(sals),
                "margin": torch.stack(margins), "pitch_mask": torch.stack(pmasks),
                "solo": torch.tensor(solos, device=self.dev), "meta": None}
