"""FE-v1 frontend: reference implementation (numpy streaming + torch batch).

The C# runtime must match this exactly. All weight matrices are generated here
and stored in shared/frontend/v1/weights.bin with a documented layout.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

SR = 48000
HOP = 480
BANDS = 192
F_MIN = 30.0
F_MAX = 16000.0
RESOLUTIONS = (4096, 2048, 1024, 512)
MAGIC = 0x41564657  # 'AVFW'


def band_centers() -> np.ndarray:
    k = np.arange(BANDS, dtype=np.float64)
    return F_MIN * (F_MAX / F_MIN) ** (k / (BANDS - 1))


def band_resolution(fc: np.ndarray) -> np.ndarray:
    n = np.full(fc.shape, 512, dtype=np.int32)
    n[fc < 5000] = 1024
    n[fc < 1600] = 2048
    n[fc < 400] = 4096
    return n


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def build_weights():
    """Return list of dicts: nfft, bins(int32[]), weights(float32[])."""
    fc = band_centers()
    ext = np.empty(BANDS + 2)
    ext[1:-1] = fc
    # Reflect edges in log-frequency: fc[-1] = fc0^2/fc1, fc[F] = fc[-1]^2/fc[-2].
    ext[0] = fc[0] * fc[0] / fc[1]
    ext[-1] = fc[-1] * fc[-1] / fc[-2]
    logfc = np.log(fc)
    per_band = []
    for k in range(BANDS):
        nfft = int(band_resolution(fc[k:k + 1])[0])
        n_bins = nfft // 2 + 1
        freqs = np.arange(n_bins, dtype=np.float64) * SR / nfft
        logf = np.log(np.maximum(freqs, 1e-9))
        lo, mid, hi = ext[k], ext[k + 1], ext[k + 2]
        w = np.zeros(n_bins, dtype=np.float64)
        sel = (freqs > lo) & (freqs < hi)
        left = sel & (freqs <= mid)
        right = sel & (freqs > mid)
        w[left] = (logf[left] - math.log(lo)) / (math.log(mid) - math.log(lo))
        w[right] = (math.log(hi) - logf[right]) / (math.log(hi) - math.log(mid))
        idx = np.nonzero(w > 0)[0]
        binw = SR / nfft
        scale_cont = (hi - lo) / (2.0 * binw)
        if idx.size < 2:
            # Linear interpolation of power between the two nearest bins.
            j1 = int(np.searchsorted(freqs, mid))
            j0 = max(j1 - 1, 0)
            j1 = min(j0 + 1, n_bins - 1)
            if freqs[j1] == freqs[j0]:
                a = 0.0
            else:
                a = (mid - freqs[j0]) / (freqs[j1] - freqs[j0])
            idx = np.array([j0, j1], dtype=np.int64)
            ww = np.array([1.0 - a, a], dtype=np.float64)
            ww = ww * scale_cont
        else:
            ww = w[idx]
            # Match the continuous triangle integral (dense limit) so bands with
            # only 2-3 bins do not step at FFT-resolution borders (DECISIONS D-004).
            ww = ww * (scale_cont / ww.sum())
        per_band.append({
            "nfft": nfft,
            "bins": idx.astype(np.int32),
            "weights": ww.astype(np.float32),
        })
    return per_band


def save_weights(out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_band = build_weights()
    fc = band_centers()
    blob = bytearray()
    blob += struct.pack("<III", MAGIC, 1, BANDS)
    desc = []
    for b in per_band:
        bins = b["bins"]
        w = b["weights"]
        blob += struct.pack("<ii", b["nfft"], len(bins))
        blob += bins.astype("<i4").tobytes()
        blob += w.astype("<f4").tobytes()
        desc.append({
            "nfft": b["nfft"],
            "count": int(len(bins)),
            "bins": bins.tolist(),
            "weights": [float(x) for x in w],
        })
    (out / "weights.bin").write_bytes(bytes(blob))
    meta = {
        "version": "FE-v1",
        "sr": SR,
        "hop": HOP,
        "bands": BANDS,
        "fmin": F_MIN,
        "fmax": F_MAX,
        "center_hz": fc.tolist(),
        "resolutions": [4096, 2048, 1024, 512],
        "window": "hann_periodic",
        "power": "|rfft(x*w)|^2 / (sum(w))^2",
        "channels": ["M=(L+R)/2", "S=(L-R)/2", "L", "R"],
        "features": [
            "(dB(P_M)+60)/30",
            "(dB(P_S)+60)/30",
            "clip((dB(P_L)-dB(P_R))/20,-1,1)",
        ],
        "weights_layout": (
            "little-endian: uint32 magic=0x41564657, uint32 version=1, "
            "uint32 bands; then per band: int32 nfft, int32 count, "
            "int32[count] bin indices (rfft bins), float32[count] weights. "
            "Band power = sum(weights[i] * bin_power[bins[i]])."
        ),
        "bands_description": desc,
    }
    (out / "frontend.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_weights(path: str | Path):
    raw = Path(path).read_bytes()
    magic, ver, bands = struct.unpack_from("<III", raw, 0)
    if magic != MAGIC or ver != 1 or bands != BANDS:
        raise ValueError("bad weights.bin header")
    off = 12
    per_band = []
    for _ in range(BANDS):
        nfft, count = struct.unpack_from("<ii", raw, off)
        off += 8
        bins = np.frombuffer(raw, dtype="<i4", count=count, offset=off).copy()
        off += 4 * count
        w = np.frombuffer(raw, dtype="<f4", count=count, offset=off).copy()
        off += 4 * count
        per_band.append((nfft, bins.astype(np.int64), w.astype(np.float64)))
    return per_band


class NumpyFrontend:
    """Causal streaming frontend. push() accepts arbitrary stereo blocks."""

    def __init__(self, weights_path: str | Path):
        self.per_band = load_weights(weights_path)
        self.fc = band_centers()
        # history represents samples [-N0+HOP, 0) so the first frame ends at HOP
        self._hist = np.zeros((2, RESOLUTIONS[0] - HOP), dtype=np.float64)
        self._t = 0
        self._windows = {n: _hann(n) for n in RESOLUTIONS}

    def reset(self):
        self._hist[:] = 0.0
        self._t = 0

    def push(self, stereo: np.ndarray):
        """stereo [n,2] float; returns list of frames (one per hop)."""
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim == 1:
            stereo = np.column_stack([stereo, stereo])
        if stereo.shape[0] == 0:
            return []
        buf = np.concatenate([self._hist, stereo.T], axis=1)
        out = []
        while buf.shape[1] >= RESOLUTIONS[0]:
            chunk = buf[:, :RESOLUTIONS[0]]
            out.append(self._frame(chunk))
            buf = buf[:, HOP:]
            self._t += 1
        self._hist = buf.copy()
        return out

    def _frame(self, chunk: np.ndarray):
        L = chunk[0]
        R = chunk[1]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)
        pm = np.zeros(BANDS)
        ps = np.zeros(BANDS)
        pl = np.zeros(BANDS)
        pr = np.zeros(BANDS)
        cache = {}
        for k, (nfft, bins, w) in enumerate(self.per_band):
            if nfft not in cache:
                win = self._windows[nfft]
                seg = chunk[:, -nfft:]
                sw = win.sum()
                cache[nfft] = {}
                for name, sig in (("M", M), ("S", S), ("L", L), ("R", R)):
                    spec = np.fft.rfft(sig[-nfft:] * win)
                    cache[nfft][name] = (spec.real ** 2 + spec.imag ** 2) / (sw * sw)
            pw = cache[nfft]
            pm[k] = float(np.dot(w, pw["M"][bins]))
            ps[k] = float(np.dot(w, pw["S"][bins]))
            pl[k] = float(np.dot(w, pw["L"][bins]))
            pr[k] = float(np.dot(w, pw["R"][bins]))
        return make_frame(self._t, pm, ps, pl, pr, self.fc)


def make_frame(t, pm, ps, pl, pr, fc):
    eps = 1e-12
    dbm = 10.0 * np.log10(pm + eps)
    dbs = 10.0 * np.log10(ps + eps)
    dbl = 10.0 * np.log10(pl + eps)
    dbr = 10.0 * np.log10(pr + eps)
    feat0 = (dbm + 60.0) / 30.0
    feat1 = (dbs + 60.0) / 30.0
    feat2 = np.clip((dbl - dbr) / 20.0, -1.0, 1.0)
    return {
        "t": t * HOP / SR,
        "bandsDb_M": dbm,
        "P_M": pm,
        "P_L": pl,
        "P_R": pr,
        "features": np.stack([feat0, feat1, feat2]).astype(np.float32),
    }


# ---------------------------------------------------------------- torch batch

def torch_stft_power(x, nfft):
    """x [B,n] or [B,T,H] float32 -> [B,T,nfft//2+1] power, causal windows."""
    import torch
    if x.dim() == 2:
        B, n = x.shape
        T = n // HOP
    else:
        B, T, _ = x.shape
    pad = nfft - HOP
    xp = torch.nn.functional.pad(x.reshape(B, T * HOP), (pad, 0))
    win = torch.hann_window(nfft, periodic=True, dtype=x.dtype, device=x.device)
    spec = torch.stft(xp, nfft, hop_length=HOP, window=win, center=False,
                      return_complex=True, pad_mode="constant")
    # spec [B, nfft//2+1, T']; crop/pad to T
    if spec.shape[-1] > T:
        spec = spec[..., :T]
    elif spec.shape[-1] < T:
        spec = torch.nn.functional.pad(spec, (0, T - spec.shape[-1]))
    pw = spec.real ** 2 + spec.imag ** 2
    pw = pw.transpose(1, 2)  # [B,T,bins]
    return pw / (win.sum() ** 2)


class TorchFrontend:
    def __init__(self, weights_path: str | Path, device="cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.per_band = load_weights(weights_path)
        self.fc = torch.tensor(band_centers(), dtype=torch.float32, device=self.device)
        mats = {}
        for nfft in RESOLUTIONS:
            nb = nfft // 2 + 1
            W = np.zeros((nb, BANDS), dtype=np.float32)
            for k, (n, bins, w) in enumerate(self.per_band):
                if n == nfft:
                    W[bins, k] = w
            mats[nfft] = torch.from_numpy(W).to(self.device)
        self.mats = mats

    def band_power(self, x):
        """x [B,n,2] stereo -> dict of [B,T,192] channel powers."""
        torch = self.torch
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], 2)
        B, n, C = x.shape
        assert C == 2, "expected stereo input [B,n,2]"
        T = n // HOP
        Lf = x[..., 0].reshape(B, T * HOP)
        Rf = x[..., 1].reshape(B, T * HOP)
        M = 0.5 * (Lf + Rf)
        S = 0.5 * (Lf - Rf)
        chans = {"M": M, "S": S, "L": Lf, "R": Rf}
        out = {c: torch.zeros(B, T, BANDS, device=x.device, dtype=x.dtype) for c in chans}
        for nfft in RESOLUTIONS:
            mat = self.mats[nfft]
            for c, sig in chans.items():
                pw = torch_stft_power(sig, nfft)
                out[c] = out[c] + pw @ mat
        return out

    def features(self, x):
        p = self.band_power(x)
        dbm = 10.0 * self.torch.log10(p["M"] + 1e-12)
        dbs = 10.0 * self.torch.log10(p["S"] + 1e-12)
        dbl = 10.0 * self.torch.log10(p["L"] + 1e-12)
        dbr = 10.0 * self.torch.log10(p["R"] + 1e-12)
        feat = self.torch.stack([
            (dbm + 60.0) / 30.0,
            (dbs + 60.0) / 30.0,
            self.torch.clip((dbl - dbr) / 20.0, -1.0, 1.0),
        ], dim=2)
        return feat, p


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[2]
    meta = save_weights(root / "shared" / "frontend" / "v1")
    size = (root / "shared" / "frontend" / "v1" / "weights.bin").stat().st_size
    print("FE-v1 weights:", size, "bytes,", meta["bands"], "bands")
    print("fc[0..2]:", meta["center_hz"][:3], "fc[-1]:", meta["center_hz"][-1])
