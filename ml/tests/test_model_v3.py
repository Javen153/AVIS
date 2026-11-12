"""M-v3 model: harmonic prior, step/forward equivalence, pitch residual, ONNX parity.

CPU only, so the suite stays usable while a GPU training job owns the device.
ONNX tolerance is the acceptance number from docs/REQUIREMENTS_V3.md (<= 1e-3).
"""
import math
from pathlib import Path

import pytest
import torch

from avis_ml.model_v3 import (BANDS, EMB, F_MAX, F_MIN, AvisNetV3, HarmonicPrior,
                              StepWrapperV3, count_params)

PARAMS_LO = 1_800_000
PARAMS_HI = 2_100_000
V21_PARAMS = 2_990_000  # previous version, v3 must stay below it


def band_of(f: float) -> float:
    return (math.log(f) - math.log(F_MIN)) / (math.log(F_MAX / F_MIN)) * (BANDS - 1)


def harmonic_stack(f0: float = 220.0, level: float = 1.0, spread: int = 0,
                   harmonics: int = 6) -> torch.Tensor:
    """[1,3,192] FE-v1 features: -120 dB floor, `harmonics` peaks at `level`.

    `level` is the FE-v1 channel value: 1.0 -> -30 dB band power. `spread` is the
    half width of each peak in bands (0 = one band per harmonic).
    """
    x = torch.full((1, 3, BANDS), -2.0)
    for h in range(1, harmonics + 1):
        k = int(round(band_of(f0 * h)))
        x[0, 0, max(0, k - spread):min(BANDS, k + spread + 1)] = level
    return x


# ------------------------------------------------------------------ prior
def test_harmonic_prior_peaks_at_fundamental():
    p = HarmonicPrior()(harmonic_stack(220.0))[0]
    assert p.shape == (BANDS,)
    assert torch.isfinite(p).all()
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    assert float(p.max()) > 0.9, float(p.max())
    assert abs(int(p.argmax()) - round(band_of(220.0))) <= 2


def test_harmonic_prior_silence_is_finite():
    """A constant dB floor has no harmonic excess: prior must be (near) exactly 0."""
    p = HarmonicPrior()(torch.full((2, 3, BANDS), -2.0))
    assert p.shape == (2, BANDS)
    assert torch.isfinite(p).all()
    assert float(p.max()) <= 1e-3, float(p.max())
    assert float(p.min()) >= 0.0
    assert float(p.abs().sum()) <= 1e-2


def test_harmonic_prior_frame_scale_invariance():
    """+3 dB (x2 linear band power) must not change the prior shape.

    The ratio is level-free up to the +1e-6 epsilon. At -12 dB band power the
    epsilon is negligible (measured 3.3e-5); at the mixer's usual -30 dB it
    starts to bias the frame mean and the deviation grows to ~2e-3, still two
    orders of magnitude below the prior's [0,1] range.
    """
    prior = HarmonicPrior()
    for level, tol in ((1.6, 1e-4), (1.0, 5e-3)):
        base = harmonic_stack(220.0, level=level)
        louder = base.clone()
        louder[0, 0] += 0.1
        pa, pb = prior(base)[0], prior(louder)[0]
        assert float((pa - pb).abs().max()) <= tol, (level, float((pa - pb).abs().max()))
        assert int(pa.argmax()) == int(pb.argmax())


def test_harmonic_prior_noise_stays_low():
    """White noise must not produce a false harmonic peak (old formula gave 1.0).

    Measured on 1 s of white noise at -10 dBFS through FE-v1 (100 frames): raw
    prior max 0.32, frame-max median 0.17, p90 0.27; at -30 dBFS 0.30 / 0.12.
    Bounds are 0.45 / 0.25 to stay robust to the random draw while still
    failing the pre-fix behaviour (~1.0).
    """
    import numpy as np

    from avis_ml.frontend import TorchFrontend

    tf = TorchFrontend(Path(__file__).resolve().parents[2] / "shared" / "frontend"
                       / "v1" / "weights.bin", device="cpu")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000) * 10 ** (-10 / 20)
    feat, _ = tf.features(torch.from_numpy(np.stack([x, x], 1).astype(np.float32))[None])
    feat = feat.reshape(-1, 3, BANDS)
    p = HarmonicPrior()(feat)
    assert torch.isfinite(p).all()
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    assert float(p.max()) <= 0.45, float(p.max())
    assert float(p.max(dim=1).values.median()) <= 0.25


def test_harmonic_prior_missing_fundamental_is_octave_ambiguous():
    """Known limitation (docs/REQUIREMENTS_V3.md): without the fundamental the
    subharmonic sum peaks on the 2nd harmonic, not on the missing f0.

    Measured: removing the 220 Hz band leaves the peak at 0.47 on ~444 Hz.
    The prior is only a residual hint; the GRU sees the full spectrum.
    """
    x = harmonic_stack(220.0, harmonics=6)
    k0 = int(round(band_of(220.0)))
    x[0, 0, max(0, k0 - 1):k0 + 2] = -2.0  # fundamental removed, floor left
    p = HarmonicPrior()(x)[0]
    assert torch.isfinite(p).all()
    assert float(p.max()) > 0.3, float(p.max())
    top = int(p.argmax())
    assert top != k0  # cannot recover the missing fundamental
    assert abs(top - round(band_of(440.0))) <= 2, top


def test_harmonic_prior_is_batch_and_channel_independent():
    prior = HarmonicPrior()
    frames = [harmonic_stack(110.0, spread=1), harmonic_stack(220.0, level=0.8, spread=1),
              torch.full((1, 3, BANDS), -1.5)]
    p = prior(torch.cat(frames))
    for i, f in enumerate(frames):
        # conv1d over a batch is not bit-identical to a single row
        assert float((p[i] - prior(f)[0]).abs().max()) <= 1e-6
    # only the mid channel is used: side and L/R must not affect the prior
    noisy = frames[0].clone()
    noisy[0, 1:] = torch.randn(2, BANDS) * 5.0
    assert float((prior(noisy) - prior(frames[0])).abs().max()) <= 1e-6


# ------------------------------------------------------------------ network
def test_v3_param_count_bound():
    net = AvisNetV3()
    n = count_params(net)
    assert PARAMS_LO < n < PARAMS_HI, n
    assert n < V21_PARAMS, n
    # the prior kernel ships as a buffer, so it is exported and never trained
    assert not list(net.prior.parameters())
    assert "kernel" in net.prior.state_dict()


def test_v3_step_shapes():
    net = AvisNetV3().eval()
    feat = torch.randn(2, 3, BANDS)
    h = torch.randn(2, 2, net.hid)
    emb, act, pitch, sal, h2 = net.step(feat, h)
    assert emb.shape == (2, BANDS, EMB)
    assert act.shape == pitch.shape == sal.shape == (2, BANDS)
    assert h2.shape == (2, 2, net.hid)
    assert torch.isfinite(h2).all()
    norm = emb.norm(dim=-1)  # embedding is L2-normalised per band (runtime contract)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    for t in (act, pitch, sal):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_v3_step_equals_forward_with_nonzero_h0():
    torch.manual_seed(0)
    net = AvisNetV3().eval()
    T = 16
    x = torch.randn(1, T, 3, BANDS)
    h0 = 0.3 * torch.randn(2, 1, net.hid)  # non-zero state, not the usual zeros
    with torch.no_grad():
        fwd = net(x, h0)
        h = h0.clone()
        seq = {k: [] for k in ("emb", "act", "pitch", "sal")}
        for t in range(T):
            emb, act, pitch, sal, h = net.step(x[:, t], h)
            for k, v in zip(("emb", "act", "pitch", "sal"), (emb, act, pitch, sal)):
                seq[k].append(v[0])
        for k in seq:
            assert (torch.stack(seq[k], dim=0) - fwd[k][0]).abs().max() <= 1e-5, k
        assert (h - fwd["h"]).abs().max() <= 1e-5


def test_v3_pitch_logit_uses_prior_gain():
    torch.manual_seed(1)
    net = AvisNetV3().eval()
    feat = harmonic_stack(220.0)
    h = torch.zeros(2, 1, net.hid)
    with torch.no_grad():
        net.prior_gain.zero_()
        off = net.step(feat, h)
        net.prior_gain.fill_(1.0)
        on = net.step(feat, h)
    assert float((off[2] - on[2]).abs().max()) > 1e-2  # pitch changed
    # the residual is pitch-only: shared decoder and other heads are untouched
    for i, k in enumerate(("emb", "act", "pitch", "sal")):
        if k != "pitch":
            assert torch.equal(off[i], on[i]), k


def test_v3_gradients_reach_prior_gain_and_all_heads():
    torch.manual_seed(2)
    net = AvisNetV3()
    net.zero_grad()
    feat = torch.randn(2, 3, BANDS)
    h = torch.randn(2, 2, net.hid)
    emb, act, pitch, sal, _ = net.step(feat, h)
    (emb.sum() + act.sum() + pitch.sum() + sal.sum()).backward()
    targets = {
        "prior_gain": net.prior_gain,
        "head_emb": net.head_emb.weight,
        "head_act": net.head_act.weight,
        "head_pitch": net.head_pitch.weight,
        "head_sal": net.head_sal.weight,
        "adapt_pitch": net.adapt_pitch[0].weight,
        "shared_decoder": net.dec[0].weight,
    }
    for name, p in targets.items():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


# ------------------------------------------------------------------ ONNX
def test_v3_onnx_step_parity_20_frames(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(3)
    net = AvisNetV3().eval()
    path = tmp_path / "v3_step.onnx"
    torch.onnx.export(StepWrapperV3(net),
                      (torch.randn(1, 3, BANDS), torch.randn(2, 1, net.hid)), str(path),
                      input_names=["feat", "h_in"],
                      output_names=["emb", "act", "pitch", "sal", "h_out"],
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    T = 20
    x = torch.randn(1, T, 3, BANDS)
    h = 0.3 * torch.randn(2, 1, net.hid)
    worst = 0.0
    with torch.no_grad():
        for t in range(T):
            h_in = h
            ref = net.step(x[:, t], h_in)
            got = sess.run(None, {"feat": x[:, t].numpy(), "h_in": h_in.numpy()})
            assert got[4].shape == (2, 1, net.hid)
            for i, name in enumerate(("emb", "act", "pitch", "sal", "h_out")):
                err = float(abs(ref[i].numpy() - got[i]).max())
                assert math.isfinite(err), name
                assert err <= 1e-3, (name, t, err)
                worst = max(worst, err)
            h = ref[4]
    assert worst <= 1e-3, worst


# ------------------------------------------------------------------ wiring
def test_train_build_model_selects_v3():
    """Integration: train.py arch switch builds the v3 net and its counter."""
    from avis_ml.train import build_model

    net, hid, counter = build_model({"arch": "v3", "hid": 208})
    assert isinstance(net, AvisNetV3)
    assert hid == 208 and net.hid == 208
    assert counter(net) == count_params(net)
