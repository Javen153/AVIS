import numpy as np
import pytest

from avis_ml import frontend as fe
from avis_ml import synth as sy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "shared" / "frontend" / "v1" / "weights.bin"


@pytest.fixture(scope="session", autouse=True)
def weights():
    if not W.exists():
        fe.save_weights(W.parent)


def test_weight_layout_roundtrip():
    pb = fe.load_weights(W)
    assert len(pb) == fe.BANDS
    assert all(len(bins) == len(w) and len(bins) >= 2 for _, bins, w in pb)
    fc = fe.band_centers()
    assert fc[0] == pytest.approx(30.0, rel=1e-6)
    assert fc[-1] == pytest.approx(16000.0, rel=1e-6)
    assert np.all(np.diff(fc) > 0)


def test_sine_1k_level():
    fr = fe.NumpyFrontend(W)
    x = np.stack([sy.sine(1000, 1.0, 1.0)] * 2, axis=1)
    frames = fr.push(x)
    assert len(frames) == 100
    db = np.array([f["bandsDb_M"] for f in frames[50:]])
    peak = db.mean(axis=0)
    k = int(np.argmax(peak))
    assert -7.0 <= peak[k] <= -5.0, (k, peak[k])
    assert abs(fe.band_centers()[k] - 1000) / 1000 < 0.02


def test_noise_no_steps():
    fr = fe.NumpyFrontend(W)
    x = sy.white_noise(2.0, 0.1, seed=5)
    x = np.stack([x, x], axis=1)
    frames = fr.push(x)
    db = np.array([f["bandsDb_M"] for f in frames[20:]])
    mean_db = np.mean(db, axis=0)
    d = np.abs(np.diff(mean_db))
    # no step larger than 1.5 dB anywhere, in particular at resolution borders
    assert d.max() <= 1.5, d.max()
    res = fe.band_resolution(fe.band_centers())
    bd = np.where(np.diff(res) != 0)[0]
    assert d[bd].max() <= 1.5, d[bd]


def test_numpy_torch_parity():
    torch = pytest.importorskip("torch")
    tf = fe.TorchFrontend(W, device="cpu")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48000 * 2, 2)).astype(np.float32) * 0.1
    nf = fe.NumpyFrontend(W)
    nf_frames = nf.push(x)
    feat_np = np.stack([f["features"] for f in nf_frames])  # [T,3,192]
    xt = torch.from_numpy(x[None].copy())
    feat_t, p = tf.features(xt)
    feat_t = feat_t[0].numpy()
    assert feat_np.shape == feat_t.shape, (feat_np.shape, feat_t.shape)
    diff = np.abs(feat_np - feat_t)
    assert diff.max() <= 1e-4, diff.max()
    assert diff.mean() <= 1e-6
    for key in ("M", "S", "L", "R"):
        assert torch.isfinite(p[key]).all()
