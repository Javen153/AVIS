import torch
import pytest

from avis_ml import losses
from avis_ml.model import AvisNet, StepWrapper, count_params


def test_param_count_and_step_shape():
    net = AvisNet()
    n = count_params(net)
    assert 3_500_000 < n < 5_500_000, n
    feat = torch.randn(1, 3, 192)
    h = torch.zeros(2, 1, 384)
    emb, act, pitch, sal, h2 = net.step(feat, h)
    assert emb.shape == (1, 192, 20)
    assert act.shape == (1, 192)
    assert h2.shape == (2, 1, 384)


def test_step_equals_forward():
    torch.manual_seed(0)
    net = AvisNet().eval()
    T = 24
    x = torch.randn(1, T, 3, 192)
    with torch.no_grad():
        fwd = net(x)
        h = torch.zeros(2, 1, 384)
        embs, acts, pitches, sals = [], [], [], []
        for t in range(T):
            emb, act, pitch, sal, h = net.step(x[:, t], h)
            embs.append(emb)
            acts.append(act)
            pitches.append(pitch)
            sals.append(sal)
        emb_s = torch.cat(embs, dim=1)  # [1,T,192,20]? step returns [1,192,20]
    # step emb is [1,192,20]; concatenate along T axis
    emb_s = torch.stack([e[0] for e in embs], dim=0)  # [T,192,20]
    act_s = torch.stack([a[0] for a in acts], dim=0)
    pitch_s = torch.stack([p[0] for p in pitches], dim=0)
    sal_s = torch.stack([s[0] for s in sals], dim=0)
    assert (emb_s - fwd["emb"][0]).abs().max() <= 1e-5
    assert (act_s - fwd["act"][0]).abs().max() <= 1e-5
    assert (pitch_s - fwd["pitch"][0]).abs().max() <= 1e-5
    assert (sal_s - fwd["sal"][0]).abs().max() <= 1e-5


def test_deep_clustering_zero_at_ideal():
    torch.manual_seed(1)
    B, T, F, D = 1, 4, 16, 4
    K = 3
    dom = torch.randint(0, K, (B, T, F))
    active = torch.ones(B, T, F)
    Y = torch.zeros(B * T * F, K)
    Y.scatter_(1, dom.reshape(-1)[:, None], 1.0)
    emb = torch.zeros(B * T * F, D)
    emb[:, :K] = Y[:, :K]
    # need orthonormal columns and D>=K; D=4,K=3 fine
    l = losses.deep_clustering(emb.reshape(B, T, F, D), dom, active)
    assert l.item() < 1e-6, l.item()


def test_deep_clustering_is_scale_invariant_and_finite():
    """The loss is scale-invariant, but the raw reduction overflows under AMP.

    Band weights are linear power (up to ~1e-3) and the old form squared
    (V^T Vw) before dividing by sum(w)^2, which reached inf in fp16 and
    produced non-finite gradients (D-040). Dividing first keeps it bounded.
    """
    torch.manual_seed(2)
    B, T, F, D = 1, 4, 32, 4
    dom = torch.randint(0, 3, (B, T, F))
    active = torch.ones(B, T, F)
    emb = torch.nn.functional.normalize(torch.randn(B, T, F, D), dim=-1)
    w = torch.rand(B, T, F) * 1e-3
    l1 = losses.deep_clustering(emb, dom, active, weights=w)
    l2 = losses.deep_clustering(emb, dom, active, weights=w * 1e6)
    assert torch.isfinite(l1) and torch.isfinite(l2)
    assert abs(float(l1) - float(l2)) <= 1e-4 * max(1.0, abs(float(l1)))
    lp = losses.deep_clustering(emb.half(), dom, active, weights=w.half())
    assert torch.isfinite(lp), lp


def test_solo_coherence_is_zero_for_one_cluster():
    """The solo term must vanish when all active bands share one direction."""
    D = 4
    B, T, F = 2, 3, 8
    one = torch.nn.functional.normalize(torch.randn(B, 1, 1, D), dim=-1)
    v = one.expand(B, T, F, D).contiguous()
    active = torch.ones(B, T, F)
    solo = torch.tensor([1.0, 0.0])
    assert float(losses.solo_coherence(v, active, solo)) < 1e-6
    mixed = torch.nn.functional.normalize(torch.randn(B, T, F, D), dim=-1)
    l = float(losses.solo_coherence(mixed, active, solo))
    assert 0.2 < l < 1.2, l
