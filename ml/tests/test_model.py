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
        embs, acts = [], []
        for t in range(T):
            emb, act, pitch, sal, h = net.step(x[:, t], h)
            embs.append(emb)
            acts.append(act)
        emb_s = torch.cat(embs, dim=1)  # [1,T,192,20]? step returns [1,192,20]
    # step emb is [1,192,20]; concatenate along T axis
    emb_s = torch.stack([e[0] for e in embs], dim=0)  # [T,192,20]
    act_s = torch.stack([a[0] for a in acts], dim=0)
    assert (emb_s - fwd["emb"][0]).abs().max() <= 1e-5
    assert (act_s - fwd["act"][0]).abs().max() <= 1e-5


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
