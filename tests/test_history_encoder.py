"""history_encoder.py：形状、输入敏感性、确定性。"""

import torch

from pgmt.policy.history_encoder import HistoryEncoder


def test_output_shape_and_dim():
    enc = HistoryEncoder()
    o = torch.randn(4, 96)
    H = torch.randn(4, 10, 96)
    s = enc(o, H)
    assert s.shape == (4, 256)


def test_history_matters():
    enc = HistoryEncoder()
    o = torch.randn(2, 96)
    H1 = torch.randn(2, 10, 96)
    H2 = torch.randn(2, 10, 96)
    s1, s2 = enc(o, H1), enc(o, H2)
    assert not torch.allclose(s1, s2), "不同历史应产生不同 s_hist"


def test_current_obs_matters():
    enc = HistoryEncoder()
    H = torch.randn(2, 10, 96)
    s1, s2 = enc(torch.randn(2, 96), H), enc(torch.randn(2, 96), H)
    assert not torch.allclose(s1, s2), "不同当前观测应产生不同 s_hist"


def test_deterministic():
    torch.manual_seed(0)
    enc = HistoryEncoder()
    o, H = torch.randn(2, 96), torch.randn(2, 10, 96)
    out1 = enc(o, H)
    torch.manual_seed(0)
    enc2 = HistoryEncoder()
    out2 = enc2(o, H)
    assert torch.allclose(out1, out2)


def test_zero_history_well_defined():
    """全零历史（冷启动）输出应有限。"""
    enc = HistoryEncoder()
    o = torch.randn(2, 96)
    s = enc(o, torch.zeros(2, 10, 96))
    assert torch.isfinite(s).all()
