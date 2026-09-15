"""ifm.py：Stage 1/2 形状、地形 token 注入效果、梯度、确定性。"""

import torch

from pgmt.cfg.assumptions import get
from pgmt.policy.ifm import IFM


def _inputs(B=2):
    s_hist = torch.randn(B, 256)
    c_future = torch.randn(B, get("A2").value.K, 61)
    return s_hist, c_future


def test_stage1_shape_and_kv_len():
    ifm = IFM()
    s_hist, c_future = _inputs()
    s_int, attn = ifm(s_hist, c_future, return_attn=True)
    assert s_int.shape == (2, 256)
    # KV = [s_hist] + 6 帧 = 7 token
    assert attn.shape == (2, get("A3").value.mhca_heads, 1, 7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 1), atol=1e-6)


def test_stage2_shape_and_kv_len():
    ifm = IFM()
    s_hist, c_future = _inputs()
    z = torch.randn(2, 4, 256)
    s_int, attn = ifm(s_hist, c_future, z, return_attn=True)
    assert s_int.shape == (2, 256)
    # KV = [s_hist] + 6 帧 + 4 地形 token = 11
    assert attn.shape == (2, 4, 1, 11)


def test_terrain_tokens_influence_output():
    ifm = IFM()
    s_hist, c_future = _inputs()
    z1 = torch.randn(2, 4, 256)
    z2 = torch.randn(2, 4, 256)
    out1 = ifm(s_hist, c_future, z1)
    out2 = ifm(s_hist, c_future, z2)
    assert not torch.allclose(out1, out2), "不同地形 token 应产生不同 s_int"


def test_terrain_tokens_influence_attention():
    """注入的地形 token 应获得注意力权重（被 policy 真正关注）。"""
    ifm = IFM()
    s_hist, c_future = _inputs()
    z = torch.randn(2, 4, 256)
    _, attn = ifm(s_hist, c_future, z, return_attn=True)
    terrain_attn = attn[..., 7:].sum()  # 后 4 列为地形 token
    assert terrain_attn.item() > 0


def test_gradient_flows_to_terrain_tokens():
    ifm = IFM()
    s_hist, c_future = _inputs()
    z = torch.randn(2, 4, 256, requires_grad=True)
    ifm(s_hist, c_future, z).sum().backward()
    assert z.grad is not None and z.grad.abs().sum().item() > 0


def test_deterministic():
    torch.manual_seed(0)
    ifm = IFM()
    s_hist, c_future = _inputs()
    z = torch.randn(2, 4, 256)
    out1 = ifm(s_hist, c_future, z)
    torch.manual_seed(0)  # 重置种子 → 第二个模型权重与第一个一致
    ifm2 = IFM()
    out2 = ifm2(s_hist, c_future, z)
    assert torch.allclose(out1, out2)
