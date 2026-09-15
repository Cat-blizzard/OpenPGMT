"""共享 MLP 构造器（actor / critic / glimpse 编码器共用）。

论文未指定隐层激活函数；统一取 ELU（rsl-rl / legged 系惯例，
见 assumptions.A3.activation），末层不带激活。
"""

from __future__ import annotations

import torch.nn as nn

from pgmt.cfg.assumptions import get

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


class MLP(nn.Module):
    """输入 → dims[0] → ... → dims[-1] 的全连接网络，隐层用 ELU。

    Args:
        dims: 各层宽度（含输入与输出，len>=2）。
        activation: 隐层激活函数名（见 _ACTIVATIONS）。
    """

    def __init__(self, dims, activation=None):
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"MLP 至少需要输入/输出两层，得到 {dims}")
        act = get("A3").value.activation if activation is None else activation
        if act not in _ACTIVATIONS:
            raise ValueError(f"未知激活函数: {act}（可选 {sorted(_ACTIVATIONS)}）")
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_ACTIVATIONS[act]())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
