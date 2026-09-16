"""data/viz_retarget.py：冒烟 + 坐标系回归。

回归防护：源骨架曾直接按 rig 原始坐标（上=+x）绘制，骨架横躺且
超出轴限制——用户目检时"source 没有姿态"。修复为 W 变换到 G1 系
（上=+z）并骨盆对齐原点。
"""

import os

import numpy as np
import pytest

from data.bvh import load_bvh
from data.retarget_lafan1 import W, retarget
from data.viz_retarget import plot_frames

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lafan1")
HAS_DATA = os.path.isdir(DATA_DIR) and any(f.endswith(".bvh") for f in os.listdir(DATA_DIR))

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="需要 data/raw/lafan1 真实数据")


def test_source_frame_is_g1_world(tmp_path):
    """源骨架在 G1 系中应直立：头顶 z > 髋 z，且足在髋下方 ~0.8m。"""
    bvh = load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh"))
    d = retarget(bvh)
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * float(d["scale"])
    f = 200
    root = gpos[f, bvh.joint_index("Hips")]
    head = gpos[f, bvh.joint_index("Head")] - root
    foot = gpos[f, bvh.joint_index("LeftFoot")] - root
    assert head[2] > 0.5, f"头顶应在髋上方 >0.5m: {head}"
    assert abs(head[0]) < 0.2 and abs(head[1]) < 0.2, f"头顶应基本在髋正上方: {head}"
    assert -0.9 < foot[2] < -0.5, f"足应在髋下方 0.5~0.9m: {foot}"


def test_plot_frames_produces_png(tmp_path):
    out = str(tmp_path / "viz.png")
    plot_frames(os.path.join(DATA_DIR, "walk1_subject1.bvh"), out,
                [100, 200, 300, 400], "smoke")
    assert os.path.getsize(out) > 10_000, "viz PNG 应正常生成"
