#!/usr/bin/env python
"""M0 冒烟测试 —— Isaac Lab 路径（服务器运行）。

两阶段:
  Phase 1: torch + CUDA + sm_120（Blackwell）张量运算
  Phase 2: Isaac Lab 物理仿真（headless 启动 + 自由落体箱 + 100 步）

说明:
  - 完整 PPO 训练闭环在 M2 随真实环境一起验证（isaaclab_rl + manager-based env）。
  - 本文件按 Isaac Lab 2.2/2.3 的公开 API 编写，首次在服务器运行时若报
    API 差异（Isaac Lab 迭代快），按报错微调即可，这本身就是 M0 的验证目的。

用法:
  python setup/smoke_test_isaaclab.py [--steps 100]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def phase1() -> bool:
    print("\n" + "=" * 60)
    print("Phase 1: torch + CUDA + Blackwell")
    print("=" * 60)
    import torch

    print(f"[i] torch {torch.__version__}, cuda build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] torch 看不到 CUDA 设备")
        return False
    print(f"[i] 设备: {torch.cuda.get_device_name(0)}, 数量: {torch.cuda.device_count()}")

    cap = torch.cuda.get_device_capability(0)
    print(f"[i] 计算能力 sm_{cap[0]}{cap[1]}")
    if cap[0] < 12:
        print(f"[WARN] 目标为 Blackwell sm_120，当前 sm_{cap[0]}{cap[1]}；继续但不代表目标硬件")

    try:
        x = torch.randn(2048, 2048, device="cuda:0")
        y = (x @ x).sum().item()
        assert x.isfinite().all()
        print(f"[i] GPU 矩阵乘 OK（sum={y:.2e}）")
    except Exception as e:
        print(f"[FAIL] GPU 运算失败: {e}")
        return False

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    print(f"[i] 驱动: {driver}")
    print("[PASS] Phase 1")
    return True


def phase2(steps: int) -> bool:
    print("\n" + "=" * 60)
    print("Phase 2: Isaac Lab 物理仿真")
    print("=" * 60)

    # AppLauncher 必须在导入其他 isaaclab 模块之前初始化
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app
        print("[i] Isaac Sim 启动成功")
    except Exception as e:
        print(f"[FAIL] Isaac Sim 启动失败: {e}")
        print("      检查: isaaclab 安装完整性、驱动版本、--headless 环境（DISPLAY 变量）。")
        return False

    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim import SimulationContext

        sim_cfg = sim_utils.SimulationCfg(dt=0.02, device="cuda:0")
        sim = SimulationContext(sim_cfg)

        # 地面
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())

        # 自由落体箱（0.5m 立方体，从 1.0m 高处落下）
        box_cfg = RigidObjectCfg(
            prim_path="/World/box",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
        box = RigidObject(box_cfg)

        sim.reset()
        print(f"[i] 场景就绪，步进 {steps} 步 ...")
        for _ in range(steps):
            box.write_data_to_sim()
            sim.step()
            box.update(dt=sim_cfg.dt)

        z = box.data.root_pos_w[:, 2].clone()
        assert torch.isfinite(z).all(), "状态含 NaN/Inf"
        # 从 1.0 m 落下，应停在 z≈0.25（半高 0.25）
        assert (z > 0.2).all() and (z < 0.6).all(), \
            f"箱体高度异常: z ∈ [{z.min():.3f}, {z.max():.3f}]（预期 ~0.25）"
        print(f"[i] 箱体静止高度 z ∈ [{z.min():.3f}, {z.max():.3f}]，物理仿真 OK")

        sim.close()
        simulation_app.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Isaac Lab 物理仿真失败: {e}")
        return False

    print("[PASS] Phase 2")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    results = [
        ("Phase 1: torch/CUDA/Blackwell", phase1()),
        ("Phase 2: Isaac Lab 物理", phase2(args.steps)),
    ]

    print("\n" + "=" * 60)
    print("冒烟测试汇总（Isaac Lab 路径）")
    print("=" * 60)
    failed = 0
    for i, (name, ok) in enumerate(results, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok and failed == 0:
            failed = i
    if failed == 0:
        print("\n全部通过 —— Isaac Lab × Blackwell 可用，M2 按 Isaac Lab API 编写环境。")
    else:
        print(f"\n阶段 {failed} 失败 —— 按报错信息处置或反馈给开发侧。")
    sys.exit(failed)


if __name__ == "__main__":
    main()
