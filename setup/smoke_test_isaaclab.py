#!/usr/bin/env python
"""M0 冒烟测试 —— Isaac Lab 路径（兜底，服务器运行）。

两阶段:
  Phase 1: torch + CUDA 张量运算
  Phase 2: Isaac Lab 物理仿真（headless 启动 + 自由落体箱 + 100 步）

说明:
  - 完整 PPO 训练闭环在 M2 随真实环境一起验证（isaaclab_rl + manager-based env）。
  - 本文件按 Isaac Lab 2.2/2.3 的公开 API 编写，首次在服务器运行时若报
    API 差异（Isaac Lab 迭代快），按报错微调即可，这本身就是 M0 的验证目的。
  - 仅当 PP4 路径（setup/install_server_isaacgym.sh + smoke_test.py）失败时才走这里。

用法:
  python setup/smoke_test_isaaclab.py [--steps 100]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def phase1(device: str = "cuda:0") -> bool:
    print("\n" + "=" * 60)
    print("Phase 1: torch + CUDA")
    print("=" * 60)
    import torch

    print(f"[i] torch {torch.__version__}, cuda build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] torch 看不到 CUDA 设备")
        return False
    print(f"[i] 设备: {torch.cuda.get_device_name(0)}, 数量: {torch.cuda.device_count()}")

    cap = torch.cuda.get_device_capability(0)
    print(f"[i] 计算能力 sm_{cap[0]}{cap[1]}")
    # 本路径（Isaac Lab + cu128）同样可跑在 Ada sm_89 上，只做信息提示
    if (cap[0], cap[1]) != (8, 9):
        print(f"[WARN] 非目标架构（期望 sm_89 / RTX 5880 Ada），当前 sm_{cap[0]}{cap[1]}；继续")

    try:
        x = torch.randn(2048, 2048, device=device)
        y = (x @ x).sum().item()
        assert x.isfinite().all()
        print(f"[i] GPU 矩阵乘 OK（sum={y:.2e}）")
    except Exception as e:
        print(f"[FAIL] GPU 运算失败: {e}")
        return False

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip().splitlines()
        print(f"[i] 驱动: {out[0] if out else '(nvidia-smi 无输出)'}")
    except FileNotFoundError:
        print("[WARN] 未找到 nvidia-smi（不影响 torch 判据）")

    print("[PASS] Phase 1")
    return True


def phase2(steps: int, device: str = "cuda:0") -> bool:
    print("\n" + "=" * 60)
    print("Phase 2: Isaac Lab 物理仿真")
    print("=" * 60)

    # AppLauncher 必须在导入其他 isaaclab 模块之前初始化
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        # Keep Kit/Vulkan and PhysX on the same explicitly selected GPU.  The
        # multi-GPU renderer otherwise probes every visible card on a shared
        # server; CUDA_VISIBLE_DEVICES alone does not reliably constrain Kit.
        app_launcher = AppLauncher(
            headless=True, device=device, multi_gpu=False,
            kit_args="--/renderer/multiGpu/enabled=False --/renderer/multiGpu/autoEnable=False",
        )
        simulation_app = app_launcher.app
        print("[i] Isaac Sim 启动成功")
    except Exception as e:
        print(f"[FAIL] Isaac Sim 启动失败: {e}")
        print("      检查: isaaclab 安装完整性、驱动版本、--headless 环境（DISPLAY 变量）。")
        return False

    sim = None
    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim import SimulationContext

        # Isaac Lab defaults to ``/tmp/isaaclab/logs``.  That directory can
        # be left owned by another user on shared servers, so keep smoke-test
        # logs in this checkout unless the caller explicitly overrides it.
        log_dir = os.environ.get(
            "PGMT_ISAACLAB_LOG_DIR",
            os.path.join(os.getcwd(), "data", "processed", "isaaclab_logs"),
        )
        os.makedirs(log_dir, exist_ok=True)
        sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=device, log_dir=log_dir)
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
                collision_props=sim_utils.CollisionPropertiesCfg(),
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
        # Kit's immediate shutdown may terminate the interpreter before code
        # after the cleanup block is reached, so emit the decisive success
        # marker while the simulation is still alive.
        print("[PASS] Phase 2")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Isaac Lab 物理仿真失败: {e}")
        return False
    finally:
        # 无论成败都要释放：失败路径不关会留下 Isaac Sim 进程占住 GPU
        try:
            if sim is not None:
                sim.close()
        except Exception:
            pass
        try:
            if simulation_app is not None:
                # Isaac Sim 5.1 can block in the full stage-cleanup path when
                # another Kit process owns the shared KVDB lock.  This smoke
                # test has no replicator output to preserve, so use the
                # immediate shutdown path and retain compatibility with the
                # lightweight fake app used by the unit tests.
                try:
                    simulation_app.close(skip_cleanup=True)
                except TypeError:
                    simulation_app.close()
        except Exception:
            pass

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default=os.environ.get("PGMT_CUDA_DEVICE", "cuda:0"))
    args = p.parse_args()

    # 首个失败即停止（Phase 2 依赖 Phase 1 的 torch/CUDA 可用性）
    phases = [
        ("Phase 1: torch/CUDA", lambda: phase1(args.device)),
        ("Phase 2: Isaac Lab 物理", lambda: phase2(args.steps, args.device)),
    ]
    failed = 0
    for i, (name, fn) in enumerate(phases, 1):
        if fn():
            continue
        failed = i
        break

    print("\n" + "=" * 60)
    print("冒烟测试汇总（Isaac Lab 路径）")
    print("=" * 60)
    for i, (name, _) in enumerate(phases, 1):
        if failed == 0 or i < failed:
            status = "PASS"
        elif i == failed:
            status = "FAIL"
        else:
            status = "SKIP（前序阶段失败）"
        print(f"  [{status}] {name}")

    if failed == 0:
        print("\n全部通过 —— Isaac Lab 路径可用，M2 按 Isaac Lab API 编写环境。")
    else:
        print(f"\n阶段 {failed} 失败 —— 按报错信息处置或反馈给开发侧。")
    sys.exit(failed)


if __name__ == "__main__":
    main()
