#!/usr/bin/env python
"""M0 冒烟测试 —— 在训练服务器上运行，验证 Isaac Gym × Blackwell 全链路。

三阶段（任一失败即退出，退出码 = 失败阶段编号，0 = 全过）:
  Phase 1: torch + CUDA + sm_120（Blackwell）张量运算
  Phase 2: Isaac Gym 物理仿真（128 env 自由落体箱，100 步）
  Phase 3: rsl-rl PPO 训练闭环（128 env 最小任务，10 iter）

用法:
  python setup/smoke_test.py [--num-envs 128] [--steps 100] [--iters 10]

注意: 本文件针对 Isaac Gym（社区 PP4 适配版）路径编写；
      Isaac Lab 路径用 smoke_test_isaaclab.py。
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
    if torch.version.cuda != "12.8":
        print(f"[WARN] torch 为 cu{torch.version.cuda}，期望 cu128（Blackwell 需要）")

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


def phase2(num_envs: int, steps: int) -> bool:
    print("\n" + "=" * 60)
    print("Phase 2: Isaac Gym 物理仿真")
    print("=" * 60)
    try:
        import isaacgym  # noqa: F401
        from isaacgym import gymapi, gymtorch
    except Exception as e:
        print(f"[FAIL] import isaacgym 失败: {e}")
        print("      诊断: 官方 PP4 绑定仅支持 Python<=3.9 且按旧 torch ABI 编译；")
        print("      请确认: conda 环境 python 版本与 torch 2.1.2 cu121（勿用 cu128）。")
        print("      处置: 重跑 setup/check_isaacgym.sh 核对绑定与内核，")
        print("      仍不行则转 Isaac Lab（setup/install_server.sh）。")
        return False

    import torch

    try:
        gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = 0.02
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.num_threads = 2

        sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)  # 无渲染头

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        plane.restitution = 0.0
        gym.add_ground(sim, plane)

        box = gym.create_box(sim, 0.5, 0.5, 0.5, gymapi.AssetOptions())

        spacing = 2.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        per_row = int(num_envs ** 0.5)
        for i in range(num_envs):
            env = gym.create_env(sim, lower, upper, per_row)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # 1 m 高处自由落体
            gym.create_actor(env, box, pose, f"box_{i}", 1, 1)

        gym.prepare_sim(sim)
        rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        assert rb.shape == (num_envs, 13), f"状态张量形状异常: {rb.shape}"

        for _ in range(steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_rigid_body_state_tensor(sim)
        gym.sync_frame_time(sim)

        z = rb[:, 2].clone()
        assert torch.isfinite(z).all(), "状态含 NaN/Inf"
        # 从 1.0 m 落下，应停在 z≈0.25（半高 0.25）
        assert (z > 0.2).all() and (z < 0.6).all(), \
            f"箱体高度异常: z ∈ [{z.min():.3f}, {z.max():.3f}]（预期 ~0.25）"
        print(f"[i] {steps} 步物理仿真 OK，箱体静止高度 z ∈ [{z.min():.3f}, {z.max():.3f}]")

        gym.destroy_sim(sim)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 物理仿真失败: {e}")
        print("      若为 CUDA/PhysX 初始化错误（no kernel image 等），多为 PhysX 缺本机 GPU")
        print("      架构内核 —— 用 setup/check_isaacgym.sh 判定；不可行则转 Isaac Lab。")
        return False

    print("[PASS] Phase 2")
    return True


def phase3(num_envs: int, iters: int) -> bool:
    print("\n" + "=" * 60)
    print("Phase 3: rsl-rl PPO 训练闭环")
    print("=" * 60)
    try:
        import torch
        import isaacgym  # noqa: F401
        from isaacgym import gymapi, gymtorch
        from rsl_rl.env import VecEnv
        from rsl_rl.runners import OnPolicyRunner
    except Exception as e:
        print(f"[FAIL] import 失败: {e}")
        print("      确认已安装: pip install rsl-rl-lib==2.1.2")
        return False

    class SmokeVecEnv(VecEnv):
        """最小任务：自由箱体受控，追踪随机目标点。观测 9 维，动作 3 维（力）。

        按 rsl_rl 2.3.1 的 VecEnv 约定:
          - get_observations()/reset() 返回 (obs_tensor, extras)，
            特权观测经 extras["observations"]["critic"] 传递
          - step() 返回 (obs, rewards, dones, infos)
        """

        def __init__(self, num_envs: int, device: str = "cuda:0"):
            self.num_envs = num_envs
            self.num_actions = 3
            self.device = device
            self.cfg = {}
            self.max_episode_length = 10**9
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)

            self.gym = gymapi.acquire_gym()
            sim_params = gymapi.SimParams()
            sim_params.dt = 0.02
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
            sim_params.use_gpu_pipeline = True
            sim_params.physx.use_gpu = True
            sim_params.physx.num_subscenes = 0
            self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0, 0, 1)
            plane.static_friction = 1.0
            plane.dynamic_friction = 1.0
            plane.restitution = 0.0
            self.gym.add_ground(self.sim, plane)

            box = self.gym.create_box(self.sim, 0.5, 0.5, 0.5, gymapi.AssetOptions())
            spacing = 2.0
            lower = gymapi.Vec3(-spacing, 0.0, -spacing)
            upper = gymapi.Vec3(spacing, spacing, spacing)
            self.envs, self.actors = [], []
            for i in range(num_envs):
                env = self.gym.create_env(self.sim, lower, upper, int(num_envs ** 0.5))
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0.0, 0.0, 0.25)
                a = self.gym.create_actor(env, box, pose, f"box_{i}", 1, 1)
                self.envs.append(env)
                self.actors.append(a)
            self.gym.prepare_sim(self.sim)

            self.rb = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
            self.targets = torch.zeros(num_envs, 3, device=device)
            self.obs_buf = torch.zeros(num_envs, 9, device=device)
            self._reset_targets()

        def _reset_targets(self):
            self.targets[:, 0] = torch.rand(self.num_envs, device=self.device) * 2 - 1
            self.targets[:, 1] = torch.rand(self.num_envs, device=self.device) * 2 - 1
            self.targets[:, 2] = 0.25

        def _refresh_obs(self):
            self.obs_buf[:, 0:3] = self.rb[:, 0:3]
            self.obs_buf[:, 3:6] = self.rb[:, 7:10]  # 线速度
            self.obs_buf[:, 6:9] = self.targets

        # ---- rsl-rl 2.3.1 VecEnv 接口 ----
        def get_observations(self):
            self._refresh_obs()
            extras = {"observations": {"critic": self.obs_buf.clone()}}
            return self.obs_buf.clone(), extras

        def step(self, actions):
            for i in range(self.num_envs):
                f = gymapi.Vec3(actions[i, 0].item(), actions[i, 1].item(), actions[i, 2].item())
                self.gym.apply_rigid_body_force(
                    self.envs[i], self.actors[i], f, gymapi.Vec3(0, 0, 0), gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._refresh_obs()

            dist = (self.obs_buf[:, 0:3] - self.targets).norm(dim=1)
            rewards = -dist
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            infos = {"observations": {"critic": self.obs_buf.clone()}}
            return self.obs_buf.clone(), rewards, dones, infos

        def reset(self):
            self._reset_targets()
            self._refresh_obs()
            extras = {"observations": {"critic": self.obs_buf.clone()}}
            return self.obs_buf.clone(), extras

    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "num_steps_per_env": 24,
        "save_interval": 10**9,
        "empirical_normalization": False,
        "logger": "tensorboard",
    }

    try:
        env = SmokeVecEnv(num_envs)
        runner = OnPolicyRunner(env, train_cfg, log_dir="runs/smoke", device="cuda:0")
        runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] PPO 训练失败: {e}")
        return False

    print("[PASS] Phase 3")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    results = []
    results.append(("Phase 1: torch/CUDA/Blackwell", phase1()))
    results.append(("Phase 2: Isaac Gym 物理", phase2(args.num_envs, args.steps)))
    results.append(("Phase 3: rsl-rl PPO", phase3(args.num_envs, args.iters)))

    print("\n" + "=" * 60)
    print("冒烟测试汇总")
    print("=" * 60)
    failed = 0
    for i, (name, ok) in enumerate(results, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok and failed == 0:
            failed = i
    if failed == 0:
        print("\n全部通过 —— 可在本机开始 M2 训练。")
    else:
        print(f"\n阶段 {failed} 失败 —— 按 README「服务器安装与冒烟」决策树处置。")
    sys.exit(failed)


if __name__ == "__main__":
    main()
