import torch

from pgmt.cfg.assumptions import get
from pgmt.contracts import HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.train.policy import Stage1Policy, Stage2Policy
from pgmt.train.train_stage2 import MockStage2Env
from pgmt.train.ppo import PPO


def _obs(n=2):
    return {
        "obs": torch.zeros(n, OBS_DIM),
        "history": torch.zeros(n, HISTORY_LEN, OBS_DIM),
        "future": torch.zeros(n, get("A2").value.K, REF_FRAME_DIM),
        "privileged": torch.zeros(n, PRIV_DIM),
        "elevation": torch.zeros(n, get("A8").value.map_size, get("A8").value.map_size),
    }


def test_stage2_migrates_stage1_split_heads_and_uses_terrain():
    torch.manual_seed(0)
    stage1 = Stage1Policy()
    stage2 = Stage2Policy.from_stage1(stage1)
    assert stage2.head_names == ("upper", "lower", "terrain", "aux")
    old = stage1.critic(stage1_state := torch.zeros(2, PRIV_DIM), torch.zeros(2, 256))
    new = stage2.critic(stage1_state, torch.zeros(2, 256))
    assert torch.allclose(new[:, [0, 1, 3]], old, atol=1e-6)
    output = stage2.act(_obs())
    assert output.actions.shape == (2, 29)
    assert output.values.shape == (2, 4)


def test_stage2_cpu_rollout_update_four_heads():
    env = MockStage2Env(2, torch.device("cpu"), horizon=3)
    policy = Stage2Policy()
    cfg = type("Cfg", (), {
        "num_steps_per_env": 2, "num_learning_epochs": 1,
        "num_mini_batches": 1, "gamma": 0.99, "lam": 0.95,
        "learning_rate": 1e-3, "clip_param": 0.2, "entropy_coef": 0.01,
        "max_grad_norm": 1.0,
    })()
    ppo = PPO(policy, config=cfg, total_updates=1)
    obs, collected = ppo.collect_rollout(env, env.reset())
    metrics = ppo.update()
    assert collected["reward_mean"] == float(collected["reward_mean"])
    assert metrics["num_updates"] == 1
    assert "value_loss_terrain" in metrics
