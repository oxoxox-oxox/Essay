"""SB3 PPO 独立评估脚本（与 src/train_ppo.py 分离）。

评估环境：configs/train.yaml 的 eval_world（--world 可覆盖），
确定性动作（model.predict(deterministic=True)），
输出成功率/碰撞率/超时率/平均步数/平均奖励。

用法:
    python src/eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip        # 无头 20 集
    python src/eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip --episodes 3 --display   # 可视化
    python src/eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip \
        --world configs/world/robot_world.yaml                                          # 换评估世界
"""

from __future__ import annotations

import argparse
import os
import sys

# headless 服务器（Linux 且无 DISPLAY）强制 matplotlib Agg 后端；桌面/Windows 保留 GUI 供 --display 可视化
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.policy import GlobalCriticActorCriticPolicy  # noqa: E402,F401  # 让 PPO.load 能反序列化自定义 policy_class
from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402


def final_eval(model: PPO, cfg: dict, episodes: int, seed: int,
               display: bool, max_steps: int, chunk_size: int = 1) -> dict:
    """在 eval_world 上跑 deterministic 评估，输出成功率/碰撞率等指标。"""
    irsim_env = IrSimEnv(
        resolve_path(cfg["env"]["eval_world"]),
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg=cfg["env"],
        seed=seed + 2000,
        display=display,
        log_level="CRITICAL",
    )
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    steps_list, reward_list = [], []
    try:
        for ep in range(episodes):
            obs = irsim_env.reset(random=True)
            ep_reward, steps = 0.0, 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = np.clip(action, -1.0, 1.0)
                if chunk_size > 1:
                    obs, reward, done, info = irsim_env.step_chunk(action.reshape(chunk_size, -1))
                else:
                    obs, reward, done, info = irsim_env.step_single(action)
                ep_reward += reward
                steps += 1
                if done or steps >= max_steps:
                    break
            outcome = ("success" if info.get("arrive") else
                       "collision" if info.get("collision") else "timeout")
            outcomes[outcome] += 1
            steps_list.append(steps)
            reward_list.append(ep_reward)
            print(f"  ep{ep + 1:>2d}  {outcome:<8s} steps={steps:>3d} reward={ep_reward:>8.1f}")
    finally:
        irsim_env.close()

    n = max(episodes, 1)
    return {
        "success_rate": outcomes["success"] / n,
        "collision_rate": outcomes["collision"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "avg_steps": float(np.mean(steps_list)),
        "avg_reward": float(np.mean(reward_list)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SB3 PPO 模型评估（训练请用 src/train_ppo.py）"
    )
    ap.add_argument("--checkpoint", required=True,
                    help="SB3 zip 模型路径，如 checkpoints/ppo_mw_N1/best_model.zip")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--episodes", type=int, default=20, help="评估 episode 数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--display", action="store_true", help="可视化（弹 matplotlib 窗口）")
    ap.add_argument("--world", type=str, default=None,
                    help="评估世界 YAML（覆盖 eval_world；默认 configs/train.yaml 的 eval_world）")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    if args.world:
        cfg = deep_update(cfg, {"env": {"eval_world": args.world}})

    model = PPO.load(args.checkpoint, device=args.device)
    # 从模型动作空间推断 chunk 长度（单步=2 维，chunk N 维 = N*2）
    chunk_size = int(model.action_space.shape[0]) // 2
    cfg = deep_update(cfg, {"obs": {"prev_chunk_size": chunk_size},
                            "chunk": {"size": chunk_size}})
    # 按模型的 obs 空间自动同步 include_global（方向2 checkpoint 的 obs 比局部观测多 global_dim 维）
    obs_cfg = cfg["obs"]
    local_base = (
        int(obs_cfg.get("max_bins", 100))
        + (int(obs_cfg.get("goal_dim", 2)) if obs_cfg.get("include_goal", True) else 0)
        + (int(obs_cfg.get("action_dim", 2)) * chunk_size if obs_cfg.get("include_prev_action", False) else 0)
    )
    model_obs_dim = int(model.observation_space.shape[0])
    cfg = deep_update(cfg, {"obs": {"include_global": bool(model_obs_dim > local_base)}})
    print(f"[eval] loaded {args.checkpoint} (device={args.device}, chunk={chunk_size}, "
          f"obs={model_obs_dim}, include_global={cfg['obs']['include_global']})")
    metrics = final_eval(model, cfg, args.episodes, args.seed, args.display,
                         int(cfg["eval"].get("max_steps", 500)), chunk_size)
    print(f"[eval] success={metrics['success_rate']:.2%} "
          f"collision={metrics['collision_rate']:.2%} "
          f"timeout={metrics['timeout_rate']:.2%} "
          f"avg_steps={metrics['avg_steps']:.1f} avg_reward={metrics['avg_reward']:.1f}")


if __name__ == "__main__":
    main()
