"""策略评估：在 ir-sim 环境跑 episode，统计成功率/碰撞率/轨迹误差等。

用法:
    python eval/evaluate.py --checkpoint checkpoints/mlptd3_N1/model.pt \
        [--world configs/world/robot_world.yaml] [--episodes 20] [--save runs/eval.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402
from utils.metrics import compute_episode_metrics, summarize_metrics  # noqa: E402

from eval.policy import env_obs_cfg_from_meta, load_fp32_actor  # noqa: E402


def run_episode(
    env: IrSimEnv,
    policy,
    chunk_size: int = 1,
    gamma: float = 0.99,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """跑单个 episode。

    Args:
        env: IrSimEnv 实例。
        policy: 可调用对象，输入 obs (numpy (obs_dim,))，输出归一化动作
            chunk (numpy, (chunk_size, action_dim))，范围 [-1,1]。
        chunk_size: action chunk 长度 N。
        gamma: chunk 内折现（仅影响记录的总回报）。
        max_steps: 决策步上限（覆盖 env 内 max_steps 用于评估）。
    """
    obs = env.reset(random=True)
    max_steps = max_steps or env.max_steps

    positions = [np.squeeze(np.asarray(env._env.get_robot_state(), dtype=np.float64))[:2]]
    dist_list = [float(env._dist_to_goal())]
    total_reward = 0.0
    arrive = collision = False
    step_count = 0
    discount = 1.0

    for _ in range(int(max_steps)):
        with torch.no_grad():
            action = policy(obs)
        action = np.asarray(action, dtype=np.float32).reshape(chunk_size, env.action_dim)
        obs, reward, done, info = env.step_chunk(action, gamma=gamma)
        total_reward += discount * reward
        discount *= gamma ** chunk_size
        step_count += 1
        positions.append(np.squeeze(np.asarray(env._env.get_robot_state(), dtype=np.float64))[:2])
        dist_list.append(float(env._dist_to_goal()))
        if info.get("arrive"):
            arrive = True
        if info.get("collision"):
            collision = True
        if done:
            break

    goal = np.squeeze(np.asarray(env.robot.goal, dtype=np.float64))[:2]
    return compute_episode_metrics(
        positions=np.asarray(positions),
        goal=goal,
        dist_to_goal_each_step=dist_list,
        arrive=arrive,
        collision=collision,
        episode_steps=step_count,
        step_time=env.step_time,
        total_reward=total_reward,
    )


def evaluate_policy(
    env: IrSimEnv,
    policy,
    chunk_size: int = 1,
    gamma: float = 0.99,
    episodes: int = 20,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> dict:
    """跑多个 episode 并汇总指标。"""
    if seed is not None:
        env._env.set_random_seed(seed)
    results = [
        run_episode(env, policy, chunk_size, gamma, max_steps, device)
        for _ in range(int(episodes))
    ]
    return summarize_metrics(results)


def _make_policy(actor, device):
    def policy(obs):
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        with torch.no_grad():
            return actor(t).cpu().numpy().reshape(-1, actor.chunk_size, actor.action_dim)[0]

    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description="在 ir-sim 中评估 TD3 策略")
    ap.add_argument("--checkpoint", required=True, help="FP32 checkpoint (.pt)")
    ap.add_argument("--config", default="configs/eval.yaml", help="评估配置（默认 eval.yaml；训练用 train.yaml）")
    ap.add_argument("--world", help="评估用世界 YAML（覆盖默认的 eval_world / world_name）")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save", help="保存 JSON 汇总路径")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    if args.world:
        # --world 需同时覆盖 world_name 与 eval_world：env 创建以 eval_world 优先（历史 bug：只改 world_name 不生效）
        cfg = deep_update(cfg, {"env": {"world_name": args.world, "eval_world": args.world}})
    if args.episodes:
        cfg = deep_update(cfg, {"eval": {"episodes": args.episodes}})

    actor, meta, _ = load_fp32_actor(resolve_path(args.checkpoint), args.device)
    chunk = meta["chunk_size"]
    source = args.checkpoint

    cfg = deep_update(cfg, {"obs": env_obs_cfg_from_meta(cfg["obs"], meta)})

    env = IrSimEnv(
        resolve_path(cfg["env"].get("eval_world", cfg["env"].get("world_name_evaluate", cfg["env"]["world_name"]))),
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg=cfg["env"],
        seed=args.seed,
        display=cfg["env"].get("display", False),
        log_level=cfg["env"].get("log_level", "WARNING"),
    )

    if args.episodes:
        episodes = args.episodes
    else:
        episodes = cfg["eval"].get("episodes_visual", 3)

    policy = _make_policy(actor, args.device)
    summary = evaluate_policy(
        env,
        policy,
        chunk_size=chunk,
        episodes=episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )
    summary["source"] = source
    summary["chunk_size"] = chunk

    for k, v in summary.items():
        if isinstance(v, float):
            summary[k] = round(v, 4)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(resolve_path(args.save), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # 评估结束不调用 env.close()：保留可视化窗口供查看。


if __name__ == "__main__":
    main()
