#!/usr/bin/env python3
"""用 ir-sim 演示 Gazebo 训练的 PPO 导航策略（纯 torch 前向，零新依赖）。

模型契约（原 amr_rl_ws 工作区的 Gazebo 训练；工作区已移除，模型文件在 demos/data/）:
    obs   62 维 = 60 束激光（Gazebo 360 束 -pi..+pi、3.5m 截断、每 6 束取 1、/3.5）
                 + [dist/10, wrap(atan2(dy,dx)-yaw)/pi]
    动作  2 维连续 [v, w]，deterministic = clip(mu, [0,-1], [0.5,1])
          （已用 SB3 2.9.0 的 PPO.predict 逐位验证一致，maxdiff=0）

策略网络：policy_weights.pt（62 -> 64 -> 64 -> 2，隐藏层 tanh，输出层无 tanh）
    x = tanh(Linear(62->64)(obs))
    x = tanh(Linear(64->64)(x))
    mu = Linear(64->2)(x)          # 注意：无输出 tanh（该模型 squash_output=False）
    action = clip(mu, [0,-1], [0.5,1])

激光束序：ir-sim lidar2d 与 Gazebo 约定一致（0°=车头、正角=左、ranges[i]
          对应 angle_min + i*angle_increment），已实测确认。

用法示例:
    python demos/ppo_irsim_demo.py                 # 可视化 10 个 episode
    python demos/ppo_irsim_demo.py --episodes 50 --headless   # 无窗口批量
    python demos/ppo_irsim_demo.py --noise-std 0   # 关闭激光噪声
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

import irsim

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PT = Path(__file__).resolve().parent / "data" / "policy_weights.pt"
DEFAULT_CFG = Path(__file__).resolve().parent / "data" / "model_config.yaml"
DEFAULT_WORLD = Path(__file__).resolve().parent / "ppo_grid_world.yaml"

# 训练环境常数（amr_gazebo_env.py）
MAX_LASER_RANGE = 3.5
LASER_DOWNSAMPLE = 6          # 360 束 -> 60 束
GOAL_THRESHOLD = 0.3          # 到达判定（训练用值）
COLLISION_DIST = 0.2          # 碰撞判定（训练用值）
MAX_STEPS = 500
ACT_LOW = np.array([0.0, -1.0], dtype=np.float32)
ACT_HIGH = np.array([0.5, 1.0], dtype=np.float32)

# 训练场景的 24 个固定障碍格点（amr_gazebo_env.obstacle_positions，缺 (0,0)）
OBSTACLE_POINTS = {
    (x, y)
    for x in range(-4, 5, 2)
    for y in range(-4, 5, 2)
} - {(0, 0)}


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class PPOMLP:
    """policy_weights.pt 的纯 torch 前向，输出与 SB3 model.predict(deterministic=True) 一致。"""

    def __init__(self, weights_path: Path, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["observation_dim"] == 62, cfg
        assert cfg["action_dim"] == 2 and not cfg["is_discrete"], cfg
        self.sd = torch.load(weights_path, map_location="cpu")
        for k in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.2.weight",
            "action_net.weight",
        ):
            assert k in self.sd, f"policy_weights.pt 缺少 {k}"

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs (62,) float32 -> action [v, w]（已 clip 到动作空间，等价 SB3）。"""
        x = torch.as_tensor(obs, dtype=torch.float32)
        s = self.sd
        x = torch.tanh(torch.nn.functional.linear(x, s["mlp_extractor.policy_net.0.weight"], s["mlp_extractor.policy_net.0.bias"]))
        x = torch.tanh(torch.nn.functional.linear(x, s["mlp_extractor.policy_net.2.weight"], s["mlp_extractor.policy_net.2.bias"]))
        mu = torch.nn.functional.linear(x, s["action_net.weight"], s["action_net.bias"])
        return np.clip(mu.numpy(), ACT_LOW, ACT_HIGH)


def build_obs(env, goal: np.ndarray, noise_std: float) -> np.ndarray:
    """构造与 amr_gazebo_env._get_observation 完全一致的 62 维观测。"""
    scan = env.get_lidar_scan()
    ranges = np.asarray(scan["ranges"], dtype=np.float32)
    if len(ranges) != 360:
        raise RuntimeError(f"期望 360 束激光，实际 {len(ranges)} 束；请检查 world YAML 的 lidar number")

    # 激光：截断 -> 可选高斯噪声（对齐 Gazebo 训练噪声 std=0.01）-> 每 6 束取 1 -> /3.5
    ranges = np.clip(ranges, 0.0, MAX_LASER_RANGE)
    if noise_std > 0:
        ranges = np.clip(ranges + np.random.normal(0.0, noise_std, ranges.shape).astype(np.float32),
                         0.0, MAX_LASER_RANGE)
    laser_norm = ranges[::LASER_DOWNSAMPLE] / MAX_LASER_RANGE

    # 位姿与目标
    state = np.squeeze(np.asarray(env.get_robot_state(), dtype=np.float64))
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    gx, gy = float(goal[0]), float(goal[1])

    dx, dy = gx - x, gy - y
    distance = math.hypot(dx, dy)
    angle_to_goal = wrap_pi(math.atan2(dy, dx) - yaw)
    distance_norm = min(distance / 10.0, 1.0)
    angle_norm = angle_to_goal / math.pi

    return np.concatenate([laser_norm, [distance_norm, angle_norm]]).astype(np.float32)


def sample_free_grid(rng: np.random.Generator) -> tuple[float, float]:
    """在 [-4,4]^2 的整数格点上采样一个非障碍位置（与训练一致）。"""
    while True:
        p = (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        if p not in OBSTACLE_POINTS:
            return float(p[0]), float(p[1])


def reset_episode(env, rng: np.random.Generator):
    """随机起点+目标（自由格点、互不重合），设置到 ir-sim。"""
    start = sample_free_grid(rng)
    goal = sample_free_grid(rng)
    while goal == start:
        goal = sample_free_grid(rng)

    env.reset(random=False)
    robot = env.robot
    robot.set_state([start[0], start[1], 0.0])
    robot.set_goal([goal[0], goal[1], 0.0])
    env.refresh()
    return np.array(goal, dtype=np.float64)


def run_episode(env, policy: PPOMLP, display: bool, noise_std: float,
                max_steps: int = MAX_STEPS) -> dict:
    rng = np.random.default_rng()
    goal = reset_episode(env, rng)
    obs = build_obs(env, goal, noise_std)

    state = np.squeeze(np.asarray(env.get_robot_state(), dtype=np.float64))
    start = [round(float(state[0]), 1), round(float(state[1]), 1)]
    steps = 0
    path_len = 0.0
    prev_pos = np.array([0.0, 0.0])
    min_obs_dist = float(MAX_LASER_RANGE)
    prev_pos[:] = state[:2]

    while steps < max_steps:
        action = policy.predict(obs)
        env.step(np.asarray(action, dtype=np.float64).reshape(-1, 1), action_id=0)
        if display:
            env.render(interval=0.1)
        steps += 1

        state = np.squeeze(np.asarray(env.get_robot_state(), dtype=np.float64))
        dist = math.hypot(goal[0] - state[0], goal[1] - state[1])
        path_len += float(np.hypot(state[0] - prev_pos[0], state[1] - prev_pos[1]))
        prev_pos[:] = state[:2]

        ranges = np.asarray(env.get_lidar_scan()["ranges"], dtype=np.float32)
        valid = ranges[np.isfinite(ranges)]
        min_obs_dist = min(min_obs_dist, float(np.min(valid)) if valid.size else MAX_LASER_RANGE)

        # 与训练一致：碰撞即终止（ir-sim collision_mode='stop' 时机器人会停住，
        # 若不在碰撞发生当步终止，后续步数会变成假 timeout）
        if env.robot.collision:
            outcome = "collision"
            break
        if dist < GOAL_THRESHOLD:
            outcome = "success"
            break
        if min_obs_dist < COLLISION_DIST:
            outcome = "collision"
            break
        obs = build_obs(env, goal, noise_std)
    else:
        outcome = "timeout"

    return {
        "outcome": outcome,
        "steps": steps,
        "path_len": round(path_len, 3),
        "final_dist": round(dist, 3),
        "min_obs_dist": round(min_obs_dist, 3),
        "start": start,
    }


def main():
    ap = argparse.ArgumentParser(description="ir-sim 演示 Gazebo 训练的 PPO 导航策略")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT, help="policy_weights.pt 路径")
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG, help="model_config.yaml 路径")
    ap.add_argument("--world", type=Path, default=DEFAULT_WORLD, help="ir-sim world YAML")
    ap.add_argument("--episodes", type=int, default=10, help="运行的 episode 数")
    ap.add_argument("--headless", action="store_true", help="不弹可视化窗口（批量测试用）")
    ap.add_argument("--noise-std", type=float, default=0.01, help="激光高斯噪声 std（训练噪声=0.01；0 关闭）")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    policy = PPOMLP(args.model, args.config)
    display = not args.headless
    env = irsim.make(str(args.world), display=display,
                     disable_all_plot=not display, log_level="WARNING")

    print(f"PPO loaded: {args.model}")
    print(f"obs=62  act=clip(mu,[0,-1],[0.5,1])  场景: {args.world.name}  "
          f"episodes={args.episodes} noise_std={args.noise_std}")

    results = []
    try:
        for ep in range(1, args.episodes + 1):
            info = run_episode(env, policy, display, args.noise_std, args.max_steps)
            results.append(info)
            print(f"  ep{ep:>2d}  {info['outcome']:<8s} steps={info['steps']:>3d} "
                  f"path={info['path_len']:>6.2f} final_dist={info['final_dist']:>5.2f} "
                  f"min_obs={info['min_obs_dist']:>4.2f}")
    finally:
        env.close()

    n = len(results)
    ok = sum(r["outcome"] == "success" for r in results)
    col = sum(r["outcome"] == "collision" for r in results)
    to = sum(r["outcome"] == "timeout" for r in results)
    print("-" * 60)
    print(f"summary: success={ok}/{n} ({100.0 * ok / n:.0f}%)  collision={col}  "
          f"timeout={to}  avg_steps={np.mean([r['steps'] for r in results]):.1f}")


if __name__ == "__main__":
    main()
