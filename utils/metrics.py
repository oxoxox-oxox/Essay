"""评估指标汇总工具。

evaluate.py 中逐 episode 生成 dict，此处汇总为均值/标准差。
"""

from __future__ import annotations

import numpy as np

NUMERIC_KEYS = (
    "success_rate",
    "collision_rate",
    "avg_dist_to_goal",
    "path_length",
    "episode_steps",
    "episode_time",
    "total_reward",
)


def compute_episode_metrics(
    positions: np.ndarray,
    goal: np.ndarray,
    dist_to_goal_each_step: list[float],
    arrive: bool,
    collision: bool,
    episode_steps: int,
    step_time: float,
    total_reward: float,
) -> dict:
    """从单 episode 轨迹计算指标。

    Args:
        positions: (T, 2) 机器人轨迹（世界坐标）。
        goal: (2,) 目标位置。
        dist_to_goal_each_step: 每一步到目标的距离。
        arrive / collision / episode_steps / total_reward: episode 统计。
        step_time: 仿真步长（秒）。

    Returns:
        dict: 单 episode 指标。
    """
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    start = np.asarray(positions[0], dtype=np.float64)
    final = np.asarray(positions[-1], dtype=np.float64)
    straight = np.linalg.norm(goal - start) + 1e-9

    # 轨迹误差：到"起点-目标"直线的平均垂直距离
    vec = goal - start
    vec_n = vec / straight
    perp = np.linalg.norm(
        np.asarray(positions, dtype=np.float64) - start, axis=1
    )  # 简化：距离起点
    proj = (np.asarray(positions, dtype=np.float64) - start) @ vec_n
    lateral = np.sqrt(np.maximum(perp ** 2 - np.clip(proj, 0, straight) ** 2, 0.0))
    traj_error = float(lateral.mean())

    path_length = float(
        np.sum(
            np.linalg.norm(np.diff(np.asarray(positions, dtype=np.float64), axis=0), axis=1)
        )
    )

    return {
        "success": float(arrive),
        "collision": float(collision),
        "avg_dist_to_goal": float(np.mean(dist_to_goal_each_step)),
        "final_dist_to_goal": float(dist_to_goal_each_step[-1]) if dist_to_goal_each_step else 0.0,
        "trajectory_error": traj_error,
        "path_length": path_length,
        "episode_steps": float(episode_steps),
        "episode_time": float(episode_steps * step_time),
        "total_reward": float(total_reward),
    }


def summarize_metrics(episode_metrics: list[dict]) -> dict:
    """把多个 episode 的指标汇总为均值±标准差，并给出成功率/碰撞率。"""
    out: dict = {"n_episodes": len(episode_metrics)}
    if not episode_metrics:
        return out

    out["success_rate"] = float(np.mean([m["success"] for m in episode_metrics]))
    out["collision_rate"] = float(np.mean([m["collision"] for m in episode_metrics]))

    total_rewards = [m["total_reward"] for m in episode_metrics]
    out["avg_reward"] = float(np.mean(total_rewards))
    out["avg_reward_std"] = float(np.std(total_rewards))

    for key in (
        "avg_dist_to_goal",
        "final_dist_to_goal",
        "trajectory_error",
        "path_length",
        "episode_steps",
        "episode_time",
        "total_reward",
    ):
        values = [m[key] for m in episode_metrics]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))

    return out
