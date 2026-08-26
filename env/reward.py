"""奖励函数设计（对齐 DRL-robot-navigation-IR-SIM 的 SIM.get_reward + 目标 shaping）。

每步奖励：
    - 到达目标: +goal_reward (默认 100)
    - 碰撞:     +collision_penalty (默认 -100)
    - 其他:     +lin_vel - ang_penalty_scale*|ang_vel| - 障碍贴近惩罚
                + goal_angle_coef * cos(与目标夹角)      # 朝目标方向有奖励（修复空场绕圈）
                + goal_dist_coef * (上一距离 - 当前距离)  # 靠近目标 shaping（potential 式，每米）

其中障碍贴近惩罚: min(激光距离) < proximity_threshold (1.35) 时,
  减去 proximity_scale * (proximity_threshold - min_laser)，越近罚越多。
"""

from __future__ import annotations

import numpy as np


class RewardFn:
    """基于动作/障碍贴近/到达/碰撞/目标方向的奖励（DRL 风格 + goal shaping）。

    Attributes:
        cfg (dict): reward 配置段。
        last_reward (float): 最近一次返回的奖励，便于调试。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.goal_reward = float(cfg.get("goal_reward", 100.0))
        self.collision_penalty = float(cfg.get("collision_penalty", -100.0))
        self.time_penalty = float(cfg.get("time_penalty", 0.0))
        self.backward_penalty = float(cfg.get("backward_penalty", 0.0))
        self.proximity_threshold = float(cfg.get("proximity_threshold", 0.5))
        self.proximity_scale = float(cfg.get("proximity_scale", 0.5))
        self.ang_penalty_scale = float(cfg.get("ang_penalty_scale", 0.5))
        # goal shaping（0 = 关闭，保持旧行为）
        self.goal_angle_coef = float(cfg.get("goal_angle_coef", 0.0))
        self.goal_dist_coef = float(cfg.get("goal_dist_coef", 0.0))
        self._prev_dist: float | None = None

    def reset(self) -> None:
        """episode 开始时调用（重置距离 shaping 的上一距离）。"""
        self._prev_dist = None

    def __call__(
        self,
        dist_to_goal: float,
        collision: bool,
        arrive: bool,
        angle_to_goal: float | None = None,
        action: np.ndarray | None = None,
        laser_scan: np.ndarray | None = None,
    ) -> float:
        """计算单步（单控制步）奖励。

        Args:
            dist_to_goal: 当前到目标的距离（用于靠近 shaping）。
            collision: 是否碰撞。
            arrive: 是否到达目标。
            angle_to_goal: 朝向与目标方向夹角（rad，用于朝目标奖励 cos(angle)）。
            action: 施加的真实动作 [lin_vel, ang_vel]（world 单位，非归一化）。
            laser_scan: 原始 lidar ranges（米），用于障碍贴近惩罚。

        Returns:
            float: 奖励标量。
        """
        if arrive:
            return float(self.goal_reward)

        if collision:
            return float(self.collision_penalty)

        lin = float(action[0]) if action is not None else 0.0
        ang = float(action[1]) if action is not None else 0.0

        reward = lin - self.ang_penalty_scale * abs(ang)

        if lin < 0.0:
            reward += self.backward_penalty * abs(lin)

        if laser_scan is not None and len(laser_scan) > 0:
            min_laser = float(np.min(np.asarray(laser_scan)))
            if min_laser < self.proximity_threshold:
                reward -= self.proximity_scale * (self.proximity_threshold - min_laser)

        # ---- goal shaping ----
        if self.goal_angle_coef != 0.0 and angle_to_goal is not None:
            # 朝向目标 +1、背对 -1：让"朝目标走"本身有奖励（空场关键信号）
            reward += self.goal_angle_coef * float(np.cos(angle_to_goal))
        if self.goal_dist_coef != 0.0 and self._prev_dist is not None:
            # 靠近目标得正、远离得负（potential 式 shaping，每米）
            reward += self.goal_dist_coef * (self._prev_dist - float(dist_to_goal))
        self._prev_dist = float(dist_to_goal)

        self.last_reward = float(reward)
        return float(reward)


def discounted_chunk_reward(rewards: list[float], gamma: float) -> float:
    """将一个 chunk 内 N 个单步奖励折现求和: sum_k gamma^k * r_k。

    Args:
        rewards: 长度为 N 的单步奖励列表（可能提前截断）。
        gamma: 折扣因子。

    Returns:
        float: chunk 的折现回报。
    """
    total = 0.0
    discount = 1.0
    for r in rewards:
        total += discount * r
        discount *= gamma
    return float(total)


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
