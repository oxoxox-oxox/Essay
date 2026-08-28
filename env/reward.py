"""Reward function design (aligned with SIM.get_reward from DRL-robot-navigation-IR-SIM + goal shaping).

Per-step reward:
    - reached goal: +goal_reward (default 100)
    - collision:    +collision_penalty (default -100)
    - otherwise:    +lin_vel - ang_penalty_scale*|ang_vel| - obstacle proximity penalty
                    - time_penalty                       # per-step time penalty (encourages fast arrival)
                    + goal_angle_coef * cos(angle to goal)      # heading toward the goal yields reward (fixes open-field circling)
                    + goal_dist_coef * (prev_dist - dist)       # approaching-goal shaping (potential-style, per meter)

where the obstacle proximity penalty is: when min(laser distance) < proximity_threshold (1.35),
   subtract proximity_scale * (proximity_threshold - min_laser), the closer the larger the penalty.
"""

from __future__ import annotations

import numpy as np


class RewardFn:
    """Reward based on action / obstacle proximity / arrival / collision / heading to goal (DRL style + goal shaping).

    Attributes:
        cfg (dict): reward config section.
        last_reward (float): the most recently returned reward, for debugging.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.goal_reward = float(cfg.get("goal_reward", 100.0))
        self.collision_penalty = float(cfg.get("collision_penalty", -100.0))
        self.time_penalty = float(cfg.get("time_penalty") or 0.0)
        self.backward_penalty = float(cfg.get("backward_penalty", 0.0))
        self.proximity_threshold = float(cfg.get("proximity_threshold", 0.5))
        self.proximity_scale = float(cfg.get("proximity_scale", 0.5))
        self.ang_penalty_scale = float(cfg.get("ang_penalty_scale", 0.5))
        # goal shaping (0 = disabled, keeps old behavior)
        self.goal_angle_coef = float(cfg.get("goal_angle_coef", 0.0))
        self.goal_dist_coef = float(cfg.get("goal_dist_coef", 0.0))
        self._prev_dist: float | None = None

    def reset(self) -> None:
        """Called at episode start (resets the previous distance for distance shaping)."""
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
        """Compute the reward for a single (single control) step.

        Args:
            dist_to_goal: current distance to the goal (for approaching shaping).
            collision: whether a collision occurred.
            arrive: whether the goal was reached.
            angle_to_goal: heading vs goal direction angle (rad, for the cos(angle) toward-goal reward).
            action: the applied real action [lin_vel, ang_vel] (world units, not normalized).
            laser_scan: raw lidar ranges (meters), for the obstacle proximity penalty.

        Returns:
            float: the reward scalar.
        """
        if arrive:
            return float(self.goal_reward)

        if collision:
            return float(self.collision_penalty)

        lin = float(action[0]) if action is not None else 0.0
        ang = float(action[1]) if action is not None else 0.0

        reward = lin - self.ang_penalty_scale * abs(ang) - self.time_penalty

        if lin < 0.0:
            reward += self.backward_penalty * abs(lin)

        if laser_scan is not None and len(laser_scan) > 0:
            min_laser = float(np.min(np.asarray(laser_scan)))
            if min_laser < self.proximity_threshold:
                reward -= self.proximity_scale * (self.proximity_threshold - min_laser)

        # ---- goal shaping ----
        if self.goal_angle_coef != 0.0 and angle_to_goal is not None:
            # facing the goal +1, facing away -1: "walking toward the goal" itself yields reward (key signal in open fields)
            reward += self.goal_angle_coef * float(np.cos(angle_to_goal))
        if self.goal_dist_coef != 0.0 and self._prev_dist is not None:
            # approaching the goal is positive, moving away is negative (potential-style shaping, per meter)
            reward += self.goal_dist_coef * (self._prev_dist - float(dist_to_goal))
        self._prev_dist = float(dist_to_goal)

        self.last_reward = float(reward)
        return float(reward)


def discounted_chunk_reward(rewards: list[float], gamma: float) -> float:
    """Discounted sum of the N single-step rewards inside a chunk: sum_k gamma^k * r_k.

    Args:
        rewards: list of N single-step rewards (may be truncated early).
        gamma: discount factor.

    Returns:
        float: the chunk's discounted return.
    """
    total = 0.0
    discount = 1.0
    for r in rewards:
        total += discount * r
        discount *= gamma
    return float(total)


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
