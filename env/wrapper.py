"""ir-sim 环境封装：把 world YAML 变成 RL 训练/评估可用的接口。

观测 (obs, float32 向量)，输入仿照 DRL-robot-navigation-IR-SIM 的 prepare_state：
    [lidar 分箱 min 值 / range_max (max_bins,),
     dist / goal_dist_norm, cos, sin,
     上一 chunk 动作归一化 (action_dim*prev_chunk_size,)（可选）,
     全局特征 (global_dim,)（可选；obs.include_global=True 时追加，给 Critic 用）]
动作 (numpy, (action_dim,)):
    归一化到 [-1,1] 网络输出，内部映射回真实速度范围 [vel_min, vel_max]
    （diff 机器人: [linear_vel, angular_vel]）。

lidar 分箱（对齐 DRL prepare_state）:
    - bin_size = ceil(lidar_len / max_bins)，每箱取 min 值再 / range_max。
    - max_bins = lidar_len 时 bin_size=1，等价于逐束 / range_max。

上一 chunk 动作（action chunk 适配版）:
    - 在 chunk 模式下模型每个决策点（每 N 步）跑一次，观测中编码
      "刚刚执行的那一个 chunk"，而不是单步的上一动作。
    - 逐维按 DRL 归一化：每步 [lin*2, (ang+1)/2]。
    - episode 起点用全零；chunk 中途提前结束(done)时用零补齐到定长。
"""

from __future__ import annotations

import os
import sys

# irsim 在 import 时会显式 matplotlib.use("TkAgg"/"Qt5Agg")（irsim.env.env_base 的
# BACKEND_PREFERENCES），headless 服务器上会打印一堆 "Failed to use ... backend" 告警
# 后才回退 Agg（MPLBACKEND 环境变量对显式 use() 无效）。这里在 irsim 之前拦截
# matplotlib.use：GUI 后端加载失败时静默回退 Agg；桌面/Windows 环境 use 成功，行为不变。
if "matplotlib" not in sys.modules:
    try:
        import matplotlib as _mpl

        _real_use = _mpl.use

        def _silent_fallback_use(backend, *args, **kwargs):
            try:
                return _real_use(backend, *args, **kwargs)
            except Exception:
                return _real_use("Agg", *args, **kwargs)

        _mpl.use = _silent_fallback_use
    except ImportError:
        pass

import numpy as np
import irsim
from irsim.util.random import rng

from .reward import RewardFn, wrap_angle


class IrSimEnv:
    """包装 :class:`irsim.env.EnvBase` 为 RL 环境。

    Args:
        world_name: 世界 YAML 路径（绝对或相对仓库根目录）。
        reward_cfg: reward 配置段。
        obs_cfg: obs 配置段（lidar_range_max 等）。
        env_cfg: env 配置段（random_start / start_range / fixed_goal 等）。
        seed: ir-sim 随机种子。
        display: 是否显示可视化（训练时 False）。
        log_level: ir-sim 日志级别。
    """

    def __init__(
        self,
        world_name: str,
        reward_cfg: dict,
        obs_cfg: dict,
        env_cfg: dict | None = None,
        seed: int | None = None,
        display: bool = False,
        log_level: str = "WARNING",
    ) -> None:
        self._env = irsim.make(
            world_name,
            display=display,
            disable_all_plot=not display,
            log_level=log_level,
            seed=seed,
        )

        # 覆盖仿真步长（env_cfg.step_time，如 0.0833 对齐 12Hz 真机帧率）。
        # ir-sim 物理积分用 _world_param.step_time，时钟/显示用 _world.step_time，两处都要改。
        env_cfg = env_cfg or {}
        if env_cfg.get("step_time") is not None:
            st = float(env_cfg["step_time"])
            self._env._world.step_time = st
            self._env._world_param.step_time = st

        self.display = display
        self.render_interval = float(obs_cfg.get("render_interval", self._env.step_time))
        self.lidar_range_max = float(obs_cfg.get("lidar_range_max", 10.0))
        self.goal_dist_norm = float(obs_cfg.get("goal_dist_norm", 10.0))
        self.include_goal = bool(obs_cfg.get("include_goal", True))
        self.goal_dim = int(obs_cfg.get("goal_dim", 2))
        self.max_steps = int(reward_cfg.get("max_steps", 500))

        self.random_start = bool(env_cfg.get("random_start", False))
        self.start_range_low = np.asarray(
            env_cfg.get("start_range_low", [0.5, 0.5, -3.14]), dtype=float
        )
        self.start_range_high = np.asarray(
            env_cfg.get("start_range_high", [9.5, 9.5, 3.14]), dtype=float
        )
        self.fixed_goal = env_cfg.get("fixed_goal", None)

        self.action_dim = int(obs_cfg.get("action_dim", 2))
        self.vel_min = np.array(obs_cfg.get("vel_min", [-1.0, -1.0]), dtype=np.float32)
        self.vel_max = np.array(obs_cfg.get("vel_max", [1.0, 1.0]), dtype=np.float32)

        self.include_prev_action = bool(obs_cfg.get("include_prev_action", False))
        self.prev_chunk_size = int(obs_cfg.get("prev_chunk_size", 0))
        self.prev_action_dim = (
            self.action_dim * self.prev_chunk_size if self.include_prev_action else 0
        )
        self._prev_chunk = np.zeros(self.prev_action_dim, dtype=np.float32)

        # 全局特征（方向2：仅追加给 Critic；actor 由 GlobalCriticActorCriticPolicy 切掉尾部）
        # 维度固定 6：[x/scale, y/scale, sin(theta), cos(theta), gx/scale, gy/scale]
        self.include_global = bool(obs_cfg.get("include_global", False))
        self.global_scale = float(obs_cfg.get("global_scale", 30.0))
        self.global_dim = 6 if self.include_global else 0

        self.reward_fn = RewardFn(reward_cfg)
        self.step_time = float(self._env.step_time)

        scan = self._env.get_lidar_scan()
        self.lidar_len = int(np.asarray(scan["ranges"]).size)

        self.max_bins = int(obs_cfg.get("max_bins", self.lidar_len))
        self.obs_dim = (
            self.max_bins + (self.goal_dim if self.include_goal else 0) + self.prev_action_dim + self.global_dim
        )
        self.step_count = 0
        self._prev_dist: float | None = None

    @property
    def robot(self):
        """当前机器人对象（reset(random=True) 后对象会被重建，须动态获取）。"""
        return self._env.robot

    @property
    def env(self):
        return self._env

    # ------------------------------------------------------------------ #
    # 内部状态/观测读取
    # ------------------------------------------------------------------ #
    def _dist_to_goal(self) -> float:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        return float(np.hypot(dx, dy))

    def _angle_to_goal(self) -> float:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        return wrap_angle(float(np.arctan2(dy, dx)) - float(state[2]))

    def _goal_obs(self) -> np.ndarray:
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = float(np.hypot(dx, dy))
        angle = self._angle_to_goal()
        return np.array(
            [dist / self.goal_dist_norm, np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        )[: self.goal_dim]

    def _bin_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """DRL 式 lidar 分箱：每 bin_size 束取 min 值再 / range_max。

        Args:
            ranges: 原始 lidar ranges（inf 视为 range_max）。

        Returns:
            (max_bins,) float32，值在 [0, 1]。
        """
        ranges = np.where(np.isinf(ranges), self.lidar_range_max, ranges)
        ranges = np.clip(ranges, 0.0, self.lidar_range_max)
        bin_size = int(np.ceil(len(ranges) / self.max_bins))
        min_values = []
        for i in range(0, len(ranges), bin_size):
            bin = ranges[i : i + min(bin_size, len(ranges) - i)]
            min_values.append(min(bin))
        # 补齐到 max_bins（lidar_len 不是 max_bins 整数倍时尾部补满量程=无遮挡）
        while len(min_values) < self.max_bins:
            min_values.append(self.lidar_range_max)
        return (np.asarray(min_values, dtype=np.float32) / self.lidar_range_max)[: self.max_bins]

    def _global_obs(self) -> np.ndarray:
        """全局特征（绝对位姿 + 绝对目标，归一化到约 [0,1]/[-1,1]）。

        仅追加给 Critic（GlobalCriticActorCriticPolicy 切掉 actor 不用的尾段）；
        维度固定 6：[x/scale, y/scale, sin(theta), cos(theta), gx/scale, gy/scale]。
        """
        state = np.squeeze(np.asarray(self._env.get_robot_state(), dtype=np.float64))
        goal = np.squeeze(np.asarray(self.robot.goal, dtype=np.float64))
        s = self.global_scale
        return np.array(
            [
                state[0] / s, state[1] / s,
                np.sin(state[2]), np.cos(state[2]),
                goal[0] / s, goal[1] / s,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        ranges = np.asarray(self._env.get_lidar_scan()["ranges"], dtype=np.float64)
        obs = [self._bin_lidar(ranges)]
        if self.include_goal:
            obs.append(self._goal_obs())
        if self.include_prev_action:
            obs.append(self._prev_chunk)
        if self.include_global:
            obs.append(self._global_obs())
        return np.concatenate(obs).astype(np.float32)

    def _set_prev_chunk(self, executed: np.ndarray) -> None:
        """把刚执行完的动作（可能被截断）定长化为上一 chunk 观测。

        逐维按 DRL 归一化：每步 [lin*2, (ang+1)/2]（action 已归一化 [-1,1]）。
        """
        if not self.include_prev_action:
            return
        a = np.asarray(executed, dtype=np.float32).reshape(-1, self.action_dim)
        n = self.prev_action_dim
        chunk = np.zeros(n, dtype=np.float32)
        for k, row in enumerate(a):
            if k >= self.prev_chunk_size:
                break
            lin = row[0] * 2.0
            ang = (row[1] + 1.0) / 2.0
            chunk[k * self.action_dim : (k + 1) * self.action_dim] = [lin, ang]
        self._prev_chunk = chunk

    def _robot_status(self) -> tuple[bool, bool]:
        collision = bool(self.robot.collision)
        arrive = bool(self.robot.arrive)
        return collision, arrive

    # ------------------------------------------------------------------ #
    # 动作映射
    # ------------------------------------------------------------------ #
    def scale_action(self, action_norm: np.ndarray) -> np.ndarray:
        """网络输出 [-1,1] -> 真实速度。"""
        a = np.clip(np.asarray(action_norm, dtype=np.float64), -1.0, 1.0)
        return self.vel_min + (a + 1.0) / 2.0 * (self.vel_max - self.vel_min)

    # ------------------------------------------------------------------ #
    # RL 接口
    # ------------------------------------------------------------------ #
    def _maybe_randomize_start(self) -> None:
        """随机机器人初始位置（保持 goal 固定）。

        在 reset(random=True) 重建场景后调用：重采样一个不与障碍重叠的
        起点，并把 goal 固定在配置值（世界 YAML 里的 goal 也可保持固定）。
        """
        if not self.random_start:
            return
        if self.fixed_goal is not None:
            self.robot.set_goal(list(self.fixed_goal))
        obstacles = [o for o in self._env.objects if o is not self.robot]
        for _ in range(50):
            pose = rng.uniform(self.start_range_low, self.start_range_high)
            self.robot.set_state(pose)
            if not any(self.robot.geometry.intersects(o.geometry) for o in obstacles):
                break
        self._env.refresh()

    def reset(self, random: bool = True) -> np.ndarray:
        """重置环境并返回初始观测（上一 chunk 动作清零）。"""
        self._env.reset(random=random)
        self._maybe_randomize_start()
        self.step_count = 0
        self.reward_fn.reset()
        self._prev_dist = None
        self._prev_chunk = np.zeros(self.prev_action_dim, dtype=np.float32)
        return self._get_obs()

    def step_single(self, action_norm: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """执行单个控制步。

        Args:
            action_norm: 归一化动作 (action_dim,)。

        Returns:
            (obs, reward, done, info) 与 gym 一致。
        """
        action = self.scale_action(action_norm)
        self._env.step(np.asarray(action, dtype=np.float64).reshape(-1, 1), action_id=0)
        if self.display:
            self._env.render(interval=self.render_interval)
        self.step_count += 1

        obs = self._get_obs()
        collision, arrive = self._robot_status()
        timeout = self.step_count >= self.max_steps
        done = bool(arrive or collision or timeout)

        dist = self._dist_to_goal()
        laser_scan = np.asarray(self._env.get_lidar_scan()["ranges"], dtype=np.float64)
        reward = self.reward_fn(
            dist,
            collision,
            arrive,
            self._angle_to_goal(),
            action=action,
            laser_scan=laser_scan,
        )
        self._prev_dist = dist

        info = {
            "collision": collision,
            "arrive": arrive,
            "timeout": timeout,
            "dist_to_goal": dist,
            "step": self.step_count,
            "time": float(self._env.time),
        }
        return obs, reward, done, info

    def step_chunk(
        self, actions: np.ndarray, gamma: float = 0.99
    ) -> tuple[np.ndarray, float, bool, dict]:
        """开环执行一个动作 chunk（长度 N）。

        Args:
            actions: 归一化动作数组 (N, action_dim)。
            gamma: 用于 chunk 内折现求和。

        Returns:
            (next_obs, discounted_chunk_reward, done, info)。
            若 chunk 中途 done（到达/碰撞/超时），提前截断。
        """
        rewards: list[float] = []
        info: dict = {}
        done = False
        executed: list[np.ndarray] = []
        for step, a in enumerate(np.asarray(actions).reshape(-1, self.action_dim)):
            obs, reward, done, info = self.step_single(a)
            executed.append(np.asarray(a, dtype=np.float32))
            rewards.append(reward)
            if done:
                break
        chunk_reward = sum(
            (gamma**k) * r for k, r in enumerate(rewards)
        )
        # 决策点观测：状态为 chunk 末态，上一 chunk 动作为刚执行的动作
        self._set_prev_chunk(executed)
        obs = self._get_obs()
        return obs, float(chunk_reward), done, info

    def close(self) -> None:
        """关闭环境并释放其 matplotlib figure。

        注意: ``EnvBase.end()`` 在 ``disable_all_plot=True``（headless 训练）时
        会提前返回，不会执行 ``plt.close``，导致 figure 只增不减（>20 个就报警）。
        这里补一次针对本环境 figure 的显式关闭。
        """
        try:
            self._env.close()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt

            env_plot = getattr(self._env, "_env_plot", None)
            if env_plot is not None:
                plt.close(env_plot.fig)
        except Exception:
            pass
