"""多世界训练环境包装：每次 reset 随机换一个世界重建场景。

为什么这么做（重要）：
    ir-sim 在同一进程内多实例并发是不安全的——模块级 config 代理（env_param/world_param
    互相覆盖）、全局 rng、全局 matplotlib figure、全局对象 ID 计数器都是共享的
    （见 doc/多环境调研结论）。本包装保证**任何时刻只有一个活跃 irsim 场景**，
    在 reset 时关掉旧环境、按世界列表随机选一个重建，从而绕开所有全局状态冲突。

用法（配合 train_ppo.py --worlds / configs/train.yaml 的 env.worlds）:
    entries = [
        {"world": "configs/world/robot_world.yaml"},
        {"world": "configs/world/open_field.yaml",
         "start_range_low": [8, 8, -3.14], "start_range_high": [22, 22, 3.14],
         "fixed_goal": [15, 15, 0.0]},
    ]
    env = MultiWorldIrSimEnv(entries, reward_cfg, obs_cfg, env_cfg, seed=0)

接口与 IrSimEnv 一致（obs_dim / action_dim / reset / step_single / close），
可直接被 PPOGymEnv 包装。obs 契约（lidar 100 束 ±90°、goal 3、prev 2）必须各世界一致。
"""

from __future__ import annotations

import numpy as np

from env.wrapper import IrSimEnv
from utils.config import resolve_path

_OVERRIDE_KEYS = ("start_range_low", "start_range_high", "fixed_goal")


class MultiWorldIrSimEnv:
    """单活跃场景、reset 随机换世界的多环境训练包装。"""

    def __init__(
        self,
        entries: list[dict],
        reward_cfg: dict,
        obs_cfg: dict,
        env_cfg: dict,
        seed: int | None = None,
        display: bool = False,
        log_level: str = "WARNING",
    ) -> None:
        if not entries:
            raise ValueError("MultiWorldIrSimEnv 需要至少一个世界")
        self._entries = [
            {**dict(e), "world": resolve_path(e["world"])} for e in entries
        ]
        self._reward_cfg = reward_cfg
        self._obs_cfg = obs_cfg
        self._base_env_cfg = dict(env_cfg)
        self._rng = np.random.default_rng(seed)
        self._display = display
        self._log_level = log_level
        self._env: IrSimEnv | None = None
        self._created = False
        self._create_env()
        # 供 PPOGymEnv 使用
        self.obs_dim = self._env.obs_dim
        self.action_dim = self._env.action_dim

    # ------------------------------------------------------------------ #
    def _create_env(self) -> None:
        """关掉旧环境（含 figure），随机选一个世界重建。"""
        if self._env is not None:
            self._env.close()
        entry = self._entries[int(self._rng.integers(len(self._entries)))]
        env_cfg = dict(self._base_env_cfg)
        for k in _OVERRIDE_KEYS:
            if k in entry:
                env_cfg[k] = entry[k]
        world_seed = int(self._rng.integers(0, 2**31 - 1))
        self._env = IrSimEnv(
            entry["world"],
            reward_cfg=self._reward_cfg,
            obs_cfg=self._obs_cfg,
            env_cfg=env_cfg,
            seed=world_seed,
            display=self._display,
            log_level=self._log_level,
        )
        self._world_name = entry["world"]

    # ------------------------------------------------------------------ #
    def reset(self, random: bool = True) -> np.ndarray:
        """换世界重建 + 随机起点（random_start 时）。"""
        if self._created:
            self._create_env()
        self._created = True
        return self._env.reset(random=random)

    def step_single(self, action: np.ndarray):
        return self._env.step_single(action)

    def step_chunk(self, actions: np.ndarray, gamma: float = 0.99):
        return self._env.step_chunk(actions, gamma=gamma)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
