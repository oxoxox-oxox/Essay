"""Multi-world training env wrapper: picks a random world and rebuilds the scene on each reset.

Why this approach (important):
    Running multiple ir-sim instances concurrently in one process is unsafe — the module-level
    config proxies (env_param/world_param overwriting each other), the global rng, the global
    matplotlib figure, and the global object-ID counter are all shared.
    This wrapper guarantees that **only one irsim scene is active at any time**:
    on reset it closes the old env and rebuilds a randomly chosen one, sidestepping all
    global-state conflicts.

Usage (with train_ppo.py --worlds / env.worlds in configs/train.yaml):
    entries = [
        {"world": "configs/world/robot_world.yaml"},
        {"world": "configs/world/open_field.yaml",
         "start_range_low": [8, 8, -3.14], "start_range_high": [22, 22, 3.14],
         "fixed_goal": [15, 15, 0.0]},
    ]
    env = MultiWorldIrSimEnv(entries, reward_cfg, obs_cfg, env_cfg, seed=0)

The interface matches IrSimEnv (obs_dim / action_dim / reset / step_single / close),
so it can be wrapped directly by PPOGymEnv. The obs contract (lidar 100 beams ±90°, goal 3, prev 2) must be identical across worlds.
"""

from __future__ import annotations

import numpy as np

from env.wrapper import IrSimEnv
from utils.config import resolve_path

_OVERRIDE_KEYS = ("start_range_low", "start_range_high", "fixed_goal")


class MultiWorldIrSimEnv:
    """Multi-env training wrapper with a single active scene, random world on reset."""

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
            raise ValueError("MultiWorldIrSimEnv requires at least one world")
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
        # for use by PPOGymEnv
        self.obs_dim = self._env.obs_dim
        self.action_dim = self._env.action_dim

    # ------------------------------------------------------------------ #
    def _create_env(self) -> None:
        """Close the old env (including its figure), pick a random world and rebuild."""
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
        """Rebuild with a different world + random start (when random_start)."""
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
