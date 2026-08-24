"""环形 ReplayBuffer，支持 action chunk 的存储。

transition = (obs, action_chunk_flat, reward_chunk, next_obs, done)
其中 action_chunk_flat 长度为 chunk_size * action_dim。
"""

from __future__ import annotations

import os

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim_total: int,
        seed: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim_total = int(action_dim_total)
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.empty(
            (self.capacity, self.action_dim_total), dtype=np.float32
        )
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch_size = min(int(batch_size), self.size)
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            obs=self.obs[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            next_obs=self.next_obs[: self.size],
            dones=self.dones[: self.size],
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for key in ("obs", "actions", "rewards", "next_obs", "dones"):
            arr = data[key]
            setattr(self, key, np.asarray(arr, dtype=np.float32))
        self.size = self.obs.shape[0]
        self.pos = self.size % self.capacity
