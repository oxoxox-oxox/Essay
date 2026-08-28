"""gymnasium adapter for SB3 PPO: wraps an :class:`IrSimEnv` (shares the same obs/reward contract as the training side).

Adapts IrSimEnv's "bare" env interface into the gymnasium interface SB3 needs:
    - reset() -> (obs, info)
    - step()  -> (obs, reward, terminated, truncated, info)

Action contract (chunk-aware):
    - network output normalized to [-1, 1], shape = (chunk_size * action_dim,).
    - chunk_size=1: equivalent to single-step, calls IrSimEnv.step_single.
    - chunk_size=N>1: reshaped to (N, action_dim) then calls IrSimEnv.step_chunk (open-loop execution of N steps).
    Each [lin, ang] is mapped back to real velocity by IrSimEnv.scale_action.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.wrapper import IrSimEnv


class PPOGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, irsim_env: IrSimEnv, chunk_size: int = 1):
        super().__init__()
        self.env = irsim_env
        self.chunk_size = max(1, int(chunk_size))
        obs_dim = irsim_env.obs_dim
        # network action = chunk_size steps x action_dim per step (normalized [-1,1])
        net_action_dim = irsim_env.action_dim * self.chunk_size

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(net_action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(random=True)
        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.chunk_size > 1:
            actions = action.reshape(self.chunk_size, self.env.action_dim)
            obs, reward, done, info = self.env.step_chunk(actions)
        else:
            obs, reward, done, info = self.env.step_single(action)
        terminated = bool(info.get("arrive") or info.get("collision"))
        truncated = bool(info.get("timeout"))
        # EvalCallback uses is_success to compute the success rate
        info["is_success"] = bool(info.get("arrive"))
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()
