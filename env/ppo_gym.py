"""SB3 PPO 用的 gymnasium 适配器：包一层 :class:`IrSimEnv`（与训练侧共用同一 obs/奖励契约）。

把 IrSimEnv 的"裸"环境接口适配成 SB3 需要的 gymnasium 接口：
    - reset() -> (obs, info)
    - step()  -> (obs, reward, terminated, truncated, info)

动作契约（chunk 感知）：
    - 网络输出归一化 [-1, 1]，shape = (chunk_size * action_dim,)。
    - chunk_size=1：等价单步，调 IrSimEnv.step_single。
    - chunk_size=N>1：reshape 成 (N, action_dim) 后调 IrSimEnv.step_chunk（开环执行 N 步）。
    每步 [lin, ang] 由 IrSimEnv.scale_action 映射回真实速度。
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
        # 网络动作 = chunk_size 步 × 每步 action_dim（归一化 [-1,1]）
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
        # EvalCallback 用 is_success 统计成功率
        info["is_success"] = bool(info.get("arrive"))
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()
