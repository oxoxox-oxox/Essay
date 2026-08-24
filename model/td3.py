"""TD3 网络定义（纯全连接，无 CNN）。

结构（FP32 预训练 / INT8 PTQ 共用）：
    - 整段 obs（lidar + goal + 上一 chunk 动作）直接进全连接 MLP（纯全连接 TD3）。
    - 好处：导出 ONNX 后全图均为 GEMM，TRT 可整体降 FP16/INT8，
      不再有"卷积保持 FP32 引发的反复 Reformat cast"。
    - Actor 输出经过 tanh 归一化到 [-1, 1]，shape (B, chunk, action_dim)
    - Critic 双网络 (twin)，输入 (obs, 展平的动作 chunk)，输出 Q
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor：观测 -> 动作 chunk (N, action_dim)，tanh 输出到 [-1,1]。"""

    def __init__(
        self,
        lidar_len: int,
        goal_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        hidden1: int = 400,
        hidden2: int = 300,
        prev_dim: int = 0,
        quant_ready: bool = True,
    ) -> None:
        super().__init__()
        self.lidar_len = int(lidar_len)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.prev_dim = int(prev_dim)
        self.obs_dim = self.lidar_len + self.goal_dim + self.prev_dim

        self.model_cfg = dict(
            hidden1=int(hidden1),
            hidden2=int(hidden2),
            prev_dim=self.prev_dim,
        )

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden2, action_dim * chunk_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) -> action: (B, chunk_size, action_dim) in [-1,1]"""
        a = torch.tanh(self.head(self.net(obs)))
        return a.view(-1, self.chunk_size, self.action_dim)


class Critic(nn.Module):
    """单支 Critic：Q(s, a_chunk)。输入 action 为展平的 chunk 向量。"""

    def __init__(
        self,
        lidar_len: int,
        goal_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        hidden1: int = 400,
        hidden2: int = 300,
        prev_dim: int = 0,
        quant_ready: bool = True,
    ) -> None:
        super().__init__()
        self.lidar_len = int(lidar_len)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.prev_dim = int(prev_dim)
        self.obs_dim = self.lidar_len + self.goal_dim + self.prev_dim
        self.action_dim_total = self.action_dim * self.chunk_size

        self.model_cfg = dict(
            hidden1=int(hidden1),
            hidden2=int(hidden2),
            prev_dim=self.prev_dim,
        )

        # cat(obs, 展平动作 chunk) -> 全连接 -> Q
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim_total, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden2, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim), action: (B, chunk*action_dim) -> (B, 1)"""
        x = torch.cat([obs, action], dim=-1)
        return self.q_head(self.net(x))


def build_actor_critic(
    lidar_len: int,
    goal_dim: int,
    action_dim: int,
    chunk_size: int = 1,
    model_cfg: dict | None = None,
    quant_ready: bool = True,
) -> tuple[Actor, Critic, Critic]:
    """按配置构建 Actor + 双 Critic。

    Args:
        lidar_len: lidar beam 数。
        goal_dim: goal 观测维度。
        action_dim: 单步动作维度。
        chunk_size: action chunk 长度 N。
        model_cfg: configs/train.yaml 中 model 配置段。
        quant_ready: 兼容保留（纯全连接无 QuantStub/DeQuantStub）。
    """
    model_cfg = model_cfg or {}
    kwargs = dict(
        lidar_len=lidar_len,
        goal_dim=goal_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden1=int(model_cfg.get("hidden1", model_cfg.get("hidden", 400))),
        hidden2=int(model_cfg.get("hidden2", model_cfg.get("hidden", 300))),
        prev_dim=int(model_cfg.get("prev_dim", 0)),
        quant_ready=quant_ready,
    )
    actor = Actor(**kwargs)
    critic1 = Critic(**kwargs)
    critic2 = Critic(**kwargs)
    return actor, critic1, critic2
