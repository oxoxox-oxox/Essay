"""TD3 智能体（支持 action chunking 的宏动作形式）。

关键点：
    - buffer 中的 transition 以 chunk 为粒度：
        (obs, chunk_action, chunk_reward, next_obs, done)
    - TD target 的 bootstrapping 折扣为 gamma^N（N = chunk_size）
    - Actor 一次输出 N 个动作，policy gradient 通过整个 chunk 反向传播

该方法在 chunk_size=1 时退化为标准 TD3。

target 网络用与在线网络相同的超参重新构建（不含 QuantStub/DeQuantStub），
从而在 QAT 微调时 target 保持 FP32，避免量化误差在 bootstrapping 中累积。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.td3 import Actor, Critic, build_actor_critic


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class TD3Agent:
    """Twin Delayed DDPG（含 chunking 支持）。

    Args:
        actor: Actor 网络。
        critic1, critic2: 双 Critic 网络。
        cfg: td3 配置段。
        chunk_size: action chunk 长度 N。
        device: torch device。
    """

    def __init__(
        self,
        actor: Actor,
        critic1: Critic,
        critic2: Critic,
        cfg: dict,
        chunk_size: int = 1,
        device: str | torch.device = "cpu",
        weight_decay: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.action_dim = actor.action_dim

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)

        # 用相同超参重建 target 网络（纯 FP32，无量化 stub）
        lidar_len = actor.lidar_len
        goal_dim = actor.goal_dim
        action_dim = actor.action_dim
        n_chunk = actor.chunk_size
        t_actor, t_c1, t_c2 = build_actor_critic(
            lidar_len,
            goal_dim,
            action_dim,
            chunk_size=n_chunk,
            model_cfg=actor.model_cfg,
            quant_ready=False,
        )
        self.actor_target = t_actor.to(self.device)
        self.critic1_target = t_c1.to(self.device)
        self.critic2_target = t_c2.to(self.device)

        self.actor_target.load_state_dict(actor.state_dict(), strict=False)
        self.critic1_target.load_state_dict(critic1.state_dict(), strict=False)
        self.critic2_target.load_state_dict(critic2.state_dict(), strict=False)
        for net in (self.actor_target, self.critic1_target, self.critic2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(
            actor.parameters(), lr=float(cfg["actor_lr"]), weight_decay=float(weight_decay)
        )
        self.critic_optim = torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=float(cfg["critic_lr"]),
            weight_decay=float(weight_decay),
        )

        self.gamma = float(cfg.get("gamma", 0.99))
        self.gamma_n = self.gamma ** self.chunk_size
        self.tau = float(cfg.get("tau", 0.005))
        self.policy_noise = float(cfg.get("policy_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        self.policy_delay = int(cfg.get("policy_delay", 2))
        self._update_count = 0

    # ------------------------------------------------------------------ #
    def select_action(self, obs: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """给定观测返回归一化动作 chunk (chunk, action_dim)。

        Args:
            obs: (obs_dim,) 或 (B, obs_dim) 的 torch.Tensor。
            noise: exploration 高斯噪声标准差（相对 [-1,1] 动作）。
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs.to(self.device))[0]
        if noise > 0.0:
            action = action + torch.randn_like(action) * noise
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu()

    # ------------------------------------------------------------------ #
    def update(self, batch: dict) -> dict[str, float]:
        """离线更新（QAT 微调中也直接调用，输入来自 buffer.sample）。

        Args:
            batch: 含 obs/actions/rewards/next_obs/dones 的 dict。

        Returns:
            dict: {"critic_loss", "actor_loss", "q_value"}。
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )

        # ---------- critic ----------
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            noise = torch.clamp(
                torch.randn_like(next_actions) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip,
            )
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0).flatten(1)
            q1_target = self.critic1_target(next_obs, next_actions)
            q2_target = self.critic2_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            td_target = rewards + self.gamma_n * (1.0 - dones) * q_target

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---------- actor（延迟更新） ----------
        actor_loss = torch.tensor(0.0, device=self.device)
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            actions_pred = self.actor(obs).flatten(1)
            actor_loss = -self.critic1(obs, actions_pred).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

        with torch.no_grad():
            q_mean = float(q1.mean().item())
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_value": q_mean,
        }

    # ------------------------------------------------------------------ #
    def save_online(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )

    def load_online(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
