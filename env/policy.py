"""方向 2：Critic 独享全局特征的自定义 ActorCriticPolicy（actor 保持局部观测）。

背景
----
仓库默认 PPO 的 obs 契约是 105 维局部观测（lidar100 + goal3 + prev_action2），
actor 与 critic 共用同一输入，Critic 没有全局感知（看不到绝对位姿/绝对目标）。

本策略把环境观测向量扩展为 ``[局部特征 | 全局特征]``：
    - actor 路径只取前 ``local_obs_dim`` 维——网络输入维度与原来完全一致，
      已部署的 ONNX/INT8 真机链路（输入 105 维）不受影响；
    - critic 路径取 局部+全局 全量特征——value_net 输入维度 = local + global。

实现要点（SB3 2.9.0）
--------------------
- 重写 ``_build_mlp_extractor``：在 super().__init__ 的 ``_build`` 阶段（optimizer
  创建之前）就用 DualInputMlpExtractor 替换 mlp_extractor，保证新网络参数进入
  optimizer 并参与 SB3 的正交初始化（gain=sqrt(2)）。
- 重写 forward / evaluate_actions / get_distribution / predict_values：2.9.0 里这些
  方法各自内联了 latent 计算，统一改为走切分逻辑（DualInputMlpExtractor 内部完成）。
- 保留 SB3 命名（mlp_extractor.policy_net / value_net、action_net、value_net、
  latent_dim_pi / latent_dim_vf），train/unpack_ppo_actor.py 解包 actor 权重的逻辑
  原样可用；输出头 action_net/value_net 由 super 按 net_arch 末维建好，维度不变。
- 全局特征维度由 ``observation_space.shape[0] - local_obs_dim`` 推导，无需另传。

用法（train_ppo.py --critic-global 自动处理）::

    PPO(GlobalCriticActorCriticPolicy, env,
        policy_kwargs={"net_arch": [1024, 1024], "local_obs_dim": 105})
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy


def _build_mlp(in_dim: int, arch: list[int], activation_fn: type[nn.Module]) -> nn.Sequential:
    """与 SB3 MlpExtractor 一致：每层 Linear+激活，末尾不追加额外输出层。"""
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in arch:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(activation_fn())
        prev = int(h)
    return nn.Sequential(*layers)


class DualInputMlpExtractor(nn.Module):
    """policy_net 只吃局部观测；value_net 吃 局部+全局。

    保留 SB3 MlpExtractor 的对外接口（forward_actor / forward_critic /
    latent_dim_pi / latent_dim_vf / __call__），state_dict key 也一致
    （``mlp_extractor.policy_net.*`` / ``mlp_extractor.value_net.*``）。
    """

    def __init__(self, local_dim: int, global_dim: int, net_arch: list[int], activation_fn: type[nn.Module]) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.global_dim = int(global_dim)
        arch = [int(h) for h in net_arch]
        self.policy_net = _build_mlp(self.local_dim, arch, activation_fn)
        self.value_net = _build_mlp(self.local_dim + self.global_dim, arch, activation_fn)
        self.latent_dim_pi = arch[-1]
        self.latent_dim_vf = arch[-1]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """actor 只看局部观测（切掉全局尾段）。"""
        return self.policy_net(features[:, : self.local_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """critic 看 局部+全局 全量（全局特征拼在局部之后，直接吃全量）。"""
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class GlobalCriticActorCriticPolicy(ActorCriticPolicy):
    """Actor 保持局部观测、Critic 额外获得全局特征输入的 PPO 策略。

    Args:
        local_obs_dim: 局部观测维度（actor 网络输入维）。其余参数与
            :class:`stable_baselines3.common.policies.ActorCriticPolicy` 一致。
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        local_obs_dim: int = 105,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -3.0,
        **kwargs: Any,
    ) -> None:
        if net_arch is None:
            net_arch = [64, 64]
        net_arch = list(net_arch)

        # 必须在 super().__init__ 之前设置：_build_mlp_extractor 重写会用到
        self.local_obs_dim = int(local_obs_dim)
        obs_dim = int(observation_space.shape[0])
        self.global_obs_dim = obs_dim - self.local_obs_dim
        if not (0 < self.local_obs_dim < obs_dim):
            raise ValueError(
                f"local_obs_dim={self.local_obs_dim} 与 obs 维度 {obs_dim} 不匹配："
                f"需要 0 < local < obs（全局特征维 = {self.global_obs_dim}）"
            )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # 网络构建：在 _build（optimizer 创建前）用双输入 extractor 替换
    # ------------------------------------------------------------------ #
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DualInputMlpExtractor(
            self.local_obs_dim, self.global_obs_dim, list(self.net_arch), self.activation_fn
        )

    # ------------------------------------------------------------------ #
    # 前向：全部经由 mlp_extractor 内部的切分逻辑
    # ------------------------------------------------------------------ #
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """让 standalone policy load 也能还原 local_obs_dim（PPO.load 走 policy_kwargs，不受影响）。"""
        data = super()._get_constructor_parameters()
        data["local_obs_dim"] = self.local_obs_dim
        return data

    # 便于脚本检查的只读属性（unpack 用）
    @property
    def actor_input_dim(self) -> int:
        return self.local_obs_dim
