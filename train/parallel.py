"""多进程并行环境：把 episode 采样分散到多个 worker 进程，主进程做梯度更新与评估。

吞吐量随 worker 数近似线性提升（瓶颈从单进程 ir-sim 仿真转移到主进程消费）。

通信协议:
    - weight_queue (main -> worker):
        None                      -> 停止
        {"version": int, "state_dict": ...}  -> 最新 actor 权重
    - result_queue (worker -> main):
        (worker_id, transitions, ep_reward, ep_steps)
        transitions: [(obs, action_flat, reward, next_obs, done), ...]
"""

from __future__ import annotations

import os
import sys
from queue import Empty

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.wrapper import IrSimEnv  # noqa: E402
from model.td3 import build_actor_critic  # noqa: E402
from utils.config import resolve_path  # noqa: E402


def select_action_online(actor: torch.nn.Module, obs: torch.Tensor, noise: float = 0.0):
    """与 TD3Agent.select_action 一致的单观测前向（worker 端复用）。"""
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    with torch.no_grad():
        action = actor(obs)[0]
    if noise > 0.0:
        action = action + torch.randn_like(action) * noise
    action = torch.clamp(action, -1.0, 1.0)
    return action.cpu()


def run_episode(env: IrSimEnv, actor: torch.nn.Module, ctx: dict):
    """在 worker 内跑一个 episode，返回 transitions + episode 统计。"""
    obs = env.reset(random=True)
    ep_reward = 0.0
    ep_steps = 0
    done = False
    transitions = []
    while not done and ep_steps < ctx["episode_max_steps"]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        action = select_action_online(actor, obs_t, ctx["exploration_noise"]).numpy()
        next_obs, r_sum, done, info = env.step_chunk(action, gamma=ctx["gamma"])
        transitions.append((obs, action.reshape(-1), r_sum, next_obs, done))
        obs = next_obs
        ep_reward += r_sum
        ep_steps += 1
    return transitions, ep_reward, ep_steps


def worker_main(
    worker_id: int,
    ctx: dict,
    weight_queue,
    result_queue,
) -> None:
    """单个采样 worker：持续跑 episode，把 transitions 发回主进程。"""
    env = IrSimEnv(
        resolve_path(ctx["cfg"]["env"]["world_name"]),
        reward_cfg=ctx["cfg"]["reward"],
        obs_cfg=ctx["cfg"]["obs"],
        env_cfg=ctx["cfg"]["env"],
        seed=ctx["seed"] + worker_id,
        display=False,
        log_level="CRITICAL",
    )
    env.max_steps = max(int(env.max_steps), ctx["episode_max_steps"] * ctx["chunk"])

    goal_dim = env.goal_dim if env.include_goal else 0
    actor, _, _ = build_actor_critic(
        lidar_len=env.lidar_len,
        goal_dim=goal_dim,
        action_dim=env.action_dim,
        chunk_size=ctx["chunk"],
        model_cfg=ctx["cfg"]["model"],
        quant_ready=True,
    )
    actor.load_state_dict(ctx["init_state_dict"])

    version = -1
    while True:
        new_weights = None
        try:
            while True:
                msg = weight_queue.get_nowait()
                if msg is None:
                    return
                if msg["version"] > version:
                    new_weights = msg["state_dict"]
                    version = msg["version"]
        except Empty:
            pass
        if new_weights is not None:
            actor.load_state_dict(new_weights)

        transitions, ep_reward, ep_steps = run_episode(env, actor, ctx)
        result_queue.put((worker_id, transitions, ep_reward, ep_steps))
