"""策略加载/构建的公共工具。"""

from __future__ import annotations

import torch

from model.td3 import Actor
from utils.checkpoint import load_checkpoint
from utils.config import deep_update


def env_obs_cfg_from_meta(cfg_obs: dict, meta: dict | None) -> dict:
    """按 checkpoint 元数据构造环境观测配置。

    上一 chunk 动作的维度以模型为准（meta.model_cfg.prev_dim），
    而不是当前配置文件的开关，保证旧 checkpoint（prev_dim=0）仍可评估。
    """
    prev_dim = int((meta or {}).get("model_cfg", {}).get("prev_dim", 0))
    action_dim = int((meta or {}).get("action_dim", cfg_obs.get("action_dim", 2)))
    return deep_update(
        cfg_obs,
        {
            "include_prev_action": prev_dim > 0,
            "prev_chunk_size": prev_dim // action_dim if prev_dim > 0 else 0,
        },
    )


def build_actor(meta: dict, model_cfg: dict, quant_ready: bool = True) -> Actor:
    """按 checkpoint 元数据构建 Actor。"""
    return Actor(
        lidar_len=int(meta["lidar_len"]),
        goal_dim=int(meta["goal_dim"]),
        action_dim=int(meta["action_dim"]),
        chunk_size=int(meta["chunk_size"]),
        hidden1=int(model_cfg.get("hidden1", model_cfg.get("hidden", 400))),
        hidden2=int(model_cfg.get("hidden2", model_cfg.get("hidden", 300))),
        prev_dim=int(model_cfg.get("prev_dim", 0)),
        quant_ready=quant_ready,
    )


def load_fp32_actor(checkpoint_path: str, device: str | torch.device = "cpu") -> tuple[Actor, dict, dict]:
    """加载 FP32 训练 checkpoint 中的 Actor。

    Returns:
        (actor, meta, cfg)。
    """
    ckpt = load_checkpoint(checkpoint_path, device=device)
    meta = ckpt["meta"]
    cfg = ckpt["cfg"]
    actor = build_actor(meta, cfg.get("model", {}), quant_ready=True)
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()
    return actor, meta, cfg
