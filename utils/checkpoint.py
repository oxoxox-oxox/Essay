"""模型/元数据 checkpoint 保存与加载。

checkpoint 结构:
    {
        "cfg": dict,                # 与模型相关的配置（obs/action/model/chunk）
        "step": int,
        "tag": str,
        "actor": state_dict,
        "critic1": state_dict,
        "critic2": state_dict,
        "extra": dict,              # 额外内容（如 QAT 步数/量化配置）
    }
"""

from __future__ import annotations

import os

import torch


def _model_metadata(cfg: dict) -> dict:
    return {
        "lidar_len": int(cfg["obs"]["lidar_len"]),
        "goal_dim": int(cfg["obs"].get("goal_dim", 2)),
        "action_dim": int(cfg["action"]["action_dim"]),
        "chunk_size": int(cfg["chunk"]["size"]),
        "model_cfg": dict(cfg.get("model", {})),
    }


def save_checkpoint(
    path: str,
    cfg: dict,
    actor: torch.nn.Module,
    critic1: torch.nn.Module | None = None,
    critic2: torch.nn.Module | None = None,
    step: int = 0,
    tag: str = "",
    extra: dict | None = None,
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "cfg": cfg,
        "meta": _model_metadata(cfg),
        "step": int(step),
        "tag": tag,
        "actor": actor.state_dict(),
        "critic1": critic1.state_dict() if critic1 is not None else None,
        "critic2": critic2.state_dict() if critic2 is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)
