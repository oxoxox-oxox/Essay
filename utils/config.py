"""配置加载与路径解析。

约定：configs/train.yaml 所在目录为 project root（仓库根目录），
所有相对路径（world_name、run_dir 等）均相对该 root 解析。
"""

from __future__ import annotations

import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    """加载 YAML 配置并深拷贝，避免脚本间共享可变 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


def deep_update(base: dict, override: dict) -> dict:
    """递归合并 override 到 base 的深拷贝上（override 优先）。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str) -> str:
    """把相对路径解析到 project root，绝对路径保持不变。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_project_root() -> str:
    return PROJECT_ROOT
