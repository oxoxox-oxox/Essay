"""Config loading and path resolution.

Convention: the directory containing configs/train.yaml is the project root (repo root);
all relative paths (world_name, run_dir, etc.) are resolved relative to that root.
"""

from __future__ import annotations

import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    """Load a YAML config and deep-copy it, avoiding shared mutable dicts across scripts."""
    with open(path, "r", encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into a deep copy of base (override takes precedence)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str) -> str:
    """Resolve a relative path against the project root; absolute paths are left unchanged."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_project_root() -> str:
    return PROJECT_ROOT
