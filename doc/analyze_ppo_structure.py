#!/usr/bin/env python3
"""分析仓库内 SB3 PPO 模型的网络结构（供 doc/PPO模型结构分析.md 复现）。

用法:
    d:\anaconda\envs\rl_env\python.exe doc\analyze_ppo_structure.py
"""
from __future__ import annotations

import os
import sys

import torch
from stable_baselines3 import PPO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from env.policy import GlobalCriticActorCriticPolicy  # noqa: E402,F401  # 让 PPO.load 能反序列化自定义 policy_class
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_structure_report.txt")

MODELS = [
    "checkpoints/ppo_N1/best_model.zip",
    "checkpoints/ppo_N1/ppo_model.zip",
    "checkpoints/ppo_smoke_N1/best_model.zip",
    "checkpoints/ppo_smoke_N1/ppo_model.zip",
]

_LINES: list[str] = []


def _print(s: str = "") -> None:
    _LINES.append(s)


def count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def dump_seq(name: str, seq: torch.nn.Module) -> None:
    _print(f"  [{name}]")
    if isinstance(seq, torch.nn.Linear):
        seq = [seq]
    for i, layer in enumerate(seq):
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.shape
            b = layer.bias.shape if layer.bias is not None else None
            n = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            _print(f"    {i}: Linear({w[1]} -> {w[0]})  params={n}")
        else:
            _print(f"    {i}: {type(layer).__name__}")


def analyze(path: str) -> None:
    full = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        _print(f"=== {path} : 不存在，跳过 ===")
        _print()
        return
    _print(f"=== {path} ===")
    model = PPO.load(full, device="cpu")
    pol = model.policy
    pol.eval()

    obs = pol.observation_space
    act = pol.action_space
    _print(f"policy_class      : {model.policy_class.__name__}")
    _print(f"observation_space : shape={obs.shape} dtype={obs.dtype}")
    _print(f"action_space      : shape={getattr(act, 'shape', None)} low={getattr(act, 'low', None)} high={getattr(act, 'high', None)}")
    _print(f"net_arch          : {pol.net_arch}")
    _print(f"activation_fn     : {pol.activation_fn.__name__}")
    _print(f"squash_output     : {pol.squash_output}")
    _print(f"share_features    : {pol.share_features_extractor}")

    _print("--- 子网络（mlp_extractor + 输出头） ---")
    ext = pol.mlp_extractor
    for name, child in ext.named_children():
        dump_seq(f"mlp_extractor.{name}", child)
    dump_seq("action_net", pol.action_net)
    dump_seq("value_net", pol.value_net)

    n_ext = count_params(ext)
    n_act = count_params(pol.action_net)
    n_val = count_params(pol.value_net)
    n_total = count_params(pol)
    _print("--- 参数量 ---")
    _print(f"  mlp_extractor(共享+双头前置) : {n_ext}")
    _print(f"  action_net                 : {n_act}")
    _print(f"  value_net(输出层)           : {n_val}")
    _print(f"  策略总参数量                : {n_total}")

    _print("--- 超参数（zip 内保存值） ---")
    for k in ["learning_rate", "gamma", "gae_lambda", "n_steps", "batch_size", "n_epochs",
              "clip_range", "ent_coef", "vf_coef", "max_grad_norm"]:
        _print(f"  {k:14s}: {getattr(model, k, None)}")
    _print(f"  num_timesteps : {model.num_timesteps}")

    x = torch.randn(1, *obs.shape)
    with torch.no_grad():
        dist = pol.get_distribution(x)
        val = pol.predict_values(x)
        dist_mean = getattr(getattr(dist, "distribution", None), "mean", None)
    _print(f"  前向检查: obs {tuple(x.shape)} -> dist mean {tuple(dist_mean.shape) if dist_mean is not None else None}, value {tuple(val.shape)}")
    _print()
    del model


if __name__ == "__main__":
    _print(f"python: {sys.executable}")
    _print(f"SB3: {__import__('stable_baselines3').__version__}  torch: {torch.__version__}")
    _print()
    for p in MODELS:
        analyze(p)
    _print("done")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print("report written:", REPORT)
