"""一键跑 FP32 实验矩阵（不同 chunk N 在静态/动态场景的评估）。

用法:
    python eval/run_experiments.py \
        --name mlptd3 --chunks 1,5,10 \
        --episodes 20 [--save runs/summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import deep_update, load_config, resolve_path  # noqa: E402

from eval.evaluate import evaluate_policy, _make_policy  # noqa: E402
from eval.policy import env_obs_cfg_from_meta, load_fp32_actor  # noqa: E402


def checkpoint_path(cfg: dict, name: str, chunk: int) -> str:
    return os.path.join(cfg["project"]["checkpoint_dir"], f"{name}_N{chunk}", "model.pt")


def run_experiment(
    cfg: dict,
    name: str,
    chunk: int,
    episodes: int,
    device: str,
    seed: int,
) -> dict | None:
    ckpt = checkpoint_path(cfg, name, chunk)
    if not os.path.exists(resolve_path(ckpt)):
        return None
    actor, meta, _ = load_fp32_actor(resolve_path(ckpt), device)
    chunk = meta["chunk_size"]
    policy = _make_policy(actor, device)

    row: dict = {"name": name, "quant": "fp32", "chunk_size": chunk}

    cfg = deep_update(cfg, {"obs": env_obs_cfg_from_meta(cfg["obs"], meta)})

    scenes: list[tuple[str, str]] = []
    for scene_key, world_key in (
        ("static", "world_name"),
        ("dynamic", "eval_world"),
    ):
        world = cfg["env"].get(world_key)
        if not world:
            continue
        world_path = resolve_path(world)
        if any(path == world_path for _, path in scenes):
            continue
        scenes.append((scene_key, world_path))

    for scene_key, world_path in scenes:
        env = IrSimEnv(
            world_path,
            reward_cfg=cfg["reward"],
            obs_cfg=cfg["obs"],
            env_cfg=cfg["env"],
            seed=seed,
            display=cfg["env"].get("display", False),
            log_level=cfg["env"].get("log_level", "WARNING"),
        )
        summary = evaluate_policy(
            env,
            policy,
            chunk_size=chunk,
            episodes=episodes,
            device=device,
            seed=seed,
        )
        for k, v in summary.items():
            if k != "n_episodes":
                row[f"{scene_key}_{k}"] = round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
        env.close()

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="运行 TD3 FP32 实验矩阵并汇总 CSV")
    ap.add_argument("--name", default="mlptd3")
    ap.add_argument("--chunks", default="1,5,10", help="chunk 长度 N 列表")
    ap.add_argument("--config", default="configs/eval.yaml", help="评估配置（默认 eval.yaml；训练用 train.yaml）")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--save", default="runs/summary.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    episodes = args.episodes or cfg["eval"].get("episodes", 20)
    chunks = [int(c) for c in args.chunks.split(",")]

    os.makedirs(os.path.dirname(resolve_path(args.save)) or ".", exist_ok=True)

    rows: list[dict] = []
    for chunk in chunks:
        row = run_experiment(cfg, args.name, chunk, episodes, args.device, args.seed)
        if row is None:
            print(f"[skip] chunk={chunk}: checkpoint 不存在")
            continue
        rows.append(row)
        succ = " ".join(
            f"{k}={row.get(k)}"
            for k in ("static_success_rate", "dynamic_success_rate")
            if k in row
        )
        print(f"[ok] chunk={chunk} -> {succ}")

    if rows:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with open(resolve_path(args.save), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"[done] summary saved to {args.save}")


if __name__ == "__main__":
    main()
