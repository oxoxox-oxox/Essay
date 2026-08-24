"""从 replay buffer 采样真实 obs，生成 TRT INT8 校准数据集。

用法:
    python train/make_calib_data.py --name mlptd3 --chunk 5 \
        [--num-samples 256] [--out export/mlptd3_N5/calib_obs.npz]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alg.buffer import ReplayBuffer  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="从 replay buffer 生成 INT8 校准数据集")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", default="mlptd3")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    tag = f"{args.name}_N{args.chunk}"

    # obs 维度以 checkpoint meta 为准（与训练一致，含 prev_dim）
    ckpt_path = resolve_path(os.path.join(cfg["project"]["checkpoint_dir"], tag, "model.pt"))
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"缺少 checkpoint: {ckpt_path}（请先训练）")
    ckpt = load_checkpoint(ckpt_path, device="cpu")
    meta = ckpt["meta"]
    obs_dim = int(meta["lidar_len"]) + int(meta["goal_dim"]) + int(
        meta.get("model_cfg", {}).get("prev_dim", 0)
    )
    action_dim_total = int(meta["action_dim"]) * int(meta["chunk_size"])

    buffer_path = resolve_path(os.path.join(cfg["project"]["buffer_dir"], tag, "buffer.npz"))
    if not os.path.exists(buffer_path):
        raise SystemExit(f"缺少 replay buffer: {buffer_path}（请先训练）")
    buffer = ReplayBuffer(cfg["td3"]["buffer_size"], obs_dim, action_dim_total, seed=args.seed)
    buffer.load(buffer_path)
    n = len(buffer)
    if n == 0:
        raise SystemExit("buffer 为空")

    rng = np.random.default_rng(args.seed)
    num = min(int(args.num_samples), n)
    idx = rng.choice(n, size=num, replace=False)
    obs = np.asarray(buffer.obs[idx], dtype=np.float32)  # (num, obs_dim)

    out = args.out or os.path.join(cfg["project"]["export_dir"], tag, "calib_obs.npz")
    out = resolve_path(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, obs=obs)
    print(f"[calib] {num} obs (obs_dim={obs_dim}) -> {out}")

    deploy_dir = resolve_path(
        cfg["project"].get("deploy_models_dir", "deploy/td3_nav/models")
    )
    os.makedirs(deploy_dir, exist_ok=True)
    deploy_path = os.path.join(deploy_dir, "calib_obs.npz")
    shutil.copy2(out, deploy_path)
    print(f"[calib] synced to deploy: {deploy_path}")


if __name__ == "__main__":
    main()
