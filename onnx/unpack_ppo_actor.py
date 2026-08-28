"""Unpack the Actor weights from an SB3 PPO zip (loadable by pure torch forward, for ONNX export).

Because rl_env (SB3) and ir-sim (onnx) are two separate conda envs, ONNX export is split into two steps:
    1) this script (run in rl_env): zip -> export/{name}/policy_actor.pt + actor_config.yaml
    2) onnx/export_ppo_onnx.py (run in ir-sim): actor -> ONNX + calibration data

Usage:
    d:/anaconda/envs/rl_env/python.exe onnx/unpack_ppo_actor.py \
        --checkpoint checkpoints/ppo_mw_N1/best_model.zip --name ppo_mw

Artifacts:
    export/{name}/policy_actor.pt   # {"actor": <policy_net+action_net weights>, "log_std": ...}
    export/{name}/actor_config.yaml # obs/action dims, hidden layers, activation, etc.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Unpack the Actor weights of an SB3 PPO model (for ONNX export)")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_final_N1/best_model.zip")
    ap.add_argument("--name", default="ppo_final_N1")
    args = ap.parse_args()

    model = PPO.load(resolve_path(args.checkpoint), device="cpu")
    pol = model.policy
    pol.eval()

    # Actor part: mlp_extractor.policy_net + action_net (keys consistent with the SB3 originals, for easy cross-checking)
    actor_sd = {
        k: v
        for k, v in pol.state_dict().items()
        if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")
    }
    log_std = getattr(getattr(pol, "action_dist", None), "log_std", None)

    obs_dim = int(pol.observation_space.shape[0])
    act_dim = int(getattr(pol.action_space, "shape", [None])[0])
    hidden = [
        m.out_features
        for m in pol.mlp_extractor.policy_net
        if isinstance(m, torch.nn.Linear)
    ]
    config = {
        "observation_dim": obs_dim,
        "action_dim": act_dim,
        "hidden_sizes": hidden,
        "activation": "tanh",            # SB3 MlpPolicy default
        "squash_output": False,          # no tanh on the output layer, action = clip(mu, [-1,1])
        "net_arch": [int(x) for x in pol.net_arch]
        if isinstance(pol.net_arch, list)
        else pol.net_arch,
        "source_checkpoint": args.checkpoint,
        "obs_contract": "105 = lidar100(binned min/range_max) + goal3(dist/10,cos,sin) + prev_action2(lin*2,(ang+1)/2)",
    }

    outdir = resolve_path(os.path.join("export", args.name))
    os.makedirs(outdir, exist_ok=True)
    pt_path = os.path.join(outdir, "policy_actor.pt")
    cfg_path = os.path.join(outdir, "actor_config.yaml")

    payload = {"actor": actor_sd, "log_std": log_std}
    torch.save(payload, pt_path)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    n_params = int(sum(v.numel() for v in actor_sd.values()))
    print(f"[unpack] {args.checkpoint}")
    print(f"[unpack] obs_dim={obs_dim} act_dim={act_dim} hidden={hidden} "
          f"actor_params={n_params} (incl. log_std {log_std.numel() if log_std is not None else 0})")
    print(f"[unpack] saved: {pt_path}")
    print(f"[unpack] saved: {cfg_path}")
    print(f"[unpack] next step: d:/anaconda/envs/ir-sim/python.exe "
          f"onnx/export_ppo_onnx.py --actor {pt_path}")


if __name__ == "__main__":
    main()
