"""从 SB3 PPO zip 中解包 Actor 权重（纯 torch 前向可加载，供 ONNX 导出）。

因为 rl_env（SB3）与 ir-sim（onnx）分属两个 conda env，导出 ONNX 需要拆两步：
    1) 本脚本（rl_env 运行）: zip -> export/{name}/policy_actor.pt + actor_config.yaml
    2) train/export_ppo_onnx.py（ir-sim 运行）: actor -> ONNX + 校准数据

用法:
    d:/anaconda/envs/rl_env/python.exe train/unpack_ppo_actor.py \
        --checkpoint checkpoints/ppo_mw_N1/best_model.zip --name ppo_mw

产物:
    export/{name}/policy_actor.pt   # {"actor": <policy_net+action_net 权重>, "log_std": ...}
    export/{name}/actor_config.yaml # obs/action 维度、隐藏层、激活等描述
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
    ap = argparse.ArgumentParser(description="解包 SB3 PPO 的 Actor 权重（供 ONNX 导出）")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_mw_N1/best_model.zip")
    ap.add_argument("--name", default="ppo_mw")
    args = ap.parse_args()

    model = PPO.load(resolve_path(args.checkpoint), device="cpu")
    pol = model.policy
    pol.eval()

    # Actor 部分：mlp_extractor.policy_net + action_net（与 SB3 原始 key 一致，便于对照）
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
        "activation": "tanh",            # SB3 MlpPolicy 默认
        "squash_output": False,          # 输出层无 tanh，动作 = clip(mu, [-1,1])
        "net_arch": [int(x) for x in pol.net_arch]
        if isinstance(pol.net_arch, list)
        else pol.net_arch,
        "source_checkpoint": args.checkpoint,
        "obs_contract": "105 = lidar100(分箱min/range_max) + goal3(dist/10,cos,sin) + prev_action2(lin*2,(ang+1)/2)",
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
          f"actor_params={n_params} (含 log_std {log_std.numel() if log_std is not None else 0})")
    print(f"[unpack] saved: {pt_path}")
    print(f"[unpack] saved: {cfg_path}")
    print(f"[unpack] 下一步: d:/anaconda/envs/ir-sim/python.exe "
          f"train/export_ppo_onnx.py --actor {pt_path}")


if __name__ == "__main__":
    main()
