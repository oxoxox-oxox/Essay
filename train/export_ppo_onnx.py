"""把解包后的 PPO Actor 导出为纯 FP32 ONNX（供真机 TensorRT 8.5 建引擎/INT8 PTQ）。

前置（rl_env 运行 train/unpack_ppo_actor.py 生成 policy_actor.pt + actor_config.yaml）；
本脚本在 ir-sim env 运行（有 onnx/onnxruntime，无 SB3 依赖，纯 torch 前向）。

用法:
    d:/anaconda/envs/ir-sim/python.exe train/export_ppo_onnx.py --actor export/ppo_mw/policy_actor.pt
    # 同时生成 INT8 校准数据（用策略在训练世界 rollout 采集真实 obs）
    d:/anaconda/envs/ir-sim/python.exe train/export_ppo_onnx.py --actor export/ppo_mw/policy_actor.pt --make-calib

产物（export/{name}/）:
    actor_fp32_bs{N}.onnx   # 固定 batch 的纯 FP32 图（仅 Gemm/Tanh），输入 obs [B,105]，输出 action [B,2]（μ，机器人侧需 clip）
    calib_obs.npz           # --make-calib 时生成，key='obs'，shape (N,105) float32，INT8 PTQ 校准用
"""

from __future__ import annotations

import argparse
import os
import sys

# headless 服务器无 GUI：强制 matplotlib Agg 后端（本脚本恒为 headless）
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import deep_update, load_config, resolve_path  # noqa: E402


class PPOActor(nn.Module):
    """纯前向 Actor：obs -> Linear+Tanh x len(hidden) -> Linear(action_dim)（无输出激活）。"""

    def __init__(self, obs_dim: int, hidden_sizes: list[int], action_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def remap_sb3_actor_sd(sd: dict, hidden_sizes: list[int]) -> dict:
    """把 SB3 原始 key（mlp_extractor.policy_net.N / action_net）映射到 PPOActor.net。"""
    probe = PPOActor(1, hidden_sizes, 1)
    lin_positions = [i for i, m in enumerate(probe.net) if isinstance(m, nn.Linear)]
    new_sd: dict = {}
    for j, pos in enumerate(lin_positions):
        prefix = f"mlp_extractor.policy_net.{2 * j}" if j < len(hidden_sizes) else "action_net"
        new_sd[f"net.{pos}.weight"] = sd[prefix + ".weight"]
        new_sd[f"net.{pos}.bias"] = sd[prefix + ".bias"]
    return new_sd


def export_onnx(actor: PPOActor, obs_dim: int, batch: int, path: str, opset: int) -> None:
    actor.eval()
    dummy = torch.randn(batch, obs_dim)
    kw = dict(
        input_names=["obs"],
        output_names=["action"],
        opset_version=opset,
        do_constant_folding=True,
    )
    # 优先 legacy 导出器（精确输出指定 opset 的 Gemm/Tanh，适配 TRT 8.5 最高 opset 17）；
    # torch 2.x 默认 dynamo 导出器可能把 opset 顶到 18，回退时再用 version_converter 降回。
    try:
        torch.onnx.export(actor, dummy, path, dynamo=False, **kw)
    except Exception as e:
        print(f"[onnx] legacy export failed ({e}); 回退 dynamo 导出")
        torch.onnx.export(actor, dummy, path, **kw)

    # 校验
    import onnx
    model = onnx.load(path)
    cur_opset = model.opset_import[0].version
    if cur_opset > opset:
        print(f"[onnx] opset={cur_opset} > {opset}，尝试降版本")
        model = onnx.version_converter.convert_version(model, opset)
        onnx.save(model, path)
    onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    act_dim = actor.net[-1].out_features
    print(f"[onnx] {os.path.basename(path)}: opset={opset} input obs[B,{obs_dim}] -> action[B,{act_dim}], ops={ops}")
    # onnxruntime 数值对比（可选）
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = torch.randn(batch, obs_dim)
        ort_out = sess.run(None, {"obs": x.numpy()})[0]
        torch_out = actor(x).detach().numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        print(f"[onnx]   onnxruntime vs torch max-abs-diff = {max_diff:.3e}")
    except Exception as e:
        print(f"[onnx]   (跳过 onnxruntime 对比: {e})")


def collect_calib_obs(actor: PPOActor, cfg: dict, world_key: str, num_samples: int,
                      seed: int, max_steps: int, chunk_size: int = 1) -> np.ndarray:
    """用确定性策略在 ir-sim 里 rollout，采集决策点 obs 作为 INT8 校准数据。"""
    from env.wrapper import IrSimEnv

    env = IrSimEnv(
        resolve_path(cfg["env"][world_key]),
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg=cfg["env"],
        seed=seed,
        display=False,
        log_level="CRITICAL",
    )
    obs_list: list[np.ndarray] = []
    try:
        while len(obs_list) < num_samples:
            obs = env.reset(random=True)
            for _ in range(max_steps):
                obs_list.append(np.asarray(obs, dtype=np.float32))
                with torch.no_grad():
                    mu = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
                mu = np.clip(mu, -1.0, 1.0)
                if chunk_size > 1:
                    obs, _, done, _ = env.step_chunk(mu.reshape(chunk_size, -1))
                else:
                    obs, _, done, _ = env.step_single(mu)
                if done:
                    break
    finally:
        env.close()
    obs_arr = np.stack(obs_list[:num_samples]).astype(np.float32)
    print(f"[calib] collected {len(obs_arr)} obs (world={os.path.basename(cfg['env'][world_key])}) "
          f"shape={obs_arr.shape}")
    return obs_arr


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO Actor -> ONNX（ir-sim env 运行）")
    ap.add_argument("--actor", default="export/ppo_mw/policy_actor.pt",
                    help="unpack_ppo_actor.py 产物")
    ap.add_argument("--config-actor", default=None,
                    help="actor_config.yaml（默认与 --actor 同目录）")
    ap.add_argument("--batch", default="1,8", help="固定 batch 列表，逗号分隔（1=ROS 单步，8=吞吐基准）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--train-yaml", default="configs/train.yaml",
                    help="env/obs/reward 配置（采集校准数据用）")
    ap.add_argument("--make-calib", action="store_true", help="同时采集 INT8 校准数据")
    ap.add_argument("--chunk", type=int, default=None,
                    help="action chunk 长度 N（默认从 actor_config 的 action_dim//2 推断）")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-world", default=None,
                    help="校准采集用世界（默认训练世界 robot_world.yaml；可用 eval_world 等）")
    ap.add_argument("--calib-seed", type=int, default=0)
    args = ap.parse_args()

    pt_path = resolve_path(args.actor)
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = payload["actor"]
    cfg_actor_path = args.config_actor or os.path.join(os.path.dirname(pt_path), "actor_config.yaml")
    with open(cfg_actor_path, "r", encoding="utf-8") as f:
        actor_cfg = yaml.safe_load(f)

    obs_dim = int(actor_cfg["observation_dim"])
    act_dim = int(actor_cfg["action_dim"])
    hidden = [int(h) for h in actor_cfg["hidden_sizes"]]
    outdir = os.path.dirname(pt_path)
    os.makedirs(outdir, exist_ok=True)

    actor = PPOActor(obs_dim, hidden, act_dim)
    actor.load_state_dict(remap_sb3_actor_sd(sd, hidden))
    actor.eval()
    print(f"[export] obs={obs_dim} hidden={hidden} act={act_dim} "
          f"squash_output={actor_cfg.get('squash_output')}")

    batches = [int(b) for b in args.batch.split(",")]
    for b in batches:
        path = os.path.join(outdir, f"actor_fp32_bs{b}.onnx")
        export_onnx(actor, obs_dim, b, path, args.opset)

    if args.make_calib:
        cfg = load_config(resolve_path(args.train_yaml))
        chunk_size = args.chunk if args.chunk is not None else (act_dim // 2)
        # 与训练/评估一致：obs 上一动作通道长度 = chunk_size
        cfg = deep_update(cfg, {"obs": {"prev_chunk_size": chunk_size},
                                "chunk": {"size": chunk_size}})
        world_key = "eval_world" if args.calib_world else "world_name"
        if args.calib_world:
            cfg["env"][world_key] = args.calib_world
        obs_arr = collect_calib_obs(
            actor, cfg, world_key, args.calib_samples, args.calib_seed,
            int(cfg["eval"].get("max_steps", 500)), chunk_size,
        )
        calib_path = os.path.join(outdir, "calib_obs.npz")
        np.savez(calib_path, obs=obs_arr)
        print(f"[calib] saved: {calib_path}")

    print("[export] done")


if __name__ == "__main__":
    main()
