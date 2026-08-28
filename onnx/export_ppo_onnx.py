"""Export the unpacked PPO Actor as a pure FP32 ONNX (for building TensorRT 8.5 engines / INT8 PTQ on the robot).

Prerequisite (run onnx/unpack_ppo_actor.py in rl_env to produce policy_actor.pt + actor_config.yaml);
this script runs in the ir-sim env (has onnx/onnxruntime, no SB3 dependency, pure torch forward).

Usage:
    d:/anaconda/envs/ir-sim/python.exe onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt
    # also generate INT8 calibration data (rollout with the policy in the training world to collect real obs)
    d:/anaconda/envs/ir-sim/python.exe onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

Artifacts (export/{name}/):
    actor_fp32_bs{N}.onnx   # fixed-batch pure FP32 graph (only Gemm/Tanh), input obs [B,105], output action [B,2] (μ, needs clipping robot-side)
    calib_obs.npz           # generated with --make-calib, key='obs', shape (N,105) float32, for INT8 PTQ calibration
"""

from __future__ import annotations

import argparse
import os
import sys

# headless server has no GUI: force the matplotlib Agg backend (this script is always headless)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import deep_update, load_config, resolve_path  # noqa: E402


class PPOActor(nn.Module):
    """Pure-forward Actor: obs -> Linear+Tanh x len(hidden) -> Linear(action_dim) (no output activation)."""

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
    """Map the SB3 raw keys (mlp_extractor.policy_net.N / action_net) to PPOActor.net."""
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
    # Prefer the legacy exporter (precisely emits the specified opset's Gemm/Tanh, matching TRT 8.5's max opset 17);
    # torch 2.x's default dynamo exporter may push opset to 18, then downgrade with version_converter as a fallback.
    try:
        torch.onnx.export(actor, dummy, path, dynamo=False, **kw)
    except Exception as e:
        print(f"[onnx] legacy export failed ({e}); falling back to dynamo export")
        torch.onnx.export(actor, dummy, path, **kw)

    # Verify
    import onnx
    model = onnx.load(path)
    cur_opset = model.opset_import[0].version
    if cur_opset > opset:
        print(f"[onnx] opset={cur_opset} > {opset}, trying to downgrade")
        model = onnx.version_converter.convert_version(model, opset)
        onnx.save(model, path)
    onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    act_dim = actor.net[-1].out_features
    print(f"[onnx] {os.path.basename(path)}: opset={opset} input obs[B,{obs_dim}] -> action[B,{act_dim}], ops={ops}")
    # onnxruntime numerical comparison (optional)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = torch.randn(batch, obs_dim)
        ort_out = sess.run(None, {"obs": x.numpy()})[0]
        torch_out = actor(x).detach().numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        print(f"[onnx]   onnxruntime vs torch max-abs-diff = {max_diff:.3e}")
    except Exception as e:
        print(f"[onnx]   (skipped onnxruntime comparison: {e})")


def collect_calib_obs(actor: PPOActor, cfg: dict, world_key: str, num_samples: int,
                      seed: int, max_steps: int, chunk_size: int = 1) -> np.ndarray:
    """Roll out the deterministic policy in ir-sim, collecting decision-point obs as INT8 calibration data."""
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
    ap = argparse.ArgumentParser(description="PPO Actor -> ONNX (run in the ir-sim env)")
    ap.add_argument("--actor", default="export/ppo_final_N1/policy_actor.pt",
                    help="artifact of unpack_ppo_actor.py")
    ap.add_argument("--config-actor", default=None,
                    help="actor_config.yaml (defaults to the --actor directory)")
    ap.add_argument("--batch", default="1,8", help="fixed-batch list, comma-separated (1=ROS single-step, 8=throughput benchmark)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--train-yaml", default="configs/train.yaml",
                    help="env/obs/reward config (for collecting calibration data)")
    ap.add_argument("--make-calib", action="store_true", help="also collect INT8 calibration data")
    ap.add_argument("--chunk", type=int, default=None,
                    help="action chunk length N (default inferred from actor_config action_dim//2)")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-world", default=None,
                    help="world for calibration collection (default training world robot_world.yaml; eval_world etc. usable)")
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
        # consistent with training/eval: obs prev-action channel length = chunk_size
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
