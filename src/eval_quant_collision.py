"""INT8 quantization vs FP32 collision-rate comparison experiment (ir-sim simulation).

Goal (validates the paper claim): does quantization change the robot's collision rate in navigation.

Method:
    1. Simulate the PPO Actor as INT8 with PTQ: calibrate per-layer symmetric per-tensor scales for
       activations/weights using export/ppo_mw/calib_obs.npz (256 real obs), fake-quant (round/clip/dequant) at inference.
       The pure Gemm/Tanh MLP matches TensorRT INT8 quantization semantics (output remains FP32 μ).
    2. Paired comparison: for the same batch of random starts (same seed), run the same deterministic
       trajectory (laser noise off) with FP32 and INT8 policies, counting success/collision/timeout and the paired divergence table.

Usage:
    d:\\anaconda\\envs\\rl_env\\python.exe src\\eval_quant_collision.py --episodes 300
"""
from __future__ import annotations

import argparse
import os
import sys

# headless server has no GUI: force the matplotlib Agg backend (this script is always headless)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.wrapper import IrSimEnv  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402


# --------------------------------------------------------------------------- #
# Quantization simulation (symmetric per-tensor PTQ fake-quant)
# --------------------------------------------------------------------------- #
def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric INT8 quantization + dequantization (range [-127, 127])."""
    return torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale


class QuantActor:
    """Both FP32 and INT8 forwards, sharing the same weights; only the activation/weight quantization differs."""

    def __init__(self, policy_net, action_net, calib_obs: np.ndarray):
        self.l0: torch.nn.Linear = policy_net[0]   # 105 -> 1024
        self.l1: torch.nn.Linear = policy_net[2]   # 1024 -> 1024
        self.l2: torch.nn.Linear = action_net      # 1024 -> 2
        self.calib = torch.from_numpy(calib_obs).float()

        # Calibrate per-layer activation/weight scales (symmetric, max_abs / 127)
        with torch.no_grad():
            x = self.calib
            self.a_scale0 = float(x.detach().abs().max() / 127.0)
            h0 = torch.tanh(F.linear(x, self.l0.weight, self.l0.bias))
            self.a_scale1 = float(h0.detach().abs().max() / 127.0)
            h1 = torch.tanh(F.linear(h0, self.l1.weight, self.l1.bias))
            self.a_scale2 = float(h1.detach().abs().max() / 127.0)
            self.w_scale0 = float(self.l0.weight.detach().abs().max() / 127.0)
            self.w_scale1 = float(self.l1.weight.detach().abs().max() / 127.0)
            self.w_scale2 = float(self.l2.weight.detach().abs().max() / 127.0)

    def fp32(self, obs_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = torch.tanh(F.linear(obs_t, self.l0.weight, self.l0.bias))
            h = torch.tanh(F.linear(h, self.l1.weight, self.l1.bias))
            return F.linear(h, self.l2.weight, self.l2.bias)

    def int8(self, obs_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = fake_quant(obs_t, self.a_scale0)
            w = fake_quant(self.l0.weight, self.w_scale0)
            h = torch.tanh(F.linear(x, w, self.l0.bias))
            x = fake_quant(h, self.a_scale1)
            w = fake_quant(self.l1.weight, self.w_scale1)
            h = torch.tanh(F.linear(x, w, self.l1.bias))
            x = fake_quant(h, self.a_scale2)
            w = fake_quant(self.l2.weight, self.w_scale2)
            return F.linear(x, w, self.l2.bias)  # output μ stays FP32 (consistent with TRT output)

    def scales(self) -> dict:
        return {
            "a0": self.a_scale0, "a1": self.a_scale1, "a2": self.a_scale2,
            "w0": self.w_scale0, "w1": self.w_scale1, "w2": self.w_scale2,
        }


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# Environment and trajectories
# --------------------------------------------------------------------------- #
NOISELESS_WORLD = """\
world:
  height: 10
  width: 10
  step_time: 0.3
  sample_time: 0.3
  collision_mode: 'reactive'

robot:
  - kinematics: {name: 'diff'}
    shape: {name: 'rectangle', length: 0.4, width: 0.3}
    vel_min: [ -1.0, -1.0 ]
    vel_max: [ 1.0, 1.0 ]
    state: [3, 4, 0]
    goal: [9, 9, 0]
    arrive_mode: position
    goal_threshold: 0.3

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 7
        angle_range: 3.14
        number: 100
        noise: False
        std: 0.0
        angle_std: 0.0
        offset: [ 0.101, 0, 0 ]
        alpha: 0.3

    plot:
      show_trajectory: True

obstacle:
  - shape: { name: 'circle', radius: 1.0 }
    state: [ 5, 5, 0 ]
  - shape: { name: 'circle', radius: 0.5 }
    state: [ 7, 8, 0 ]
  - shape: { name: 'circle', radius: 1.4 }
    state: [ 3, 1, 0 ]
  - shape: {name: 'rectangle', length: 1.0, width: 1.2}
    state: [8, 5, 1]
  - shape: { name: 'rectangle', length: 0.5, width: 2.1 }
    state: [ 1, 8, 1.3 ]
  - shape: { name: 'rectangle', length: 1.5, width: 0.7 }
    state: [ 6, 2, 0.5 ]
  - shape: { name: 'linestring', vertices: [ [ 0, 0 ], [ 10, 0 ], [ 10, 10 ], [ 0, 10 ],[ 0, 0 ]  ] }
    kinematics: {name: 'static'}
    state: [ 0, 0, 0 ]
"""


def make_env(cfg: dict, world_path: str) -> IrSimEnv:
    return IrSimEnv(
        world_path,
        reward_cfg=cfg["reward"],
        obs_cfg=cfg["obs"],
        env_cfg={"random_start": False, "fixed_goal": cfg["env"].get("fixed_goal")},
        seed=0,
        display=False,
        log_level="CRITICAL",
    )


def sample_start(env: IrSimEnv, rng: np.random.Generator) -> list[float]:
    obstacles = [o for o in env._env.objects if o is not env.robot]
    low = np.asarray(env.start_range_low, dtype=float)
    high = np.asarray(env.start_range_high, dtype=float)
    for _ in range(200):
        pose = rng.uniform(low, high)
        env.robot.set_state(pose)
        env._env.refresh()
        if not any(env.robot.geometry.intersects(o.geometry) for o in obstacles):
            return list(map(float, pose))
    return [1.0, 1.0, 0.0]


def run_episode(env: IrSimEnv, policy_fn, start: list[float], max_steps: int):
    env.reset(random=False)
    env.robot.set_state(start)
    env._env.refresh()
    obs = env._get_obs()
    steps = 0
    info = {}
    while steps < max_steps:
        mu = policy_fn(torch.from_numpy(obs).float().unsqueeze(0)).detach().numpy()[0]
        obs, _, done, info = env.step_single(np.clip(mu, -1.0, 1.0))
        steps += 1
        if done:
            break
    if info.get("arrive"):
        return "success", steps
    if info.get("collision"):
        return "collision", steps
    return "timeout", steps


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 quantization vs FP32 collision-rate comparison (ir-sim)")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_mw_N1/best_model.zip")
    ap.add_argument("--calib", default="export/ppo_mw/calib_obs.npz")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--outdir", default="runs/quant_collision")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    cfg["obs"]["prev_chunk_size"] = 1

    # ---- Load model + calibrate quantization ----
    model = PPO.load(resolve_path(args.checkpoint), device="cpu")
    pol = model.policy
    pol.eval()
    calib_obs = np.load(resolve_path(args.calib))["obs"].astype(np.float32)
    actor = QuantActor(pol.mlp_extractor.policy_net, pol.action_net, calib_obs)

    # Self-check: my fp32 forward == SB3 predict (deterministic μ should match after clipping)
    with torch.no_grad():
        x = torch.from_numpy(calib_obs[:8]).float()
        mine = actor.fp32(x).numpy()
    sb3_pred, _ = model.predict(calib_obs[:8], deterministic=True)
    fp32_check = float(np.abs(np.clip(mine, -1, 1) - sb3_pred).max())
    print(f"[check] fp32 forward vs SB3 predict max-abs-diff = {fp32_check:.3e}")

    # Output consistency (FP32 vs INT8 on the calibration set)
    with torch.no_grad():
        y32 = actor.fp32(torch.from_numpy(calib_obs).float()).numpy()
        y8 = actor.int8(torch.from_numpy(calib_obs).float()).numpy()
    print("[quant] calibration set (256) FP32 vs INT8 output:")
    print(f"  max-abs-diff = {np.abs(y8 - y32).max():.5f}")
    print(f"  mean-abs-diff= {np.abs(y8 - y32).mean():.5f}")
    print(f"  cos          = {cos(y8, y32):.9f}")
    print(f"  scales(a0/a1/a2/w0/w1/w2) = {[round(v,5) for v in actor.scales().values()]}")

    # ---- Environment (noise-free deterministic + static obstacles) ----
    os.makedirs(resolve_path(args.outdir), exist_ok=True)
    world_path = os.path.join(resolve_path(args.outdir), "eval_world_noiseless.yaml")
    with open(world_path, "w", encoding="utf-8") as f:
        f.write(NOISELESS_WORLD)
    env = make_env(cfg, world_path)
    rng = np.random.default_rng(args.seed)

    outcomes = {"fp32": {"success": 0, "collision": 0, "timeout": 0},
                "int8": {"success": 0, "collision": 0, "timeout": 0}}
    # Paired divergence: key = (fp32_outcome, int8_outcome)
    pairs = {}
    diff_steps = 0  # accumulated first-step action diff (quantization perturbation on the same obs)

    try:
        for ep in range(args.episodes):
            start = sample_start(env, rng)
            # first-step action diff (same obs, quantization is the only variable)
            env.reset(random=False)
            env.robot.set_state(start)
            env._env.refresh()
            o0 = torch.from_numpy(env._get_obs()).float().unsqueeze(0)
            d0 = float((actor.fp32(o0) - actor.int8(o0)).abs().max())
            diff_steps += d0

            o_fp, s_fp = run_episode(env, actor.fp32, start, args.max_steps)
            o_i8, s_i8 = run_episode(env, actor.int8, start, args.max_steps)
            outcomes["fp32"][o_fp] += 1
            outcomes["int8"][o_i8] += 1
            pairs[(o_fp, o_i8)] = pairs.get((o_fp, o_i8), 0) + 1
            if (ep + 1) % 50 == 0:
                print(f"  ... {ep + 1}/{args.episodes} episodes done")
    finally:
        env.close()

    n = args.episodes
    print("\n===== Results =====")
    for k in ("fp32", "int8"):
        c = outcomes[k]
        print(f"[{k}] success={c['success']/n:.2%} collision={c['collision']/n:.2%} "
              f"timeout={c['timeout']/n:.2%} (n={n})")

    print("\nPaired divergence table (fp32 -> int8):")
    order = ["success", "collision", "timeout"]
    for o1 in order:
        row = "  ".join(f"{o2}:{pairs.get((o1, o2), 0)}" for o2 in order)
        print(f"  fp32={o1:<9} | {row}")
    help_q = pairs.get(("collision", "success"), 0)   # quantization "rescued" from collision
    hurt_q = pairs.get(("success", "collision"), 0)   # quantization "introduced" a collision
    print(f"\ncollision->success after quantization: {help_q} eps; success->collision after quantization: {hurt_q} eps")
    print(f"net effect (collision reduction = +): {help_q - hurt_q} eps / {n} eps")
    print(f"mean first-step max-abs action perturbation: {diff_steps / n:.5f}")

    # Archive
    meta = {
        "checkpoint": args.checkpoint, "calib": args.calib, "episodes": n,
        "seed": args.seed, "fp32": outcomes["fp32"], "int8": outcomes["int8"],
        "pairs": {f"{a}|{b}": v for (a, b), v in pairs.items()},
        "output": {"max_abs_diff": float(np.abs(y8 - y32).max()),
                   "mean_abs_diff": float(np.abs(y8 - y32).mean()),
                   "cos": cos(y8, y32)},
        "first_step_action_perturbation_mean": float(diff_steps / n),
        "help_collision_to_success": help_q, "hurt_success_to_collision": hurt_q,
    }
    np.savez(os.path.join(resolve_path(args.outdir), "quant_collision_results.npz"),
             **{k: np.asarray(v) for k, v in meta.items()})
    md = os.path.join(resolve_path(args.outdir), "quant_collision_results.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# INT8 quantization vs FP32 collision-rate comparison (ir-sim simulation)\n\n")
        f.write(f"- checkpoint: `{args.checkpoint}`\n- calibration set: `{args.calib}`\n")
        f.write(f"- episodes: {n} (paired, same starts / no laser noise / static-obstacle eval_world)\n\n")
        f.write("## Output Consistency (calibration set 256)\n\n")
        f.write(f"| max-abs-diff | mean-abs-diff | cos |\n|---|---|---|\n")
        f.write(f"| {np.abs(y8-y32).max():.5f} | {np.abs(y8-y32).mean():.5f} | {cos(y8,y32):.9f} |\n\n")
        f.write("## Collision / Success Rate\n\n")
        f.write("| Condition | success | collision | timeout |\n|---|---|---|---|\n")
        for k in ("fp32", "int8"):
            c = outcomes[k]
            f.write(f"| {k} | {c['success']/n:.2%} | {c['collision']/n:.2%} | {c['timeout']/n:.2%} |\n")
        f.write(f"\nNet effect (post-quantization, collision->success minus success->collision): {help_q - hurt_q} eps / {n} eps\n")
        f.write(f"Mean first-step max-abs action perturbation: {diff_steps / n:.5f}\n")
    print(f"\nArchived: {md}")


if __name__ == "__main__":
    main()
