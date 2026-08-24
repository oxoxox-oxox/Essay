"""诊断：对比指定 checkpoint 在 训练世界(robot_world) 与 评估世界(eval_world) 的表现。

用法: python eval/_diag_checkpoints.py <tag> [ckpt_name ...]
例:   python eval/_diag_checkpoints.py mlptd3_1024_N5 model_best.pt model.pt
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_path, deep_update  # noqa: E402
from env.wrapper import IrSimEnv  # noqa: E402
from eval.policy import load_fp32_actor, env_obs_cfg_from_meta  # noqa: E402
from eval.evaluate import evaluate_policy, _make_policy  # noqa: E402

WORLDS = [
    ("eval_world.yaml (static)", "configs/world/eval_world.yaml"),
    ("robot_world.yaml (dynamic)", "configs/world/robot_world.yaml"),
]

TAG = sys.argv[1] if len(sys.argv) > 1 else "mlptd3_N5"
CKPTS = sys.argv[2:] if len(sys.argv) > 2 else ["model_best.pt", "model.pt"]

for ckpt_name in CKPTS:
    for wlabel, world in WORLDS:
        cfg = load_config(resolve_path("configs/train.yaml"))
        ckpt = f"checkpoints/{TAG}/{ckpt_name}"
        actor, meta, _ = load_fp32_actor(resolve_path(ckpt), "cpu")
        cfg = deep_update(cfg, {"obs": env_obs_cfg_from_meta(cfg["obs"], meta)})
        env = IrSimEnv(
            resolve_path(world),
            reward_cfg=cfg["reward"],
            obs_cfg=cfg["obs"],
            env_cfg=cfg["env"],
            seed=0,
            display=False,
            log_level="CRITICAL",
        )
        policy = _make_policy(actor, "cpu")
        s = evaluate_policy(
            env, policy, chunk_size=meta["chunk_size"], episodes=30, device="cpu", seed=0
        )
        print(
            f"{ckpt_name:16s} | {wlabel:28s} success={s['success_rate']:.3f} "
            f"collision={s['collision_rate']:.3f} steps={s['episode_steps_mean']:.1f} "
            f"reward={s['avg_reward']:.1f}"
        )
        env.close()
