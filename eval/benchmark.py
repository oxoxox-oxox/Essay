"""推理延迟与频率基准（FP32 torch 模型，PC 端参考）。

用法:
    python eval/benchmark.py --checkpoint checkpoints/mlptd3_N1/model.pt \
        [--iterations 500] [--save runs/benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_path  # noqa: E402

from eval.policy import load_fp32_actor  # noqa: E402


def measure_latency(
    model: torch.nn.Module,
    obs: np.ndarray,
    device: str | torch.device = "cpu",
    iterations: int = 500,
    warmup: int = 100,
) -> dict[str, float]:
    """测量单次推理延迟。

    Returns:
        dict: {"latency_ms_mean", "latency_ms_std", "inference_freq_hz_max"}。
    """
    model.eval()
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)

    with torch.no_grad():
        for _ in range(int(warmup)):
            model(obs_t)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(int(iterations)):
            t0 = time.perf_counter()
            model(obs_t)
            times.append((time.perf_counter() - t0) * 1000.0)

    mean = float(np.mean(times))
    std = float(np.std(times))
    return {
        "latency_ms_mean": mean,
        "latency_ms_std": std,
        "inference_freq_hz_max": 1000.0 / mean if mean > 0 else float("inf"),
    }


def inference_frequency_from_latency(
    latency_ms: float, chunk_size: int = 1, step_time: float = 0.1
) -> dict[str, float]:
    """由单次推理延迟推导控制环推理频率。

    说明（对应 plan.md 指标）:
        - 无 chunking (N=1): 每个控制步推理一次 -> 频率 = 1/step_time。
        - 有 chunking (N): 每个 N 控制步推理一次 -> 频率 = 1/(N*step_time)。
        - 瓶颈视角: 若推理延迟大于执行 N 步所需时间，则实际频率由延迟决定。

    Returns:
        dict: {"control_freq_hz", "decisions_per_sec", "latency_bound_freq_hz"}。
    """
    exec_time = chunk_size * step_time  # 一次推理结果可开环执行的时长
    control_freq = 1.0 / exec_time
    latency_bound = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return {
        "control_freq_hz": control_freq,
        "decisions_per_sec": 1.0 / exec_time,
        "latency_bound_freq_hz": latency_bound,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TD3 FP32 推理延迟基准（PC 端参考）")
    ap.add_argument("--checkpoint", required=True, help="FP32 checkpoint (.pt)")
    ap.add_argument("--config", default="configs/eval.yaml", help="评估配置（默认 eval.yaml；训练用 train.yaml）")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--step-time", type=float, default=None)
    ap.add_argument("--save", help="保存 JSON 路径")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    iterations = args.iterations or cfg["benchmark"].get("iterations", 500)
    warmup = args.warmup or cfg["benchmark"].get("warmup", 100)
    step_time = args.step_time or 0.1

    model, meta, _ = load_fp32_actor(resolve_path(args.checkpoint), args.device)
    chunk = meta["chunk_size"]

    rng = np.random.default_rng(0)
    prev_dim = int(meta.get("model_cfg", {}).get("prev_dim", 0))
    obs = rng.uniform(
        0.0, 1.0, size=meta["lidar_len"] + meta["goal_dim"] + prev_dim
    ).astype(np.float32)

    metrics = measure_latency(model, obs, args.device, iterations, warmup)
    result: dict = {
        **metrics,
        **inference_frequency_from_latency(metrics["latency_ms_mean"], chunk, step_time),
        "obs_dim": meta["lidar_len"] + meta["goal_dim"] + prev_dim,
        "chunk_size": chunk,
    }

    print(json.dumps(result, indent=2))

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(resolve_path(args.save), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
