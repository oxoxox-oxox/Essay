"""把训练得到的 FP32 checkpoint 导出为纯 FP32 ONNX（固定 batch=8，供吞吐基准）。

板上（Jetson/TensorRT）用它生成 FP16 / FP32 / INT8(PTQ) 引擎：
    trtexec --onnx=actor_fp32_bs8.onnx --noTF32 --saveEngine=actor_fp32_bs8.engine   # FP32
    trtexec --onnx=actor_fp32_bs8.onnx --fp16   --saveEngine=actor_fp16_bs8.engine   # FP16
    trtexec --onnx=actor_fp32_bs8.onnx --int8   --saveEngine=actor_int8_bs8.engine   # INT8(TRT 原生 PTQ)

用法:
    python train/export_onnx.py --name mlptd3 --chunk 5 [--batch 8] [--opset 17]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.policy import load_fp32_actor  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402


def export_fp32_onnx(
    actor: torch.nn.Module,
    obs_dim: int,
    out_path: str,
    opset_version: int = 17,
    batch: int = 8,
) -> str:
    """把纯 FP32 Actor 导出为无 QDQ 的 ONNX（板上 FP16/FP32/INT8 引擎共用源）。

    Args:
        batch: 固定 batch 大小（obs 第 0 维写死为该值）。
    """
    actor.eval()
    dummy = torch.randn(batch, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        actor,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes=None,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # eager（torchscript-based）导出，与训练模块结构兼容
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 FP32 ONNX（固定 batch，默认 8）")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", default="mlptd3")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--ckpt", help="FP32 checkpoint 路径（默认 checkpoints/<tag>/model.pt）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--batch",
        type=int,
        default=8,
        help="固定 batch 大小（默认 8，导出 actor_fp32_bs<batch>.onnx）",
    )
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    tag = f"{args.name}_N{args.chunk}"

    export_dir = resolve_path(os.path.join(cfg["project"]["export_dir"], tag))
    os.makedirs(export_dir, exist_ok=True)

    ckpt = args.ckpt or os.path.join(cfg["project"]["checkpoint_dir"], tag, "model.pt")
    actor, _, _ = load_fp32_actor(resolve_path(ckpt), args.device)

    out_path = os.path.join(export_dir, f"actor_fp32_bs{args.batch}.onnx")
    export_fp32_onnx(
        actor, actor.obs_dim, out_path, opset_version=args.opset, batch=args.batch
    )
    print(f"[export] FP32 ONNX saved to {out_path}")

    deploy_dir = resolve_path(
        cfg["project"].get("deploy_models_dir", "deploy/td3_nav/models")
    )
    os.makedirs(deploy_dir, exist_ok=True)
    deploy_path = os.path.join(deploy_dir, os.path.basename(out_path))
    shutil.copy2(out_path, deploy_path)
    print(f"[export] synced to deploy: {deploy_path}")

    print(
        "\n[Jetson/TensorRT] 转换命令示例:\n"
        f"    trtexec --onnx={out_path} --noTF32 --saveEngine=actor_fp32_bs{args.batch}.engine\n"
        f"    trtexec --onnx={out_path} --fp16   --saveEngine=actor_fp16_bs{args.batch}.engine\n"
        f"    trtexec --onnx={out_path} --int8   --saveEngine=actor_int8_bs{args.batch}.engine  # TRT 原生 PTQ，需真实数据校准\n"
        "说明: --noTF32 关闭 TF32 得到真 FP32 基线；INT8 走 TRT 原生校准（PTQ），"
        "精度依赖校准数据质量。"
    )


if __name__ == "__main__":
    main()
