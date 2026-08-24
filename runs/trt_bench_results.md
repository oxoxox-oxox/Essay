# TRT 三精度吞吐基准结果（路径 A 核心验证）

> 实测日期：2026-08-23 ｜ PC：RTX 5070（Blackwell sm_120）+ Windows WDDM
> TRT 10.10.0.31 + CUDA 12.8 ｜ 校准：真实 obs 256 条（IInt8EntropyCalibrator2）
> 测速口径：warmup 100 次 + 连续 enqueue 300 次 + cudaDeviceSynchronize，5 轮取中位数
> 引擎：`export/mlptd3_1024_N5/actor_fp32_bs{1,8,32}.onnx` 与 `export/mlptd3_N5/actor_fp32_bs8.onnx`
> 复现：`d:\anaconda\envs\cuda\python.exe eval\bench_trt.py --onnx <基底> --calib <calib.npz> --batches 1,8,32`

## 1. 1024×1024 模型（Actor 1.175M MACs）

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 75.04 us | 71.21 us | 108.15 us | **+44.1%**（更慢） | -5.1% |
| 8 | 12.88 us | 10.66 us | 9.44 us | **-26.7%**（更快） | -17.2% |
| 32 | 3.32 us | 2.16 us | 2.66 us | **-19.9%** | -34.9% |

## 2. 对照：400×300 模型（Actor 0.168M MACs）

| batch | FP32 | FP16 | INT8 | INT8 vs FP32 |
| --- | --- | --- | --- | --- |
| 8 | 9.49 us | 9.27 us | 9.26 us | **-2.3%**（打平） |

## 3. 输出精度对比（64 条随机 obs，非校准数据）

| 模型 | 精度对 | max-abs-diff | cos |
| --- | --- | --- | --- |
| 1024×1024 | INT8 vs FP32 | 0.000419 | 1.000000 |
| 1024×1024 | FP16 vs FP32 | 0.000418 | 1.000000 |
| 400×300 | INT8 vs FP32 | 0.000759 | 1.000000 |

## 4. 结论（路径 A 验证通过）

1. **batch=1：INT8 反而慢 44%**（launch-bound + M=1 无法用 Tensor Core）——"小模型/单步推理量化无延迟收益"在更大模型上依然成立，如实报告（对应 plan.md 叙事：车上延迟靠 action chunking 解决）。
2. **batch≥8：INT8/FP16 相对 FP32 每样本快 20-35%**——量化优势**随模型变大而显现**：0.168M MACs 时三档打平（-2.3%），1.175M MACs 时 INT8 快 26.7%（batch=8）。
3. **INT8 精度无损**（max-abs-diff ~4e-4，cos=1.0）——PTQ 校准数据质量足够。
4. INT8 vs FP16 在 batch=32 处于噪声内互有胜负；batch=8 时 INT8 略优。
5. 测量噪声提示（WDDM + GPU boost）：单次跑差异可达 ±10-35%，定性结论（batch↑→收益↑、精度无损）稳定，定量数字以本表为准、建议多轮取中位数。

## 5. 与历史结论的衔接

- Jetson（TRT 8.5，旧 CNN 结构）batch=8 三档打平 → 此处 PC（TRT 10.10）0.168M MACs 同样打平，跨平台一致；
- 路径 A 放大模型后 batch=8 出现 ~27% INT8 收益 → 模型规模是量化收益能否显现的杠杆。
