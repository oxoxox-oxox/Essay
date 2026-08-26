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

---

# 真机实测（Jetson Orin Nano，2026-08-24）——推翻"batch=1 无收益"

> 实测：wheeltec@10.176.17.161 ｜ TRT 8.5.2.2 / numpy 1.23.5 ｜ 同 1024×1024 模型 ONNX + 同 256 条校准
> 复现：`python3 deploy/bench_trt.py`（engines 存 `deploy/engines/`）

## 1. 三精度 × batch{1,8}（1024×1024 模型）

| batch | FP32 per-sample | FP16 per-sample | INT8 per-sample | INT8 vs FP32 | FP16 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 232.26 us | 193.76 us | 155.55 us | **-33.0%**（更快） | -16.6% |
| 8 | 38.48 us | 18.33 us | 22.42 us | **-41.7%** | **-52.4%** |

## 2. 输出精度对比（64 条随机 obs）

| 精度对 | max-abs-diff | cos |
| --- | --- | --- |
| FP16 vs FP32 | 0.003141 | 1.000000 |
| INT8 vs FP32 | 0.068956 | 0.999954 |

（构建警告：Reshape 的 INT64 权重 cast 到 INT32——无害；FP16 构建有 4 个 subnormal FP16 权重警告。）

## 3. 与 PC（RTX 5070 / TRT 10.10 / WDDM）对比

| 平台 | bs1 INT8 vs FP32 | bs8 INT8 vs FP32 | bs8 FP16 vs FP32 | INT8 max-abs-diff |
| --- | --- | --- | --- | --- |
| PC（TRT 10.10） | **+44%**（更慢） | -26.7% | -17.2% | 0.000419 |
| **Jetson（TRT 8.5）** | **-33%**（更快） | -41.7% | -52.4% | 0.068956 |

## 4. 结论（重要更新）

1. **"batch=1 量化无收益"只在 PC/WDDM 成立；在目标机 Orin 上不成立**——Orin GPU 小、计算相对 launch 开销占比高，INT8/FP16 的 Tensor Core 收益在 batch=1 就能显现（INT8 -33%）。这也说明：**量化收益的平台依赖性本身就是论文结论的一部分**（PC 上 launch-bound，真机上算力受限 → 量化收益真实）。
2. **batch=8：FP16 是最优选择**（-52.4%，精度 cos=1.0 / diff 0.003）；INT8 其次（-41.7%）。batch=1：INT8 最优（-33%）。FP16 vs INT8 随 batch 交叉，工程上可按部署模式选。
3. **v2 纯 GEMM + 路径 A 大模型设计被真机验证**：旧 CNN 结构"bs1 三档打平"的历史结论不再适用于新结构——无 conv cast + 计算量上来后，量化收益在真机上全面显现。
4. **精度注意**：TRT 8.5 的 INT8 量化误差（max-abs 0.069）远大于 TRT 10.10（0.0004），但 cos 仍 0.99995（动作误差 ~3.5% 量级，控制可接受）；当前精度对比用**随机 obs**，部署前建议用**真实 obs**（replay buffer / 校准集）复核，或直接以 INT8 上车导航质量为准（E4）。
5. 单次测量噪声（未锁频）：定性结论（Orin 上量化有收益、FP16/INT8 优于 FP32）稳定，定量数值建议多轮复测。

## 5. 遗留动作

- [ ] 用真实 obs 复核 INT8 精度（当前为随机 obs）
- [ ] 若追求极致：对比 DLA 路径（Orin 的 DLA 支持 INT8/FP16，需显式配置，未启用）
- [ ] E4 上车导航质量测试（INT8+chunk5 vs FP32 基线）
