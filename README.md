# TD3 导航模型量化 + Action Chunking 加速研究（ir-sim 导航环境）

端到端导航模型（纯全连接 TD3）的资源受限部署加速流水线：
**FP32 训练 → 导出纯 FP32 ONNX → 在目标机（Jetson Orin Nano）上用 TensorRT 建引擎**（FP32 / FP16 / 可选 INT8 PTQ），
并通过 **action chunking**（一次输出 N 步、开环执行）降低推理频率。

> 本仓库只保留"训练生成 .pt → 转 ONNX"的代码。INT8 量化由板上 TensorRT 原生 PTQ 完成
> （从纯 FP32 onnx + 真实数据校准），仓库不再包含 QAT/量化训练代码。
>
> **batch 语义（重要）**：
> - **ROS 上车部署** = 实时单步推理，恒为 **batch=1**（`actor_fp32.onnx`，obs [1,113]）。
> - **batch=8**（`actor_fp32_bs8.onnx`）仅用于 trtexec 批量吞吐基准（验证 Tensor Core / 量化收益）。
> - 两者引擎相互独立，**不要把 batch=8 引擎填进 ROS 的 `engine_path`**（forward 尺寸不匹配会停车）。

---

## 研究定位（对齐 plan.md）

`plan.md` 是本研究计划，本仓库是其实现。对应关系：

| plan.md | 本仓库 |
| --- | --- |
| Problem：资源受限机器人推理慢 | 动机 → `deploy/`（板上部署）、`eval/benchmark.py`（延迟指标） |
| Method ① 量化（FP32→INT8，TRT 原生 PTQ） | `train/export_onnx.py` + `train/make_calib_data.py`（PC 侧）+ `deploy/td3_nav/models/build_ptq_engine.py`（板上） |
| Method ② action chunking（N 步开环） | `model/td3.py`、`alg/td3.py`、`env/wrapper.py`、`configs/train.yaml` `chunk.size`、`launch/planner_n5.launch` |
| Experiment：E1–E4 消融 + N=1/5/10 | 见下方「实验矩阵」；命令见 §1 / §2 |
| Metrics：延迟/频率、成功率/轨迹误差、碰撞 | `eval/benchmark.py`、`eval/evaluate.py`、`utils/metrics.py`、`deploy/td3_nav/scripts/bench_engines.sh` |
| Environment / Hardware：静/动态环境 + 目标机 | `configs/world/`（两个世界 YAML）+ Jetson Orin Nano / TRT 8.5 |

---

## ⭐ 当前状态与交接（新会话先读这里）

**项目一句话**：纯全连接 TD3 端到端导航模型，PC 端训练/导出 ONNX，目标机 Jetson Orin Nano（TRT 8.5.2.2）建引擎。
**当前进度：路径 A 已实施——N5 模型从 400×300 扩到 **1024×1024**（Actor 0.169M→1.177M 参数 / 0.168M→1.175M MACs）并重训导出 ONNX（bs1/bs8/bs32）；**核心验证已完成（PC/TRT 10.10，详见 `runs/trt_bench_results.md`）**：batch=1 INT8 仍 launch-bound（-44%），batch=8/32 INT8 每样本快 20-27%（0.168M MACs 时三档打平 → 模型变大后量化收益显现），INT8 精度无损（cos≈1.0）。导航质量：静态 91% vs 旧 87%（提升），动态 75% vs 旧 86%（回落）。E2/E4 的板上 INT8 引擎与上车测试待办。**

**两台机器（务必分清）**

- **本机**（Windows PC，conda env `ir-sim` = `d:\anaconda\envs\ir-sim\python.exe`，torch 2.11 + onnx）：只能训练/导出/生成校准数据，**没有 TRT，跑不了 trtexec**。
- **目标机**：`wheeltec@192.168.0.100`，ROS1 Noetic，Jetson Orin Nano，JetPack R35.6.1，TensorRT 8.5.2.2，catkin 工作区 `~/wheeltec_robot`。板上可 `import tensorrt`。

**当前模型结构（v2，2026-08 起）**

- **纯全连接 TD3（无 CNN）**：整段 obs 直接进 MLP（Linear-ReLU×2 → head），Actor 输出 tanh 归一化 `(B, N, 2)`。**当前隐藏层 1024×1024（路径 A，2026-08 起）**：Actor ~1.177M 参数 / 1.175M MACs；旧 400×300（0.169M）产物保留在 `mlptd3_N5/` 供对比。
- **输入仿照 DRL-robot-navigation-IR-SIM 的 TD3 `prepare_state`**（`env/wrapper.py`）：
  - lidar：分箱取 min 再 `/range_max`（`obs.max_bins`，默认 100 = lidar 束数，即不降维），inf 视为 range_max
  - goal：`[dist / goal_dist_norm, cos, sin]`（`goal_dist_norm` 默认 **10**，对齐 DRL 的 `distance /= 10`）
  - 上一 chunk 动作：逐维 `[lin*2, (ang+1)/2]`（每步 2 维）
  - obs 总维：`max_bins + goal_dim + chunk*action_dim` = N5 时 **113**（N1 时 105）
- **动机**：去掉 1D-CNN 卷积，导出 ONNX 全图为 GEMM，TRT 可整体降 FP16/INT8，不再有"卷积保 FP32 引发的反复 Reformat cast"（旧版根因，见结论 5）。
- 模型文件：`model/td3.py`（纯 MLP TD3，无 CNN）。

**当前产物实况（磁盘核实，2026-08）**

| 目录 | 内容 |
| --- | --- |
| `checkpoints/` | `mlptd3_N1/`（E1 基线）、`mlptd3_N5/`（旧 400×300）、`mlptd3_1024_N5/`（当前路径 A）各含 `model.pt` + `model_best.pt` |
| `runs/` | `mlptd3_N1/`、`mlptd3_N5/`、`mlptd3_1024_N5/`（各仅保留最终成功轮次日志/metrics.csv） |
| `buffers/` | `mlptd3_N1/`（约 353MB）、`mlptd3_N5/`（约 63MB）、`mlptd3_1024_N5/`（约 132MB） |
| `export/` | `mlptd3_N5/`（旧 400×300）与 `mlptd3_1024_N5/`（当前路径 A，4.71MB/个）：`actor_fp32_bs1/bs8/bs32.onnx` + `calib_obs.npz`（均从 `model_best.pt` 导出） |
| `deploy/td3_nav/models/` | 与 export 同步（当前为 1024 模型）：`actor_fp32.onnx`(bs1)、`actor_fp32_bs1/bs8/bs32.onnx`、`calib_obs.npz`、`build_ptq_engine.py` |

**已定论的关键结论（板上实测，历史数据保留）**

> ⚠️ 以下吞吐数据均为**旧 CNN 结构（带 1D-CNN 的 TD3）**下测得，供论文/对比参考；**纯 MLP TD3 结构下的三精度对比待重测**。

1. QAT/QDQ 路径已废弃：torch 导出的 QDQ onnx 让 TRT 8.5 把 BN 拆成独立 FP32 kernel → INT8 反而慢 2.3×（0.373ms / 3475 qps）。
2. TRT 原生 PTQ 正确：纯 `actor_fp32.onnx --int8` → TRT 折叠 BN（逐层 0.154ms / 7312 qps / 0.139ms，N5）。
3. **batch=1 下小模型 launch-bound（三档实测，N5）**：三档几乎持平，FP32 反而略快，量化无延迟收益。
   | 精度 | Throughput(qps) | mean(ms) | p99(ms) |
   | --- | --- | --- | --- |
   | INT8 (PTQ 真实校准) | 7502.98 | 0.1298 | 0.1421 |
   | FP16 | 7436.07 | 0.1300 | 0.1428 |
   | FP32 (noTF32) | 7928.96 | 0.1240 | 0.1396 |
4. 动态 batch 规模-延迟探测（batch 1/8/32/64 扫曲线）已废弃，改用固定 batch=8 做三精度吞吐基准。
5. **batch=8 三档打平：低精度无稳定收益（新引擎重测，N5，旧 CNN 结构）**：
   | 精度 | Throughput(qps) | mean(ms) | p99(ms) | per-sample(ms) |
   | --- | --- | --- | --- | --- |
   | INT8 (PTQ 随机校准测速) | 6852.73 | 0.143111 | 0.161621 | 0.0182 |
   | FP16 | 6745.65 | 0.145276 | 0.161865 | 0.0185 |
   | FP32 (noTF32) | 6900.42 | 0.142863 | 0.14917 | 0.0181 |
   `per-sample = 1000/(thr×batch)`。三档 mean 均 ~0.143ms、每样本 ~0.018ms，INT8/FP16 相对 FP32 差距 <2%（噪声内）。⚠️ 本次未锁频（无 `nvpmodel`/`jetson_clocks`），绝对数值略松，但三档同条件背靠背，相对对比有效。
   🧠 **根因（旧 CNN 结构）**：3 个卷积永远保持 FP32（TF32 kernel），低精度只作用于 FC 块；FC 省下的时间被 FP32↔FP16/INT8 边界插入的 Reformat cast 层抵消（FP16 3 个 ~0.027ms / INT8 2 个 ~0.021ms）。模型小、launch 开销占比高，量化收益无法显现。**→ 这正是 v2 改为纯全连接、去掉 CNN 的原因。**

**下一步（未做）**

1. **N10 checkpoint**：训练 N=10（E3/E4 的另一个 chunk 长度，命令见 §1.1；当前模型 1024×1024）。
2. ~~纯 MLP 三精度吞吐基准~~ **✅ 已完成**（PC/TRT 10.10，结果见 `runs/trt_bench_results.md`）：batch=1 INT8 慢 44%（launch-bound）；batch=8/32 INT8 快 20-27%；400 宽模型三档打平。待补：Jetson 板上复测（TRT 8.5，batch=8 三精度）验证跨平台一致性。
3. **同步 obs_builder.cpp**：板上 C++ 端 obs 构造仍是旧格式（goal `/range_max=7`、prev 原样），与训练（DRL 风格，`/goal_dist_norm=10`、prev `[lin*2,(ang+1)/2]`）不一致，**上车前必须同步**（见 §2.2）。
4. **INT8 + chunk5 导航质量上车测试**（E4，见 §四），与 FP32 基线对比成功率/轨迹误差/碰撞。
5. 定论文叙事：量化 + chunking 双正交手段在 Jetson 上的联合收益；大模型 batch=1 launch-bound 的负面结论同样可写。

---

## 环境要求

**PC（训练 / 导出）**

- Python >= 3.10（conda env `ir-sim`：`d:\anaconda\envs\ir-sim\python.exe`）
- `ir-sim==2.10.1`、`torch>=2.0`，`pip install -r requirements.txt`（需 `onnx`）

**机器人（部署）**

- ROS1 Noetic（`~/wheeltec_robot` catkin 工作区）
- Jetson Orin Nano，JetPack R35.6.1，TensorRT 8.5，CUDA

## 目录结构

```
Essay/
├─ plan.md            # 研究计划（Problem/Method/Experiment/Related work）
├─ 学习笔记_TRT_PTQ量化全流程.md # 个人学习笔记：ONNX → TensorRT PTQ 全流程
├─ configs/           # train.yaml 训练超参（display=false）+ eval.yaml 评估参数（display=true）+ world/ 场景
├─ env/               # ir-sim -> RL 环境封装（观测/动作/奖励）
├─ model/             # TD3（纯全连接 MLP）Actor/Critic（model/td3.py）
├─ alg/               # TD3（chunk 粒度 transition、gamma^N）+ ReplayBuffer
├─ train/             # train.py 训练 + export_onnx.py 导出 ONNX + make_calib_data.py 生成校准数据
├─ eval/              # evaluate/benchmark/run_experiments（PC 端评估与实验矩阵）
├─ deploy/            # ROS1 部署包（catkin 包 td3_nav，板上用）
├─ checkpoints/       # {name}_N{chunk}/model.pt + model_best.pt（mlptd3_N1、mlptd3_N5、mlptd3_1024_N5）
├─ runs/              # 训练日志/指标（mlptd3_N1、mlptd3_N5、mlptd3_1024_N5 最终轮次）
├─ buffers/           # replay buffer（mlptd3_N1、mlptd3_N5、mlptd3_1024_N5）
├─ export/            # ONNX 导出产物（mlptd3_N5、mlptd3_1024_N5：bs1/bs8/bs32 + calib_obs.npz）
└─ utils/             # 配置/checkpoint/日志/指标
```

---

## 实验矩阵（以 plan.md 为准）

方法：TD3（纯全连接）+ **action chunking**（N=1 为基线）。E1–E4 消融 + N 扫描：

| 实验 | Chunk N | 定义（plan.md） | 命令 | 状态 |
| --- | --- | --- | --- | --- |
| E4 | 5/10 | 量化(INT8 PTQ) + action chunking | 训练 + `export_onnx.py` + 板上建 INT8 引擎（§1.1/1.2/2.4） | ⏳ N5 checkpoint/ONNX 已有，INT8 引擎 + 上车测试未做 |
| E3 | 5/10 | action chunking only（FP32） | `train.py --chunk N` + FP32 引擎 | ✅ N5 已训练×2：400×300（静态 87% / 动态 86%）、1024×1024 路径 A（静态 91% / 动态 75%，100 集实测）；N10 缺失 |
| E2 | 1 | 量化 only（INT8 PTQ） | `train.py --chunk 1` + INT8 引擎 | ⏳ checkpoint 已有，INT8 引擎/结果未做 |
| E1 | 1 | 无量化、无 chunking（FP32 基线） | `train.py --chunk 1` | ✅ 已训练（`checkpoints/mlptd3_N1`）；板上 FP32 引擎待建 |
| N 扫描 | 1/5/10 | trade-off 研究 | `eval/run_experiments.py --chunks 1,5,10`（FP32 导航质量） | ⏳ N1/N5 可跑，缺 N10 |

> 量化与否的差异在**板上建引擎**这一步体现：同一 FP32 ONNX → FP32 引擎 = E1/E3，INT8 引擎 = E2/E4。
> 注意 `run_experiments.py` 把 `world_name` 标为 "static"、`eval_world` 标为 "dynamic"，但实际
> `robot_world.yaml`（训练场景）含 4 个 RVO 动态障碍、`eval_world.yaml`（评估场景）全静态——**标签与内容相反，记录实验时以文件内容为准**。

---

## 一、PC 端：训练 / 导出 / 校准数据

### 1.1 训练

```bash
python train/train.py --name mlptd3 --chunk 1          # E1 / E2（N=1 基线，已训练）
python train/train.py --name mlptd3 --chunk 5          # E3 / E4（N5，旧 400×300，已训练）
python train/train.py --name mlptd3_1024 --chunk 5     # E3 / E4（N5，路径 A：1024×1024，已训练）
python train/train.py --name mlptd3_1024 --chunk 10    # N10（当前缺失）
```

- 模型为**纯全连接 TD3**（`model.encoder_type: "mlp"`），无 CNN；隐藏层宽取 `configs/train.yaml` 的 `model.hidden1/hidden2`（当前路径 A：1024×1024）。
- 输入仿 DRL（`obs.max_bins`、`obs.goal_dist_norm`、prev 动作逐维 `[lin*2,(ang+1)/2]`），见「观测 / 动作」一节。
- 产物：`checkpoints/{name}_N{chunk}/model.pt`（含 actor/critic/元数据，评估成功率更高时另存 `model_best.pt`）、`runs/`、`buffers/`（`save_buffer: true` 时）。
- 轮次制训练参数见 `configs/train.yaml` 的 `train:` 段；`--parallel-workers N` 启用多进程并行环境。

### 1.2 导出纯 FP32 ONNX（固定 batch=8）

```bash
python train/export_onnx.py --name mlptd3 --chunk 5
```

- 产物：`export/mlptd3_N5/actor_fp32_bs8.onnx`（无 QDQ，obs [8,113]）。
- **全图仅 GEMM/ReLU/Tanh（无 Conv、无 BN）**：`onnx` 解析确认 ops 只有 `Constant/Gemm/Relu/Reshape/Tanh`。
- **自动同步**：导出后自动 `copy2` 一份到 `deploy/td3_nav/models/`，无需手动复制，直接随整包 scp。
- `--batch N` 可覆盖固定 batch（默认 8），产物 `actor_fp32_bsN.onnx`。

### 1.3 生成 INT8 校准数据

```bash
python train/make_calib_data.py --name mlptd3 --chunk 5 --num-samples 256
```

- 产物：`export/mlptd3_N5/calib_obs.npz`，并**自动同步**到 `deploy/td3_nav/models/`。
- 数据：从训练 replay buffer 采样真实 obs，维度按 checkpoint meta 自动取（N5 为 113）。
- 前置依赖：对应 `checkpoints/{name}_N{chunk}/model.pt` 与 `buffers/{name}_N{chunk}/buffer.npz` 必须存在。

### 1.4 PC 端评估 / 基准

> 评估脚本默认加载 **`configs/eval.yaml`**（`env.display: true`，可视化）；
> 训练用 `configs/train.yaml`（`env.display: false`，headless 加速）。两者已分离，`--config` 可覆盖。

```bash
python eval/evaluate.py --checkpoint checkpoints/mlptd3_N1/model.pt      # 导航评估（成功率/轨迹误差/碰撞，可视化）
python eval/benchmark.py --checkpoint checkpoints/mlptd3_N1/model.pt     # 推理延迟（PC torch 参考）
python eval/run_experiments.py --name mlptd3 --chunks 1,5,10             # 实验矩阵 -> runs/summary.csv
```

---

## 二、板上部署：同步 / 建引擎 / 编译

### 2.1 部署包结构

```
deploy/td3_nav/
├─ CMakeLists.txt / package.xml
├─ src/planner_node.cpp     # obs 构造 + TensorRT 推理 + /cmd_vel_planner（0.3s 节流）
├─ src/safety_node.cpp      # 近距急停/电量/充电/看门狗/限幅 -> 真机 /cmd_vel
├─ src/tensorrt_engine.cpp  # TRT 引擎封装（load/forward/CUDA buffer）
├─ src/obs_builder.cpp      # obs 构造（⚠️ 仍为旧格式，需与训练侧同步，见 2.2）
├─ include/td3_nav/*.hpp
├─ config/params.yaml       # 全部可调参数（goal、reverse_scan、速度限幅、chunk_size 等）
├─ launch/planner.launch    # N1 部署（chunk_size=1）
├─ launch/planner_n5.launch # N5 部署（chunk_size=5，开环 5 步）
├─ models/actor_fp32.onnx      # ROS 部署用纯 FP32 源（batch=1）
├─ models/actor_fp32_bs8.onnx  # 基准用纯 FP32 源（batch=8）
├─ models/build_ptq_engine.py  # INT8 PTQ 校准构建（默认 batch=8，导航需 --batch-size 1）
├─ models/calib_obs.npz        # INT8 校准数据
└─ scripts/                    # verify_scan_order.py / obs_reference.py / bench_engines.sh / dump_engines.sh
```

> ⚠️ **obs 契约待同步**：训练/导出的输入已改为 DRL 风格（goal 距离 `/goal_dist_norm=10`、prev 动作逐维 `[lin*2,(ang+1)/2]`、lidar 分箱取 min），但 `src/obs_builder.cpp` 仍为旧格式（goal `/range_max=7`、prev 原样 [-1,1]）。**上车前必须同步 obs_builder.cpp**，否则 obs 数值与训练不一致。

### 2.2 数据契约（obs / action 精确格式）

- 输入 `obs [1, 100+3+2N]`（N = `planner/chunk_size`，N1 为 105，N5 为 113）：
  - `[0:100]` lidar：前向±90°按训练角度（`linspace(-π/2, π/2, 100)`）重采样 100 束；DRL 式分箱 `bin_size=ceil(100/max_bins)` 每箱取 min，再 `/range_max`（max_bins=100 时逐束），inf/超量程 → 1.0
  - `[100:103]` goal 极坐标：`[dist / goal_dist_norm, cos(a), sin(a)]`，`a = wrap(atan2(goal_y-y, goal_x-x) - yaw)`，`goal_dist_norm=10`
  - `[103:100+3+2N]` 上一决策点实际下发的**整段 chunk 动作**，逐维归一化：每步 `[lin*2, (ang+1)/2]`（启动置 0）
- 输出 `action [1,N,2]`：N 步 `[lin, ang]` 归一化 [-1,1]，按 `vel_min + (a+1)/2*(vel_max-vel_min)` 映射后限幅；chunking 下每 0.3s 开环执行一步，N 步执行完再推理
- 每束角度用 tf2 实时查 `laser→base_link`（该车为绕 z 180°，兜底参数 `laser_yaw_fallback_deg=180`）

### 2.3 同步包到板上（PC 端执行）

```bash
scp -r deploy/td3_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
```

> 会自动带上最新的 `actor_fp32_bs8.onnx`、`calib_obs.npz`、`src/*.cpp` 源码等。

### 2.4 建引擎（Jetson 本机，batch=1，ROS 用）

```bash
cd ~/wheeltec_robot/src/td3_nav/models
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32.onnx --noTF32 --saveEngine=actor_fp32.engine   # FP32 基线（--noTF32 才是真 FP32）
python3 build_ptq_engine.py --onnx actor_fp32.onnx --calib calib_obs.npz --out actor_int8.engine --batch-size 1 --cache calib.cache   # INT8 真实校准
# 可选 FP16 引擎
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32.onnx --fp16 --saveEngine=actor_fp16.engine
```

> ⚠️ **`--batch-size 1` 不能省**：`build_ptq_engine.py` 默认是 8（基准用）。导航用 batch=1 引擎，漏写会产出 batch=8 引擎，ROS forward 尺寸不匹配 → 报 `engine forward failed`。
> `.engine` 与 TRT 版本 + GPU 架构绑定，必须在板子上构建；仓库不提供现成 `.engine`。
> `trtexec --int8` 是随机校准，**只能测速**，正式部署必须用上面的真实数据校准。

### 2.5 编译（Jetson 本机，含全部踩坑处理）

```bash
# ① 源码时间戳改为板上当前时间（防 Clock skew 导致跳过编译，见坑 A）
cd ~/wheeltec_robot/src/td3_nav
find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +

# ② 清掉 td3_nav 旧编译产物，白名单单独构建（见坑 B）
cd ~/wheeltec_robot
rm -rf build/td3_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="td3_nav" -j$(nproc)

# ③ 确认真正重编 + source + 清空白名单（见坑 A）
ls -l --time-style=full-iso devel/lib/td3_nav/planner_node   # mtime 应为刚才
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""                       # 清空恢复整工作区
chmod +x src/td3_nav/scripts/*.py
```

**编译成功判据**：`catkin_make` 输出里**必须出现 `Building CXX object .../planner_node.dir/...`** 行。若只有 `Built target` + `Clock skew detected` 而没有 Building 行 = 没重编，用的是旧二进制。编译中大量 `deprecated` 警告（`destroy()`、`enqueueV2` 等）是正常的，**不影响成功**，只要无 `error:`。

**板上二进制不同步的典型症状**：引擎加载显示 `input 113 elements`，但节点警告 `engine input=113 (expect 105)` 并持续 `engine forward failed`。`expect 105` = N1 逻辑（chunk_size=1），说明二进制旧。先 `grep -n "chunk_size" src/td3_nav/src/planner_node.cpp` 确认源码已同步，再按本节重编。

### 2.6 真机参数核对结论

- `/scan`：360°、约 1667 束、`range_max=100m`、frame `laser`，~12Hz（前向窗口截取 + 降采样到 100 束）
- `laser→base_link`：平移 [0.101, 0, 0.168]，绕 z 旋转 **180°**（tf2 自动处理）
- `/odom`：`nav_msgs/Odometry`，frame `odom_combined`，直接取 (x,y,yaw)
- `/lslidar_order`：驱动**订阅**的输入话题（`std_msgs/Int8`），可控制扫描方向；代码以 `reverse_scan` 开关 + 现场校验兜底
- 无 `/map`/amcl → goal 定义在 `odom_combined` 系（相对起点）
- 安全信号：`/PowerVoltage`(Float32)、`/robot_charging_flag`(Bool)、`/robot_recharge_flag`(Int8)、`/robot_red_flag`(UInt8)，safety 节点已接入
- 底盘限速在固件写死（`rosparam` 无速度参数），`max_linear/max_angular` 作软件限幅

---

## 三、启动与验证

### 3.1 启动 N5（engine_path 用绝对路径）

```bash
roslaunch td3_nav planner_n5.launch engine_path:=/home/wheeltec/wheeltec_robot/src/td3_nav/models/actor_int8.engine
```

- ⚠️ **不要用 `engine_path:=$(find td3_nav)/...`**：bash 会先把 `$()` 展开成 `find` 命令的输出（一堆路径），报 `The following input files do not exist`。roslaunch 的 `$(find pkg)` 只在 launch 文件内部生效。用**绝对路径**或**单引号**：`'engine_path:=$(find td3_nav)/models/actor_int8.engine'`。
- `planner_n5.launch` 设 `chunk_size=5`：obs [1,113]、一次推理开环执行 5 步（每步 0.3s）。
- 换 FP32 基线：`engine_path:=/home/wheeltec/wheeltec_robot/src/td3_nav/models/actor_fp32.engine`，其余不变，保证单一变量。
- 若启动时报 launch 语法错误 `not well-formed (invalid token)`：是 launch 注释里有 `--`（双连字符），见坑 C。

### 3.2 日志判读

**正常**：
```
[TRT] engine loaded: input 113 elements, output 10 elements
[planner] ready. goal=(5.00, 5.00), period=0.30s
# 无 expect 警告、无 engine forward failed
```

**异常 `expect 105` + `engine forward failed`**：板上二进制旧（chunk_size 未生效）→ 回 2.5 重编。
**异常 `engine forward failed` 但无 expect 警告**：引擎 batch 填错了（用了 batch=8 引擎）→ 用 2.4 的命令重建 batch=1 引擎。

### 3.3 上车验证顺序

1. 启动后先不松手刹：确认 engine 加载成功、`/cmd_vel_planner` 有输出、`rostopic echo /cmd_vel` 收到 safety 转发
2. 数值比对：`debug_log=true` 打点 obs/action，与 PC 上 torch 输出做 max abs diff（应 < 0.05）
3. bag 离线比对：`rosbag record /scan /odom` → PC 上 `obs_reference.py` 复算 obs 与 ROS 端对比
4. 空场低速实测 → 逐级加障碍

### 3.4 现场校验 scan 方向

```bash
rosrun td3_nav verify_scan_order.py
```

决定 `reverse_scan`（障碍物放车头一侧，看最近束 base 角应≈±90°）。

---

## 四、INT8 + action chunk=5 导航质量测试

目标：测 INT8 引擎下 chunk=5 的导航效果（成功率/轨迹误差/碰撞），与 FP32 基线对比。用 **batch=1** 引擎（ROS 实时单步推理），与 batch=8 吞吐基准互不影响。**前置：需先补齐 N5 checkpoint（§1.1）并完成导出/校准（§1.2/1.3）。**

**Step 1（PC）：训练 N5 + 导出 + 生成校准** → 见 1.1 / 1.2 / 1.3（产物已自动同步到 `deploy/td3_nav/models/`）。

**Step 2（PC）：同步包** → 见 2.3。

**Step 3（板上）：建 batch=1 引擎** → 见 2.4（FP32 基线 + INT8 真实校准）。

**Step 4（板上）：编译** → 见 2.5（务必确认出现 `Building CXX object`）。

**Step 5（板上）：启动 + 验证** → 见 3.1 / 3.2 / 3.3，记录数据。

```bash
# 完整启动（INT8）：
roslaunch td3_nav planner_n5.launch engine_path:=/home/wheeltec/wheeltec_robot/src/td3_nav/models/actor_int8.engine
# 对比 FP32 基线（改 engine_path）：
roslaunch td3_nav planner_n5.launch engine_path:=/home/wheeltec/wheeltec_robot/src/td3_nav/models/actor_fp32.engine
```

**Step 6（板上）：清理 models 残留**（非必需，保持干净）：

```bash
cd ~/wheeltec_robot/src/td3_nav/models
rm -f actor_fp32_dynbs.onnx actor_qdq.onnx actor_fp32_b8.engine.build.log
rm -f actor_fp32_bs8.engine            # batch=8 引擎仅在基准时生成/保留
```

**结论判读**：INT8 引擎的 obs 构造、chunk 开环执行与 FP32 完全同一份代码，导航质量差异仅来自 INT8 精度损失；若成功率和轨迹误差与 FP32 基线相当（或退化可接受），则 INT8+chunk5 可作部署方案。

---

## 五、batch=8 吞吐基准（实验用途，非上车）

用固定 batch=8 的引擎做三精度吞吐对比（验证 Tensor Core / 量化收益）。

**Step 1：PC 端生成校准数据（若未做）** → 见 1.3。

**Step 2：板上建 batch=8 INT8 引擎（真实数据校准）**

```bash
cd ~/wheeltec_robot/src/td3_nav/models
python3 build_ptq_engine.py --onnx actor_fp32_bs8.onnx --calib calib_obs.npz --out actor_int8_bs8.engine --batch-size 8 --cache calib.cache
```

**Step 3：跑三档对比**

> ⚠️ 必须 `cd models/` 再跑：`bench_engines.sh` 的引擎名是**相对当前目录**解析的，在别处运行会复用/重建 CWD 下的旧引擎（曾有 ~3445 qps 的假慢结果）。首次运行前清掉 home 残留：`rm -f ~/actor_*.engine`。

```bash
cd ~/wheeltec_robot/src/td3_nav/models
sudo nvpmodel -m 0 && sudo jetson_clocks
bash ../scripts/bench_engines.sh
```

`bench_engines.sh`：INT8 复用 Step 2 的 `actor_int8_bs8.engine`（不存在则 `trtexec --int8` 随机校准重建，仅测速）、构建并基准 FP16/FP32，末尾打印三档汇总表（含每样本延迟 = 1000/(thr×8)）。任一精度每样本延迟低于 FP32 即"量化/低精度在 batch=8 下降低延迟"成立。

**历史实测结果（板上，N5，2026-08，旧 CNN 结构）**：见"当前状态"结论 5。要点：三档（INT8/FP16/FP32）基本打平（每样本 ~0.018ms），低精度无稳定收益；FP16 此前"反常偏慢"与 INT8"反超 4%"均为旧引擎假象，新引擎重测已结案。**纯 MLP TD3 结构下的结果待重测。**

---

## 部署踩坑速查表

- **坑 A · Clock skew 跳过编译**：`scp` 带 PC 端 mtime（未来时间），板上 `catkin_make` 报 `Clock skew detected` 且**跳过实际编译**（无 `Building CXX object` 行），二进制仍旧。处理：`touch` 源码 + `rm -rf build/td3_nav` + 重编（见 2.5）。
- **坑 B · CATKIN_WHITELIST_PACKAGES**：若 `catkin_make` 提示 `Using CATKIN_WHITELIST_PACKAGES: <别的包>`，用 `-DCATKIN_WHITELIST_PACKAGES="td3_nav"` 单独构建，之后用 `""` 清空恢复整工作区（见 2.5）。
- **坑 C · launch XML 注释双连字符**：XML 注释内不能出现 `--`，如 `--int8` 会让 `roslaunch` 报 `not well-formed (invalid token): line N, column M`。改 launch 注释时避免双连字符。
- **坑 D · 命令行 `$(find)` 被 bash 展开**：`engine_path:=$(find ...)` 在 bash 里被展开成 find 输出 → `The following input files do not exist`。用绝对路径或单引号（见 3.1）。
- **坑 E · build_ptq_engine.py batch-size 语义**：默认 8（基准用）；导航用 batch=1 引擎**必须显式 `--batch-size 1`**，漏写导致 ROS forward 失败。
- **坑 F · 二进制不同步**：`engine input=113 (expect 105)` + `engine forward failed` = 板上二进制旧（chunk_size 未生效），非模型/引擎问题；`grep -n "chunk_size" src/planner_node.cpp` 确认源码同步后重编（见 2.5）。
- **坑 G · INT8 校准器（已修）**：`build_ptq_engine.py` 曾报 `Unable to cast Python instance to C++ type` + `illegal memory access`。根因（TRT 8.x pybind）：
  1. `get_batch` 必须返回**每个输入 device 内存指针的整型列表**（`std::vector<size_t>`），不能返回 numpy array 或 host 指针。
  2. 用 ctypes `cudaMalloc` 预分配 batch buffer，`get_batch` 里 `cudaMemcpy(H2D)` 后返回 device 指针。
  3. 基类用 `trt.IInt8EntropyCalibrator2`（`trt.ICalibrator` 未导出）。
  4. 校准器对象需保留 Python 强引用到构建结束。
  若仍报错：`python3 -c "import tensorrt; print(tensorrt.__version__); print(hasattr(tensorrt,'IInt8EntropyCalibrator2'))"`。
- **坑 H · CMake 编译**：CMakeLists 用 `-std=gnu++14`（`-std=c++14` 会隐藏 `M_PI` 编译失败）；`createInferRuntime` 需 `nvinfer1::` 前缀。`deprecated` 警告不影响成功。

---

## 观测 / 动作

- 观测（输入仿 DRL-robot-navigation-IR-SIM 的 TD3 `prepare_state`，见 `env/wrapper.py`）：
  - lidar：分箱取 min 再 `/range_max`（`obs.max_bins`，默认 = lidar 束数）
  - goal 极坐标：`[dist / goal_dist_norm, cos(a), sin(a)]`（3 维，`goal_dist_norm` 默认 10）
  - 可选（`obs.include_prev_action: true`）：再拼接**上一 chunk 动作**（`N*action_dim` 维），逐维归一化每步 `[lin*2, (ang+1)/2]`——每个决策点编码"刚执行的那一个动作 chunk"（episode 起点全零）
- 动作：归一化 `[-1,1]`，映射到真实速度范围（diff 机器人 `[linear_vel, angular_vel]`）
- Chunk：Actor 一次输出 `(N, action_dim)`；TD3 以 chunk 为粒度，TD 折扣为 `gamma^N`

## 替换环境

把你的 ir-sim 世界 YAML 放入 `configs/world/`，**训练与评估分别修改**两个配置文件：

```yaml
# configs/train.yaml（训练场景，display 保持 false）
env:
  world_name: "configs/world/your_world.yaml"      # 训练场景
  eval_world: "configs/world/your_eval_world.yaml" # 训练期间的内部评估场景
```

```yaml
# configs/eval.yaml（评估场景，display 保持 true）
env:
  world_name: "configs/world/your_world.yaml"      # run_experiments 的 "static" 场景
  eval_world: "configs/world/your_eval_world.yaml" # evaluate.py 默认场景 / run_experiments 的 "dynamic" 场景
```

要求：机器人带 `lidar2d` 传感器（beam 数与 `obs.lidar_range_max` 一致）、有 `goal`。
