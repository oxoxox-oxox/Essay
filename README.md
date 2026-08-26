# PPO 端到端导航模型：INT8 量化 + Action Chunking 加速研究（ir-sim 导航环境）

论文方向：机器人端到端 RL 导航因推理耗时产生决策延时。提出两种正交手段：

- **INT8 量化**：把模型量化成 INT8，提升推理速度、降低单步决策延时；
- **action chunking**：一次推理输出 N 步、开环执行，降低推理频率、防止碰撞。

> 当前算法基线为 **SB3 PPO**（单步 N=1）。TD3 路线经实测确认不可行（训练不稳定、效果差），
> 相关代码与产物已从仓库移除；历史实测数据保留在 `runs/trt_bench_results.md`、
> `runs/latency_diagnosis_results.md`，供论文对比参考。

---

## 当前状态（2026-08）

| 项 | 状态 |
| --- | --- |
| TD3 基线 | ❌ 已放弃（代码/模型/产物已清理） |
| PPO 训练（ir-sim） | ✅ **当前主模型 `checkpoints/ppo_mw_N1/`**（多世界重训 v2）；旧版 `ppo_N1/`、`ppo_smoke_N1/` 保留 |
| PPO 仿真评估 | ✅ `src/eval_ppo.py --checkpoint ...`（成功率/碰撞率/平均步数） |
| PPO 多世界重训（v2） | ✅ `checkpoints/ppo_mw_N1/`（6 世界 + goal shaping；**空场绕圈已修复**：直行 3m 到达、max\|yaw\| 0.05rad） |
| PPO 策略演示（ir-sim） | ✅ `demos/ppo_irsim_demo.py`（Gazebo 训练策略的 ir-sim 演示，模型在 `demos/data/`，实测 20/20 成功、0 碰撞、平均 105 步） |
| PPO → ONNX 导出 | ✅ `export/ppo_mw/`（actor_fp32_bs1/bs8.onnx，opset 17 纯 Gemm/Tanh + calib_obs.npz 校准数据） |
| PPO 真机导航测试 | ⏳ 未做 |
| PPO 的 action chunk 效果 | ⏳ 未做 |
| PPO 的 INT8 量化推理速度 | ✅ 真机 bs1 INT8 延迟 **-47%**（Orin/TRT 8.5，见 `runs/ppo_N1/trt_bench_results.md`）；输出精度对比/导航质量待做 |

冒烟结果（20k 步）：rollout 成功率 0→0.76，最终评估 10 集 60% 成功（σ 未收敛，需完整训练）。

---

## 环境要求

**PC（训练 / 评估 / 演示）**

- Python >= 3.10，conda env `rl_env`（已装 stable-baselines3 2.9.0 / gymnasium 1.3.0 / irsim 2.10.0）
- 或 `pip install -r requirements.txt`（含 stable-baselines3、gymnasium）

**机器人（部署）**

- ROS1 Noetic + Jetson Orin Nano（JetPack R35.6.1，TensorRT 8.5，CUDA）
- 部署包：`deploy/ppo_nav/`（PPO 契约重建版，见 `deploy/README.md`）

## 目录结构

```
Essay/
├─ guide.md             # 当前思路与下一步计划（本仓库的指导文档，优先阅读）
├─ configs/             # train.yaml（训练/评估总配置）+ world/ 场景（训练/评估世界 YAML）
├─ env/                 # ir-sim 环境封装：wrapper.py（IrSimEnv）、reward.py、ppo_gym.py（SB3 gymnasium 适配）
├─ src/                 # train_ppo.py：SB3 PPO 单步训练（多世界）；eval_ppo.py：独立评估
├─ train/               # unpack_ppo_actor.py / export_ppo_onnx.py（ONNX 导出两步）
├─ demos/               # PPO 策略的 ir-sim 演示（纯 torch 前向，模型文件在 demos/data/）
├─ deploy/              # 真机部署：ppo_nav ROS 包（planner/safety/obs/TRT 封装）+ 部署文档
├─ checkpoints/         # PPO 模型：当前 ppo_mw_N1/{best_model,ppo_model}.zip；旧 ppo_N1/、ppo_smoke_N1/
├─ export/              # ONNX 导出产物：ppo_mw/（当前）与 ppo_N1/（历史）
├─ doc/                 # 分析文档：PPO 模型结构分析（md + 复现脚本 + 原始输出）
├─ runs/                # PPO 训练日志（ppo_*）+ TD3 历史实测记录（*.md，存档供论文参考）
├─ 学习笔记_TRT_PTQ量化全流程.md  # ONNX → TensorRT PTQ 全流程学习笔记（通用知识）
└─ utils/               # config.py：配置加载/深合并/路径解析
```

## PPO 训练与评估

模型：SB3 PPO（MlpPolicy，隐藏层 1024×1024，由 `configs/train.yaml` 的 `model.hidden1/2` 控制）。
观测 105 维（100 lidar + goal 3 + 上一动作 2），动作 2 维连续速度（归一化 [-1,1]，
由 `IrSimEnv.scale_action` 映射到真实速度）。

```bash
# 完整训练（GPU，多世界）：产物 checkpoints/{name}_N1/{best_model,ppo_model}.zip + runs/{name}_N1/
d:\anaconda\envs\rl_env\python.exe src\train_ppo.py --steps 300000 --device cuda --name ppo_mw
# 冒烟测试（CPU，几分钟）
d:\anaconda\envs\rl_env\python.exe src\train_ppo.py --steps 20000 --device cpu --name ppo_mw_smoke
# 指定世界列表（覆盖 config 的 env.worlds；不传则用 config）
d:\anaconda\envs\rl_env\python.exe src\train_ppo.py --steps 300000 --device cuda --worlds configs/world/open_field.yaml,configs/world/sparse_obs.yaml
# 评估（独立脚本 src/eval_ppo.py，无头 20 集）
d:\anaconda\envs\rl_env\python.exe src\eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip
# 评估+可视化（弹 matplotlib 窗口）
d:\anaconda\envs\rl_env\python.exe src\eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip --episodes 3 --display
# 换评估世界（默认 eval_world；如要评估训练世界 robot_world.yaml 用 --world）
d:\anaconda\envs\rl_env\python.exe src\eval_ppo.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip --world configs/world/robot_world.yaml
```

训练用 `configs/train.yaml` 的 env/obs/reward 段。**多世界训练**（默认开启，`env.worlds` 列表）：
每次 reset 随机换一个世界（单活跃场景，规避 ir-sim 多实例冲突），世界包括
`robot_world`（动态杂乱）、`open_field`（空场，修复绕圈）、`sparse_obs`、`corridor`、`cluttered`、`dynamic`；
每项可覆盖 start_range/fixed_goal。reward 已加 **goal shaping**（`goal_angle_coef`/`goal_dist_coef`）。
详细用法见脚本 docstring。

## PPO → ONNX 导出与真机 INT8 准备

模型导出分两步（SB3 与 onnx 分属不同 conda env，无法在单 env 内同时满足）：

```bash
# ① 解包 Actor 权重（rl_env，有 SB3；默认取当前主模型 ppo_mw）
d:\anaconda\envs\rl_env\python.exe train\unpack_ppo_actor.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip --name ppo_mw
# ② 导出 ONNX + 采集 INT8 校准数据（ir-sim env，有 onnx；默认 batch 1,8）
d:\anaconda\envs\ir-sim\python.exe train\export_ppo_onnx.py --actor export/ppo_mw/policy_actor.pt --make-calib
```

产物（`export/ppo_mw/`）：

- `actor_fp32_bs1.onnx` / `actor_fp32_bs8.onnx`：纯 FP32 图（**仅 Gemm/Tanh，opset 17**），输入 `obs [B,105]`，输出 `action [B,2]`（μ，机器人侧需 `clip` 到 [-1,1] 再映射速度）
- `calib_obs.npz`：256 条真实 obs（策略在训练世界 rollout 采集），INT8 PTQ 校准用
- `build_ptq_engine.py`：板上建 INT8 引擎脚本（真实数据校准）
- bs1 = ROS 实时单步推理；bs8 = 板上吞吐基准（**勿**把 bs8 引擎用于导航）

**板上量化（Jetson Orin Nano / TRT 8.5，目标机操作）**：

```bash
scp -r export/ppo_mw/ wheeltec@<ip>:~/ppo_deploy/
cd ~/ppo_deploy
# FP32 基线（导航，batch=1，--noTF32 才是真 FP32）
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32.engine
# INT8 真实数据校准（导航引擎必须 --batch-size 1，漏写会产出 bs8 引擎）
python3 build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8.engine --batch-size 1
# 吞吐基准（batch=8，可选）：trtexec --int8 是随机校准，只能测速
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs8.onnx --int8 --saveEngine=actor_int8_bs8.engine
```

> `.engine` 与 TRT 版本 + GPU 架构绑定，必须在板子上构建。
> 动作输出为 μ（squash_output=False），板上 obs 构造需按 105 维契约（lidar 分箱/range_max、goal [dist/10,cos,sin]、上一动作 [lin*2,(ang+1)/2]）实现。

## 观测 / 动作契约（env/wrapper.py 的 IrSimEnv）

- 观测（仿 DRL-robot-navigation-IR-SIM 的 prepare_state）：
  - lidar：分箱取 min 再 / range_max（`obs.max_bins`，默认 = lidar 束数）
  - goal 极坐标：`[dist / goal_dist_norm, cos, sin]`（3 维，`goal_dist_norm` 默认 10）
  - 上一动作（`obs.include_prev_action: true`）：`N*action_dim` 维，逐维归一化每步 `[lin*2, (ang+1)/2]`
    ⚠️ 当前 PPO 模型训练走 `step_single`，该通道**恒为 0**（已实测验证），部署端 obs_builder 默认输出 0 保持一致
- 动作：归一化 `[-1,1]`，映射到真实速度范围（diff 机器人 `[linear_vel, angular_vel]`）
- 奖励：到达+100 / 碰撞-100 / 每步 `lin - 0.5|ang|` / 贴近惩罚 / **+0.5·cos(目标夹角) + 1.0·每米靠近（goal shaping，修复空场绕圈）**
- 终止：到达 / 碰撞 / 超时（`max_steps`）

## 替换环境

把 ir-sim 世界 YAML 放入 `configs/world/`，修改 `configs/train.yaml` 的 `env.world_name`（训练场景）
与 `env.eval_world`（评估场景）。要求：机器人带 `lidar2d` 传感器（beam 数与 `obs.lidar_range_max` 一致）、有 `goal`。

## 下一步（见 guide.md）

1. ~~分析 PPO 模型结构~~ ✅（`doc/PPO模型结构分析.md`）
2. **真机 INT8 收尾**：锁频复测取中位数、FP32 基线确认 `--noTF32`、**输出精度对比**（INT8 vs FP32 max-abs-diff/余弦）、bs8 吞吐基准；
3. **真机导航测试**：`deploy/ppo_nav` 上车（编译 → 启动 → obs 比对 → 低速试跑），FP32 vs INT8 成功率/碰撞对比；
4. **PPO action chunking**：为 PPO 实现一次输出 N 步、开环执行，验证防碰撞与降频效果。

> 历史 TD3 流水线（训练/导出 ONNX/板上 TRT 三精度验证）已移除；`runs/trt_bench_results.md`
> 保留的 TD3 真机三精度数据（如 Orin 上 batch=1 INT8 -33% 等结论）可作 PPO 量化实验的参考。
