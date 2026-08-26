# PPO 模型结构分析

> 目的（对应 guide.md「下一步」）：分析仓库内已训练 PPO 模型的网络结构，为真机 INT8 量化与 action chunking 做准备。
> 所有结构信息均由**实际加载模型**得到（SB3 2.9.0 / torch 2.11.0），复现脚本见 `doc/analyze_ppo_structure.py`，
> 原始输出见 `doc/ppo_structure_report.txt`。

---

## 1. 仓库内 PPO 模型清单

| 路径 | 来源 | obs 维度 | 动作空间 | net_arch | 策略总参数量 | 训练步数 (num_timesteps) |
| --- | --- | --- | --- | --- | --- | --- |
| `checkpoints/ppo_N1/best_model.zip` | ir-sim 完整训练（EvalCallback 最优） | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 240,000 |
| `checkpoints/ppo_N1/ppo_model.zip` | ir-sim 完整训练（最终模型） | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 300,032 |
| `checkpoints/ppo_smoke_N1/best_model.zip` | ir-sim 冒烟测试 | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 10,000 |
| `checkpoints/ppo_smoke_N1/ppo_model.zip` | ir-sim 冒烟测试 | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 20,480 |

论文/真机部署相关的模型是 **`checkpoints/ppo_mw_N1/`**（多世界重训 v2，结构与 ppo_N1 完全一致，仅权重不同）；
旧模型 `checkpoints/ppo_N1/` 保留作对比。

> 注：原 amr_rl_ws（Gazebo/ROS2）训练工作区已从仓库移除（GitHub: `wyh010731/amr_rl_ws`），
> 其策略解包产物保留在 `demos/data/policy_weights.pt`（62 维 obs 契约，见 §3.3）。

---

## 2. 主模型结构（checkpoints/ppo_mw_N1 / ppo_N1，ir-sim 环境，架构相同）

通用属性（4 个 zip 完全一致）：

- 算法：SB3 PPO，`MlpPolicy`（即 `ActorCriticPolicy`），连续动作高斯分布（DiagGaussian）
- 激活函数：**Tanh**（SB3 MlpPolicy 默认）
- `squash_output=False`：动作输出**没有 tanh 压缩**，确定性预测 = `clip(μ, [-1, 1])`
- `share_features_extractor=True`：对 MlpPolicy 只是恒等的 FlattenExtractor，**不产生实际权重共享**

### 2.1 网络架构（实际加载打印）

```
obs (105,) ──┬─> Actor(策略)：                        参数量
             │    Linear(105 -> 1024) + Tanh        108,544
             │    Linear(1024 -> 1024) + Tanh     1,049,600
             │    Linear(1024 -> 2)                  2,050
             │    └─ μ (2,)；另加 log_std 2 个可学习参数
             └─> Critic(价值)：
                  Linear(105 -> 1024) + Tanh        108,544
                  Linear(1024 -> 1024) + Tanh     1,049,600
                  Linear(1024 -> 1)                  1,025
                  └─ V (1,)
```

要点：

- SB3 中 `net_arch=[1024,1024]` 等价于 `dict(pi=[1024,1024], vf=[1024,1024])`：Actor 与 Critic 是**两份独立 MLP，权重不共享**
- Actor 参数量（不含 log_std）：1,160,194 ≈ **1.16M**；Critic：1,159,169；策略总参数量：**2,319,365 ≈ 2.32M**
- 单次前向 MACs：105×1024 + 1024×1024 + 1024×2 = **1,158,144 ≈ 1.16M**（全图纯 GEMM）

### 2.2 超参数（zip 内保存值）

| 超参数 | 值 |
| --- | --- |
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| n_steps | 1024 |
| batch_size | 128 |
| n_epochs | 10 |
| clip_range | 0.2（恒定） |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

前向检查：`obs (1,105) -> dist mean (1,2)，value (1,1)`。

---

## 3. 对后续工作的影响

### 3.1 INT8 量化（对应 guide 下一步 ②）

- 导出 ONNX 只需 **Actor** 部分：`mlp_extractor.policy_net` + `action_net`；输入 `[B,105]` FP32，输出 `μ [B,2]`
- 全图只有 `Gemm` + `Tanh`（观测归一化在 obs 构造侧完成，不进网络），**无卷积、无 BN** → TensorRT 可整体 INT8，
  与 TD3 时代“纯 MLP 可全图降精度、避免 Reformat cast”的结论一致
- 确定性动作 = `clip(μ, [-1,1])`，量化误差先作用在 μ 上；μ 的数值范围对 INT8 表示友好
- **已落地**：`train/unpack_ppo_actor.py`（SB3 zip → 纯权重）+ `train/export_ppo_onnx.py`（→ ONNX + 校准数据）；
  实测导出图 ops 仅 `[Gemm, Tanh]`（opset 17），onnxruntime 与 torch 输出 max-abs-diff ≈ 7e-7；
  板上建 INT8 引擎用 `export/ppo_mw/build_ptq_engine.py`（真实 obs 校准）

### 3.2 action chunking（对应 guide 下一步 ③）

- 当前 PPO 是**单步策略**：一次推理只输出 `(B,2)`，不能直接开环 N 步
- 可选改造方向（均未实现，需另行设计）：
  1. 训练侧：动作空间改为 `(B, 2N)`（网络输出 N 步），重新训练，obs 的“上一 chunk 动作”维度相应调整；
  2. 部署侧：保持单步策略不变，部署层做“一次决策 N 步开环执行”（类似 TD3 时代的 chunk 封装，推理频率降为 1/N）；
  3. 折中：观测里用上一动作做时序记忆（当前 obs 已含上一动作，105 维）。

### 3.3 与 Gazebo 模型的契约差异

| 项 | ir-sim 模型（checkpoints/ppo_N1） | Gazebo 模型（原 amr_rl_ws，已移除） |
| --- | --- | --- |
| obs | 105 = 100 lidar + goal 3 + 上一动作 2 | 62 = 60 束激光 + `[dist/10, wrap(angle)/π]` |
| 动作 | [-1,1]²（映射到 vel_min/vel_max） | [0,-1]~[0.5,1]²（速度限幅） |
| 网络 | 1024×1024（pi/vf 独立），≈1.16M 参数 | 64×64（pi/vf 独立），≈16.6K 参数 |
| 训练环境 | ir-sim 2D（configs/world/robot_world.yaml） | Gazebo（ROS2），obs 有 VecNormalize 归一化（obs_norm.json） |

Gazebo 模型的解包产物 `demos/data/policy_weights.pt`（62→64→64→2，隐藏层 tanh、输出无 tanh）
与 `demos/ppo_irsim_demo.py` 的演示契约一致（原 amr_rl_ws 工作区已移除，可在 GitHub 重新获取）。

---

## 4. 复现

```bash
d:\anaconda\envs\rl_env\python.exe doc\analyze_ppo_structure.py
# 控制台打印各模型逐层结构/参数量/超参数，并另存 doc\ppo_structure_report.txt
```