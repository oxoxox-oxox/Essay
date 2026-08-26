# demos / — PPO 迁移演示（ir-sim）

把 Gazebo（ROS2 + SB3 PPO）训练的导航策略（原 `amr_rl_ws` 工作区，已从仓库移除，可在 GitHub `wyh010731/amr_rl_ws` 重新获取）迁移到 ir-sim 2D 仿真里"看效果"。模型文件已内嵌到 `demos/data/`。
纯 torch 前向，**零新依赖**（`d:\anaconda\envs\ir-sim` 已具备 irsim 2.10.1 / torch 2.11 / numpy / yaml）。

## 运行

```bash
d:\anaconda\envs\ir-sim\python.exe demos\ppo_irsim_demo.py                # 可视化 10 集
d:\anaconda\envs\ir-sim\python.exe demos\ppo_irsim_demo.py --episodes 50  # 更多集数
d:\anaconda\envs\ir-sim\python.exe demos\ppo_irsim_demo.py --headless     # 无窗口批量
d:\anaconda\envs\ir-sim\python.exe demos\ppo_irsim_demo.py --noise-std 0  # 关激光噪声
```

实测（headless，noise_std=0.01，seed 随机）：**20/20 成功，0 碰撞，平均 105 步**。

## 迁移契约（与训练完全一致）

| 项 | 值 | 来源 |
| --- | --- | --- |
| obs | 62 = 60 束激光（360 束 -π..+π、3.5m 截断、每 6 束取 1、/3.5）+ `[dist/10, wrap(angle)/π]` | `amr_gazebo_env.py:_get_observation` |
| 动作 | `clip(μ, [0,-1], [0.5,1])`，等价 `model.predict(deterministic=True)` | 已用 SB3 2.9.0 逐位验证 maxdiff=0 |
| 网络 | `policy_weights.pt`：62→64→64→2，隐藏层 tanh，**输出层无 tanh** | `model_config.yaml` + `unpack_model.py` |
| dt | 0.1s | `step_time=0.1` |
| 终止 | dist<0.3 成功；robot.collision / min_laser<0.2 碰撞；≥500 步超时 | 训练同款判据 |

## 踩坑记录（后续改场景必读）

1. **输出层没有 tanh**：该模型 `squash_output=False`（普通高斯），SB3 `predict` 是把 μ 直接
   `np.clip` 到动作空间，不是 `tanh(μ)`。复制网络时输出层加 tanh 会导致动作系统性偏小。
2. **ir-sim `angle_range` 必须 < 2π**：`WrapTo2Pi(6.2832)=0`，全向激光会被包成 ±0.5°。
   用 `6.2830`（±π，1°/束，与 Gazebo 一致）。
3. **激光束序与 Gazebo 一致**（已实测）：0°=车头、正角=左、`ranges[i]` 对应
   `angle_min + i*angle_increment`；360 束时 `ranges[::6]` 即训练的下采样方式。
4. **linestring 墙会触发幽灵碰撞**：ir-sim 对 linestring 的碰撞判定过宽（离墙 1m 也算碰撞），
   训练场景的墙请用薄矩形（0.1m 厚，与 environment.world 一致），勿用 linestring。
5. **`collision_mode: 'stop'`**：碰撞后机器人会永久停住，必须在碰撞发生当步就终止 episode
   （用 `robot.collision` 标志），否则会变成假 timeout。
6. **视口默认只显示 `[0, width]×[0, height]`**：`world.x_range = [offset, offset+宽高]`，
   offset 默认 `[0,0]`。地图坐标含负值（如本场景 [-5,5]）时左下象限会被裁掉——必须在
   `world` 段写 `offset: [-5.5, -5.5]`（宽高相应设为 11），视口才会完整覆盖并留白。

## 后续可做的事

- 把 `policy_weights.pt` 用 `torch.onnx` 导出 ONNX，再到 Jetson（TensorRT）上做 INT8 量化；
  当前 PPO 是单步策略，action chunking 需自行封装（一次推理输出 N 步、开环执行）。
- 在 ir-sim 里加动态障碍/噪声，观察策略鲁棒性（sim2sim 差距主要来自 Gazebo 动力学 vs 运动学模型）。
