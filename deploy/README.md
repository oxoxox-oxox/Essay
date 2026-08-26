# deploy/ppo_nav — PPO 真机导航部署包（ROS1 Noetic / Jetson Orin Nano / TRT 8.5）

> 重建版：TD3 时代的 td3_nav 已清理，本包按 **PPO 契约**重写。
> 支持两种模型：**N=1**（obs 105 维，输出 [1,2]）与 **N=5**（obs 113 维，输出 [1,10]）。
> 消融实验（量化×chunk）的完整上机流程见 `runs/ablation_runbook.md`。

## ⚠️ 先读：一个关键事实（影响 obs 构造）

当前 PPO 模型（`checkpoints/ppo_mw_N1`，多世界重训 v2）训练走 `IrSimEnv.step_single`，而"上一动作"通道
只在 TD3 的 `step_chunk` 里更新——**PPO 训练时 obs 最后 2 维恒为 0**（已用 calib_obs.npz 验证）。
因此 `obs_builder` **默认输出 0**（`planner.use_prev_action: false`），与训练分布一致；
不要改成真实上一动作，除非你用带 prev 反馈重新训练的模型。

## 包结构

```
deploy/ppo_nav/
├─ package.xml / CMakeLists.txt   # catkin 包（gnu++14，TRT 8.5 + CUDA）
├─ include/ppo_nav/obs_builder.hpp       # obs 105 维构造（与 env/wrapper.py 一致）
├─ include/ppo_nav/tensorrt_engine.hpp   # TRT 引擎封装
├─ src/obs_builder.cpp           # lidar 前向重采样/归一化 + goal [dist/10,cos,sin] + prev(0)
├─ src/tensorrt_engine.cpp       # 引擎 load/forward（CUDA buffer，同步）
├─ src/planner_node.cpp          # /scan+/odom -> obs -> 推理 -> /cmd_vel_planner
├─ src/safety_node.cpp           # 限幅/急停/看门狗/电量/充电 -> /cmd_vel
├─ config/params.yaml            # 全部可调参数
└─ launch/planner.launch         # planner + safety 一起起；safety.launch 可单独起
```

## 数据契约（obs / action 精确格式）

- 输入 `obs [1,105]`：
  - `[0:100]` lidar：前向±90°按训练角度（linspace(-π/2, π/2, 100)）重采样 100 束，
    inf/NaN/>7.0 按 1.0；`/range_max_norm(7.0)`
  - `[100:103]` goal 极坐标：`[dist / goal_dist_norm(10), cos(a), sin(a)]`，`a = wrap(atan2(gy-y, gx-x) - yaw)`
  - `[103:105]` 上一动作：**恒 0**（见上；训练时该通道为 0）
- 输出 `action [1,2]`：μ（PPO 无输出 tanh）→ 节点内 `clip(μ, [-1,1])` →
  `scale_action`：`vel_min + (a+1)/2 * (vel_max-vel_min)` → `max_linear/max_angular` 限幅 → `/cmd_vel_planner`
- 决策由**雷达帧驱动**（收到新 `/scan` 就串行推理一次，约 12Hz）；可选 `min_decision_period` 节流
  （0=纯帧驱动；设 0.3 可把决策频率压回训练 step_time）；goal 定义在 `odom_combined` 系（相对起点）

## 上机步骤

### 1) PC 侧产物（已生成，无需重复）

```bash
d:/anaconda/envs/rl_env/python.exe train/unpack_ppo_actor.py --checkpoint checkpoints/ppo_mw_N1/best_model.zip --name ppo_mw
d:/anaconda/envs/ir-sim/python.exe train/export_ppo_onnx.py --actor export/ppo_mw/policy_actor.pt --make-calib
```

### 2) 同步到板上并建引擎（batch=1！）

```bash
scp -r export/ppo_mw/ wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>
cd ~/ppo_deploy
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32.engine
python3 build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8.engine --batch-size 1
```

### 3) 同步包并编译

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot
rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
# 把引擎放进包内 models/（或启动时用绝对路径 engine_path:= 覆盖）
mkdir -p ~/wheeltec_robot/src/ppo_nav/models
cp ~/ppo_deploy/actor_fp32.engine ~/ppo_deploy/actor_int8.engine ~/wheeltec_robot/src/ppo_nav/models/
```

> 编译成功判据：`catkin_make` 输出**必须出现** `Building CXX object .../planner_node.dir/...`；
> 若只有 `Built target` + `Clock skew detected` = 没重编（旧二进制）。TRT 的 `deprecated` 警告正常。

### 4) 启动

```bash
roslaunch ppo_nav planner.launch engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8.engine
# 或换 FP32 基线（单一变量）：engine_path:=.../actor_fp32.engine
# safety 已在 planner.launch 内；也可单独: roslaunch ppo_nav safety.launch
```

### 5) 上车验证顺序

1. 先不松手刹：确认 `[TRT] engine loaded: input 105 elements, output 2 elements`、`/cmd_vel_planner` 有输出
2. `reverse_scan` 现场校验：障碍放车头一侧，看最近束 base 角应≈±90°（决定 reverse_scan 取值）
3. obs 数值比对：`debug_log=true` 打点 obs/action，与 PC 上 torch 输出做 max abs diff（应 < 0.05）
4. `max_linear` 保持小值（建议 0.3）低速试跑，逐级加障碍

## 日志判读

**正常**：
```
[TRT] engine loaded: input 105 elements, output 2 elements
[planner] ready. goal=(5.00, 5.00), period=0.30s, use_prev_action=0
# 无 engine forward failed
```

**异常 `engine input=105 (expect 105)` 不一致**：引擎 batch 填错（用了 bs8 引擎）→ 回第 2 步重建 batch=1 引擎。

## 参数（config/params.yaml，launch 可覆盖）

goal_x/goal_y（odom 系）、min_decision_period(0.0，帧驱动；对齐训练设 0.3)、scan_timeout(0.5)、debug_log、
num_beams(100)/range_max_norm(7.0)/goal_dist_norm(10.0)/half_fov_deg(90)/laser_yaw_fallback_deg(180)/reverse_scan、
chunk_size(1)、use_prev_action(false)、vel_min/vel_max、max_linear(0.8，上车先调 0.3)/max_angular(1.0)、
safety 段（stop_dist/slow_dist/slow_factor/min_voltage/watchdog_timeout/front_only）。

## 踩坑速查

- **引擎 batch**：导航恒用 batch=1 引擎（`--batch-size 1` 不能省），bs8 引擎 forward 尺寸不匹配会停车
- **Clock skew 跳过编译**：scp 带未来 mtime → 先 `touch` 源码再编（见第 3 步）
- **CATKIN_WHITELIST_PACKAGES**：单包构建后必须用 `""` 清空恢复整工作区
- **gnu++14**：`-std=c++14` 会隐藏 M_PI 编译失败；TRT deprecated 警告不影响成功
- **命令行 `$(find)`**：bash 会先展开 `$()` → 启动时 engine_path 用绝对路径或单引号
- **TF32**：FP32 基线引擎必须 `--noTF32` 才是真 FP32