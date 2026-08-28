# deploy/ppo_nav — PPO Real-Robot Navigation Deployment Package (ROS1 Noetic / Jetson Orin Nano / TRT 8.5)

> This package is the real-robot carrier for the project's "INT8 quantization × action chunking (N=5)": PPO policy → ONNX → TensorRT engine → real-robot navigation.
> It supports two model variants: **N=1** (obs 105 dims, output [1,2]) and **N=5** (obs 113 dims, output [1,10]).
> Main experiment = **N=5 + INT8**. Full on-robot / benchmarking procedure is in `runs/ablation_runbook.md`.

## ⚠️ Read first: two key facts about obs construction

1. **N=1 models** are trained through `IrSimEnv.step_single`, so the "previous action" channel is always 0
   (verified with calib_obs.npz) → `obs_builder` **outputs 0 by default** (`use_prev_action: false`).
2. **N=5 models** are trained with action chunking; the obs "previous action" channel = the previous chunk's 5 actions
   → deployment uses `use_prev_action: true`, with the planner accumulating `prev_history_`.

   ⚠️ Both must strictly match the model in use (see the matrix in `runs/ablation_runbook.md` section 0).

## Package Structure

```
deploy/ppo_nav/
├─ package.xml / CMakeLists.txt   # catkin package (gnu++14, TRT 8.5 + CUDA)
├─ include/ppo_nav/obs_builder.hpp       # obs construction (N=1:105 dims / N=5:113 dims, consistent with env/wrapper.py)
├─ include/ppo_nav/tensorrt_engine.hpp   # TRT engine wrapper
├─ src/obs_builder.cpp           # lidar forward resampling/normalization + goal [dist/10,cos,sin] + prev
├─ src/tensorrt_engine.cpp       # engine load/forward (CUDA buffers, synchronous)
├─ src/planner_node.cpp          # /scan+/odom -> obs -> inference -> /cmd_vel_planner (chunk open-loop execution)
├─ src/safety_node.cpp           # clipping/emergency-stop/watchdog/battery/charging -> /cmd_vel
├─ config/params.yaml            # all tunable parameters (chunk_size / use_prev_action / velocity clamps…)
├─ launch/planner.launch         # starts planner + safety together; safety.launch can be started alone
├─ scripts/measure_node.py       # real-robot benchmarking measurement (latency/frequency/emergency-stop)
└─ scripts/build_ptq_engine.py   # build INT8 engine on board (real-obs calibration, batch=1 for navigation)
```

## Data Contract (exact obs / action format)

- Input `obs [1,105]` (N=1) or `[1,113]` (N=5):
  - `[0:100]` lidar: resampled to 100 beams across forward ±90° at the training angles (linspace(-π/2, π/2, 100)),
    inf/NaN/>7.0 set to 1.0; `/range_max_norm(7.0)`
  - `[100:103]` goal polar coordinates: `[dist / goal_dist_norm(10), cos(a), sin(a)]`, `a = wrap(atan2(gy-y, gx-x) - yaw)`
  - `[103:]` previous action:
    - N=1 → `[103:105]` always 0 (`use_prev_action: false`)
    - N=5 → `[103:113]` previous chunk's 5 actions (`use_prev_action: true`, accumulated by the planner)
- Output `action [1,2]` (N=1) or `[1,10]` (N=5): μ (PPO has no output tanh) → in-node `clip(μ, [-1,1])` →
  `scale_action`: `vel_min + (a+1)/2 * (vel_max-vel_min)` → `max_linear/max_angular` clamps → `/cmd_vel_planner`
- Decisions are **driven by the laser frames** (each new `/scan` triggers one serial inference, ~12Hz); with `chunk_size>1`
  a single inference produces N open-loop steps, cutting inference frequency to 1/N; optional `min_decision_period` throttling.
- goal is defined in the `odom_combined` frame (relative to start)

## On-Robot Procedure

### 1) PC-side artifacts (see root README step 3)

```bash
d:/anaconda/envs/rl_env/python.exe onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
d:/anaconda/envs/ir-sim/python.exe onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib
d:/anaconda/envs/rl_env/python.exe onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
d:/anaconda/envs/ir-sim/python.exe onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

### 2) Build engines on board (batch=1!)

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>; cd ~/ppo_deploy/ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1
cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

### 3) Sync package and build

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot
rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
# Put the engines into the package models/ (or override at launch with absolute engine_path:=)
mkdir -p ~/wheeltec_robot/src/ppo_nav/models
cp ~/ppo_deploy/ppo_final_N1/actor_{fp32,int8}_n1.engine ~/wheeltec_robot/src/ppo_nav/models/
cp ~/ppo_deploy/ppo_final_N5/actor_{fp32,int8}_n5.engine ~/wheeltec_robot/src/ppo_nav/models/
```

> Build success criterion: the `catkin_make` output **must contain** `Building CXX object .../planner_node.dir/...`;
> if you only see `Built target` + `Clock skew detected` = not recompiled (old binary). TRT `deprecated` warnings are normal.

### 4) Launch (change parameters per cell; full 4-cell matrix in the runbook)

```bash
# Main experiment: INT8 + N=5
rosparam set /planner/chunk_size 5
rosparam set /planner/use_prev_action true
roslaunch ppo_nav planner.launch engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine
# Or switch to the FP32 baseline (single variable): engine_path:=.../actor_fp32_n5.engine
# N=1 cells: chunk_size 1, use_prev_action false, use actor_*_n1.engine
# safety is already in planner.launch; or launch alone: roslaunch ppo_nav safety.launch
```

### 5) On-vehicle verification order

1. Keep the handbrake released first: confirm `[TRT] engine loaded: input 113 elements, output 10 elements` (N=5)
   or `input 105, output 2` (N=1) — **input dimension must match**, otherwise the engine was built/parameterized wrong.
2. On-site `reverse_scan` check: place an obstacle beside the front, verify the closest beam base angle ≈±90° (decides the reverse_scan value).
3. obs value comparison: `debug_log=true` prints obs/action, compare with the torch output on PC via max abs diff (should be < 0.05).
4. Keep `max_linear` small (recommended 0.3), test at low speed, add obstacles gradually.

## Log Interpretation

**Normal** (N=1):
```
[TRT] engine loaded: input 105 elements, output 2 elements
[planner] ready. goal=(5.00, 5.00), period=0.30s, use_prev_action=0
# no engine forward failed
```
**Normal** (N=5):
```
[TRT] engine loaded: input 113 elements, output 10 elements
[planner] ready. goal=(5.00, 5.00), period=0.30s, use_prev_action=1
```
**Abnormal `engine input=105 (expect 105)` mismatch**: wrong engine batch (a bs8 engine was used) → return to step 2 and rebuild a batch=1 engine.

## Parameters (config/params.yaml, overridable at launch)

goal_x/goal_y (odom frame), min_decision_period(0.0, frame-driven; set 0.3 to match training), scan_timeout(0.5), debug_log,
num_beams(100)/range_max_norm(7.0)/goal_dist_norm(10.0)/half_fov_deg(90)/laser_yaw_fallback_deg(180)/reverse_scan,
chunk_size(1; use 5 for N=5), use_prev_action(false; use true for N=5), vel_min/vel_max, max_linear(0.8, set to 0.3 before driving)/max_angular(1.0),
safety section (stop_dist/slow_dist/slow_factor/min_voltage/watchdog_timeout/front_only).

## Pitfall Quick Reference

- **Engine batch**: navigation always uses batch=1 engines (`--batch-size 1` is mandatory); a bs8 engine will stop the robot on forward size mismatch
- **N=5 obs dim**: 113 = 100 lidar + 3 goal + 10 prev; if the engine input doesn't match it will stop the robot
- **Clock skew skips compilation**: scp carries future mtimes → `touch` the sources before building (see step 3)
- **CATKIN_WHITELIST_PACKAGES**: after building a single package, reset with `""` to restore the whole workspace
- **gnu++14**: `-std=c++14` will hide the M_PI compilation failure; TRT deprecated warnings don't affect success
- **Command-line `$(find)`**: bash expands `$()` first → use an absolute path or single quotes for engine_path at launch
- **TF32**: FP32 baseline engines must use `--noTF32` to be true FP32
