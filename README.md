# PPO End-to-End Navigation: INT8 Quantization × Action Chunking (N=5) — A Production-Ready Robot Deployment Repository

> Combines **INT8 quantization** and **action chunking (N=5)**.
> Core deliverables: the `deploy/ppo_nav/` ROS deployment package + `export/` artifacts + the `runs/ablation_runbook.md` on-robot workflow.

---

## Project Positioning

- **Problem being solved**: PPO end-to-end RL navigation inference latency → decision latency. Two orthogonal levers reduce single-step decision latency:
  - **INT8 quantization**: ONNX → TensorRT PTQ, real-robot bs1 inference latency **-33%** (Orin/TRT 8.5, measured in `runs/trt_bench_results.md`);
  - **action chunking (N=5)**: one inference emits 5 steps executed open-loop, cutting inference frequency to 1/5 and preventing collisions.
- **Main experiment**: navigation with **N=5 + INT8 quantization both active** (compared against: N=5/FP32, N=1/INT8, N=1/FP32, see the 4-cell ablation matrix).

> ⚠️ The current algorithm is **PPO** (MlpPolicy 1024×1024).

---

## Current Status (2026-08)

| Item | Status |
| --- | --- |
| PPO training (N=1 / N=5, ir-sim, 83ms step) | ✅ `checkpoints/ppo_final_N1/`, `checkpoints/ppo_final_N5/` |
| PPO simulation evaluation (success/collision rates) | ✅ `src/eval_ppo.py --checkpoint ...` |
| PPO → ONNX export (pure Gemm/Tanh, opset 17) | ✅ `onnx/export_ppo_onnx.py` (N=1→105 dim / N=5→113 dim) |
| INT8 real-robot inference speed | ✅ Real-robot bs1 INT8 **-33%** (Orin/TRT 8.5, see `runs/trt_bench_results.md`) |
| INT8 output accuracy / collision comparison (sim) | ✅ 300 paired episodes: max-abs 0.0087, collision rate +1% (`runs/quant_collision/`) |

---

## Environment Requirements

**PC (training / evaluation / export)**

- Python >= 3.10, conda env:
- `pip install -r requirements.txt`

**Robot (deployment)**

- ROS1 Noetic + Jetson Orin Nano (JetPack R35.6.1, TensorRT 8.5, CUDA)
- Deployment package: `deploy/ppo_nav/` (see `deploy/README.md`)

## Directory Structure

```
Essay/
├─ configs/             # train.yaml (master training/eval config) + world/ scenarios (training/eval world YAML)
├─ env/                 # ir-sim env wrappers: wrapper.py (IrSimEnv), reward.py, ppo_gym.py, multi_world.py
├─ src/                 # train_ppo.py: SB3 PPO training (N=1/N=5 selectable); eval_ppo.py: standalone eval
├─ onnx/                # unpack_ppo_actor.py / export_ppo_onnx.py (two-step ONNX export) / export_ablation.ps1
├─ deploy/              # real-robot deployment: ppo_nav ROS package (planner/safety/obs/TRT wrappers) + scripts/build_ptq_engine.py + README
├─ checkpoints/         # PPO models: ppo_final_N1/, ppo_final_N5/
├─ export/              # ONNX export artifacts (ppo_final_N1/, ppo_final_N5/: actor_fp32_bs1.onnx + calib_obs.npz)
├─ doc/                 # PPO model structure analysis (md + reproduction script + raw output)
├─ runs/                # training logs (ppo_*) + on-robot runbook + TRT measured records
└─ utils/               # config.py: config loading / deep merge / path resolution
```

## Quick Start: Train → Export → Quantize → Deploy

### 1) Training (N=1 and N=5)

```bash
# Full training (GPU, multi-world; chunk=1 outputs checkpoints/ppo_final_N1/, chunk=5 outputs ppo_final_N5/)
python src\train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 1 --ent-coef 0.01 --n-steps 4096
python src\train_ppo.py --steps 300000 --device cuda --name ppo_final --chunk 5 --ent-coef 0.01 --n-steps 4096
```

> Entropy-collapse guard: at the 83ms step size use `n_steps 4096` + `ent_coef 0.01`; export from `best_model.zip` (saved by eval success rate), not `ppo_model.zip`.

### 2) Evaluation (optional, simulation validation)

```bash
python src\eval_ppo.py --checkpoint checkpoints/ppo_final_N1/best_model.zip
```

### 3) Export ONNX + collect calibration data (two steps, two separate conda envs)

```bash
# ① Unpack Actor (rl_env) → export/ppo_final_N1/, export/ppo_final_N5/
python onnx\unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx\unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
# ② Export ONNX + INT8 calibration data (ir-sim env)
python onnx\export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib
python onnx\export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

Artifacts (`export/ppo_final_N{1,5}/`): `actor_fp32_bs1/bs8.onnx` (pure Gemm/Tanh), `calib_obs.npz` (real-obs calibration set), `policy_actor.pt`, `actor_config.yaml`.
One-shot script: `onnx/export_ablation.ps1`.

### 4) Build engines on board (FP32 baseline + INT8 quantization; batch=1!)

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy
cd ~/ppo_deploy/ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1
cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py \
    --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

### 5) Real-robot deployment and benchmarking

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
# Build + copy the 4 engines to models/ + launch; full procedure in runs/ablation_runbook.md
```

**On-robot 4-cell ablation matrix** (`runs/ablation_runbook.md`):

| Cell | Quantization | chunk | Engine | chunk_size | use_prev_action | obs dim |
|---|---|---|---|---|---|---|
| ① | FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | **INT8** | **N=5 (main experiment)** | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## Observation / Action Contract (IrSimEnv in env/wrapper.py)

- Observation:
  - lidar: binned-min then divided by `range_max` (`obs.max_bins`, default = lidar beam count)
  - goal polar coordinates: `[dist / goal_dist_norm, cos, sin]` (3 dims, `goal_dist_norm` default 10)
  - previous action (`obs.include_prev_action: true`): `N*action_dim` dims, per-dim `[lin*2, (ang+1)/2]`
    - N=1: trained through `step_single`, this channel is always 0 (deployment `use_prev_action=false`)
    - N=5: during chunk training writes back the previous chunk's 5 actions (deployment `use_prev_action=true`, planner accumulates `prev_history_`)
- Action: normalized `[-1,1]`, mapped to real velocities (diff-drive robot `[linear_vel, angular_vel]`); PPO output is μ, must be clipped to [-1,1]
- Reward: reaching +100 / collision -100 / per step `lin - 0.5|ang|` / proximity penalty / goal shaping
- Termination: reaching / collision / timeout (`max_steps`)

## Replacing the Environment

Place ir-sim world YAML files in `configs/world/`, then modify `env.world_name` (training scenario)
and `env.eval_world` (eval scenario) in `configs/train.yaml`. Requirements: robot equipped with a `lidar2d` sensor (beam count matching `obs.lidar_range_max`), and a `goal`.
