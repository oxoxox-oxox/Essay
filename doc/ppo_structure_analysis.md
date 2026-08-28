# PPO Model Structure Analysis

> Purpose: analyze the network structure of the trained PPO models in this repository, in preparation for real-robot INT8 quantization and action chunking (N=5).
> All structural information comes from **actually loading the models** (SB3 2.9.0 / torch 2.11.0); reproduction script: `doc/analyze_ppo_structure.py`,
> raw output: `doc/ppo_structure_report.txt`.

---

## 1. PPO Model Inventory in This Repository

| Path | Source | obs dim | Action space | net_arch | Total policy params | Training steps (num_timesteps) |
| --- | --- | --- | --- | --- | --- | --- |
| `checkpoints/ppo_N1/best_model.zip` | ir-sim full training (EvalCallback best) | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 240,000 |
| `checkpoints/ppo_N1/ppo_model.zip` | ir-sim full training (final model) | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 300,032 |
| `checkpoints/ppo_smoke_N1/best_model.zip` | ir-sim smoke test | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 10,000 |
| `checkpoints/ppo_smoke_N1/ppo_model.zip` | ir-sim smoke test | 105 | [-1,1]² | [1024,1024] | 2,319,365 | 20,480 |

The project's main models are **`checkpoints/ppo_final_N1/` and `checkpoints/ppo_final_N5/`** (83ms step, used for the INT8+N=5 ablation;
structure identical to the 1024×1024 below, only obs/action dims differ); `checkpoints/ppo_mw_N1/` (multi-world retrain v2) has exactly the same structure as ppo_N1, only the weights differ.

> Note: the original amr_rl_ws (Gazebo/ROS2) training workspace and `demos/` demos have been removed from the repository (GitHub: `wyh010731/amr_rl_ws`);
> this repository keeps only the models and deployment for the ir-sim contract (PPO single-step / action chunk).

---

## 2. Main Model Structure (checkpoints/ppo_mw_N1 / ppo_N1, ir-sim env, same architecture)

Common attributes (identical across all 4 zips):

- Algorithm: SB3 PPO, `MlpPolicy` (i.e. `ActorCriticPolicy`), continuous Gaussian action distribution (DiagGaussian)
- Activation: **Tanh** (SB3 MlpPolicy default)
- `squash_output=False`: action output has **no tanh squashing**, deterministic prediction = `clip(μ, [-1, 1])`
- `share_features_extractor=True`: for MlpPolicy this is just an identity FlattenExtractor, **no real weight sharing**

### 2.1 Network Architecture (printed from the actually loaded model)

```
obs (105,) ──┬─> Actor(policy):                      params
             │    Linear(105 -> 1024) + Tanh        108,544
             │    Linear(1024 -> 1024) + Tanh     1,049,600
             │    Linear(1024 -> 2)                  2,050
             │    └─ μ (2,); plus 2 learnable log_std params
             └─> Critic(value):
                  Linear(105 -> 1024) + Tanh        108,544
                  Linear(1024 -> 1024) + Tanh     1,049,600
                  Linear(1024 -> 1)                  1,025
                  └─ V (1,)
```

Key points:

- In SB3, `net_arch=[1024,1024]` is equivalent to `dict(pi=[1024,1024], vf=[1024,1024])`: Actor and Critic are **two independent MLPs, weights not shared**
- Actor params (excluding log_std): 1,160,194 ≈ **1.16M**; Critic: 1,159,169; total policy params: **2,319,365 ≈ 2.32M**
- Per-forward MACs: 105×1024 + 1024×1024 + 1024×2 = **1,158,144 ≈ 1.16M** (pure GEMM across the whole graph)

### 2.2 Hyperparameters (values saved in the zips)

| Hyperparameter | Value |
| --- | --- |
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| n_steps | 1024 |
| batch_size | 128 |
| n_epochs | 10 |
| clip_range | 0.2 (constant) |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

Forward check: `obs (1,105) -> dist mean (1,2), value (1,1)`.

---

## 3. Implications for the Follow-Up Work

### 3.1 INT8 Quantization

- Only the **Actor** part is needed for ONNX export: `mlp_extractor.policy_net` + `action_net`; input `[B,105]` FP32, output μ `[B,2]`
- The whole graph is only `Gemm` + `Tanh` (obs normalization is done on the obs-construction side, not in the network), **no conv, no BN** → TensorRT can do full-graph INT8
  (a pure MLP can be down-precisioned across the whole graph, avoiding Reformat casts, consistent with previous conclusions)
- Deterministic action = `clip(μ, [-1,1])`; quantization error first acts on μ; μ's value range is friendly to INT8 representation
- **Already implemented**: `onnx/unpack_ppo_actor.py` (SB3 zip → pure weights) + `onnx/export_ppo_onnx.py` (→ ONNX + calibration data);
  the exported graph ops are only `[Gemm, Tanh]` (opset 17), onnxruntime vs torch output max-abs-diff ≈ 7e-7;
  build the on-board INT8 engine with `deploy/ppo_nav/scripts/build_ptq_engine.py` (real-obs calibration)

### 3.2 Action Chunking (N=5, already implemented)

- The current PPO single-step policy outputs only `(B,2)` per inference; the N=5 refactor uses "training-side action space changed to `(B, 2N)`":
  the network outputs 5 steps of actions, and the "previous chunk action" obs dim is correspondingly `5*action_dim` (obs 113 dims).
- Training: `src/train_ppo.py --chunk 5` (`step_chunk` open-loop execution, obs encodes the previous chunk's actions).
- Deployment: the planner in `deploy/ppo_nav` produces N steps per inference and pushes them into pending_, then the next N-1 frames only pop actions without inference
  (open-loop), cutting inference frequency to 1/N; with `use_prev_action=true` it writes the previous chunk's actions back into obs.

### 3.3 Contract Difference Summary

| Item | N=1 model (checkpoints/ppo_final_N1) | N=5 model (checkpoints/ppo_final_N5) |
| --- | --- | --- |
| obs | 105 = 100 lidar + goal 3 + prev action 2 (always 0 during training) | 113 = 100 lidar + goal 3 + prev chunk 10 |
| action | [-1,1]² (mapped to vel_min/vel_max) | [-1,1]¹⁰ (5 steps × [lin,ang]) |
| network | 1024×1024 (pi/vf independent), ≈1.16M params | same, only input/output dims differ |
| training env | ir-sim 2D (configs/world/*.yaml) | same |

---

## 4. Reproduction

```bash
d:\anaconda\envs\rl_env\python.exe doc\analyze_ppo_structure.py
# prints per-model layer-by-layer structure / param counts / hyperparameters, and saves doc\ppo_structure_report.txt
```
