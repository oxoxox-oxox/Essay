# Real-Robot 4-Cell Ablation · On-Robot Runbook (Quantization × Action Chunk)

> Goal: on a Jetson Orin Nano, run 4 cells (quantization ON/OFF × chunk ON/OFF),
> measuring scan→decision latency, inference time, inference frequency, action frequency, safety emergency-stops, GPU usage.
> **Main experiment = cell ② (INT8 × N=5)**; corresponding models: `ppo_final_N1` (single-step) and `ppo_final_N5` (chunk=5).

---

## 0. Four-Cell Matrix and Parameters

| Cell | Quantization | chunk | Engine | chunk_size | use_prev_action | obs dim |
| --- | --- | --- | --- | --- | --- | --- |
| ① | OFF FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | ON INT8 | N=5 | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | ON INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | OFF FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

---

## 1. PC-Side Artifacts (run after training)

```bash
# N=1 (obs 105, act 2)
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N1/best_model.zip --name ppo_final_N1
python onnx/export_ppo_onnx.py --actor export/ppo_final_N1/policy_actor.pt --make-calib

# N=5 (obs 113, act 10)
python onnx/unpack_ppo_actor.py --checkpoint checkpoints/ppo_final_N5/best_model.zip --name ppo_final_N5
python onnx/export_ppo_onnx.py --actor export/ppo_final_N5/policy_actor.pt --make-calib --chunk 5
```

Each `export/ppo_final_N{1,5}/` contains: `actor_fp32_bs1/bs8.onnx`, `calib_obs.npz`, `policy_actor.pt`, `actor_config.yaml`.
The on-board INT8 engine build script uses the repo-tracked `deploy/ppo_nav/scripts/build_ptq_engine.py`.

---

## 2. Build Engines On Board (sync first, then build)

```bash
scp -r export/ppo_final_N1 export/ppo_final_N5 wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>; cd ~/ppo_deploy
cd ppo_final_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1
cd ../ppo_final_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 ~/wheeltec_robot/src/ppo_nav/scripts/build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

> Navigation always uses batch=1; `--noTF32` is required for a true FP32 baseline. Copy the 4 `.engine` files into
> `~/wheeltec_robot/src/ppo_nav/models/`.

---

## 3. Build the Deployment Package

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot && rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
```

> Criterion: the output must contain `Building CXX object ... planner_node`. TRT deprecated warnings are normal.

---

## 4. Benchmarking Protocol (performed for every cell)

1. **Lock frequency**: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. **Start measurement**:

   ```bash
   # Terminal A: GPU/compute (keep running throughout, Ctrl+C to save after)
   tegrastats --interval 500 --logfile /home/wheeltec/gpu_<cell>.log
   # Terminal B: latency/frequency/emergency-stop measurement node
   python3 ~/wheeltec_robot/src/ppo_nav/scripts/measure_node.py _out:=/home/wheeltec/measure_<cell>.csv
   ```

3. **Launch navigation** (change `engine_path` and parameters per cell):

   ```bash
   # Example: cell ② INT8 + N=5
   rosparam set /planner/chunk_size 5
   rosparam set /planner/use_prev_action true
   roslaunch ppo_nav planner.launch engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine
   ```

   (Or use launch args / temporarily edit `config/params.yaml`. N=1 cells: `chunk_size 1`, `use_prev_action false`.)
4. **Fixed test lane**: goal at `(5,5)` (odom frame), place an obstacle 1–2m ahead of the robot, start at low speed
   (`/planner/max_linear 0.3`), repeat **N times** per cell (recommended ≥10), recording each time
   whether it reached / collided / emergency-stopped.
5. **Finish**: Ctrl+C to stop navigation and measurement, save `measure_<cell>.csv` and `gpu_<cell>.log`.

---

## 5. Data to Keep Per Cell

| Data | Source |
| --- | --- |
| scan→decision latency | `scan_age_ms` column of `measure_*.csv` (mean/median/p99) |
| Inference time (quantization item) | `fwd_ms` column of `measure_*.csv` |
| Inference frequency | `inference Hz` from measure summary |
| Action frequency | `action Hz` from measure summary |
| Safety emergency-stops | `safety stops` from measure summary |
| GPU usage | `gpu_*.log` (GR3D usage from tegrastats) |
| Collisions / arrivals | manually recorded in step 4.4 |

---

## 6. On-Vehicle Verification Order

1. Confirm the log: `[TRT] engine loaded: input 113 elements, output 10 elements` (N=5)
   or `input 105, output 2` (N=1) — **input dimension must match**, otherwise the engine was built/parameterized wrong.
2. On-site `reverse_scan` check (place an obstacle beside the front, verify the closest beam base angle ≈ ±90°).
3. With `debug_log=true`, compare obs/action against the PC torch output (max abs diff should be < 0.05).
4. Keep `max_linear` at 0.3 for low-speed trials, add obstacles gradually.
