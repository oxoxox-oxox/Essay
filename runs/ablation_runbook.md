# 真机 4 格消融实验 · 上机 Runbook（量化 × action chunk）

> 目标：在 Jetson Orin Nano 上跑 4 格（量化 ON/OFF × chunk ON/OFF），
> 测 scan→决策延时、推理耗时、推理频率、动作频率、安全急停次数、GPU 占用。
> 对应模型：以 83ms step_time 重训的 `ppo_83ms_N1`（单步）与 `ppo_83ms_N5`（chunk=5）。

---

## 0. 4 格矩阵与参数

| 格 | 量化 | chunk | 引擎 | chunk_size | use_prev_action | obs 维 |
|---|---|---|---|---|---|---|
| ① | 关 FP32 | N=5 | `actor_fp32_n5.engine` | 5 | true | 113 |
| ② | 开 INT8 | N=5 | `actor_int8_n5.engine` | 5 | true | 113 |
| ③ | 开 INT8 | N=1 | `actor_int8_n1.engine` | 1 | false | 105 |
| ④ | 关 FP32 | N=1 | `actor_fp32_n1.engine` | 1 | false | 105 |

> ⚠️ N=5 必须 `use_prev_action=true`（模型用了 prev 通道）；N=1 必须 `false`（prev 恒 0，与训练一致）。

---

## 1. PC 侧产物（训练完成后执行）

```bash
# N=1（obs 105, act 2）
d:/anaconda/envs/rl_env/python.exe train/unpack_ppo_actor.py --checkpoint checkpoints/ppo_83ms_N1/best_model.zip --name ppo_83ms_N1
d:/anaconda/envs/ir-sim/python.exe train/export_ppo_onnx.py --actor export/ppo_83ms_N1/policy_actor.pt --make-calib

# N=5（obs 113, act 10）
d:/anaconda/envs/rl_env/python.exe train/unpack_ppo_actor.py --checkpoint checkpoints/ppo_83ms_N5/best_model.zip --name ppo_83ms_N5
d:/anaconda/envs/ir-sim/python.exe train/export_ppo_onnx.py --actor export/ppo_83ms_N5/policy_actor.pt --make-calib --chunk 5
```

产物每个 `export/ppo_83ms_N{1,5}/` 下有：`actor_fp32_bs1/bs8.onnx`、`calib_obs.npz`、`policy_actor.pt`、`actor_config.yaml`。
把 `export/ppo_mw/build_ptq_engine.py` 复制进这两个目录（板上建 INT8 引擎用）。

---

## 2. 板上建引擎（先同步，再建）

```bash
scp -r export/ppo_83ms_N1 export/ppo_83ms_N5 wheeltec@<ip>:~/ppo_deploy/
ssh wheeltec@<ip>; cd ~/ppo_deploy
cd ppo_83ms_N1
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n1.engine
python3 build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n1.engine --batch-size 1
cd ../ppo_83ms_N5
/usr/src/tensorrt/bin/trtexec --onnx=actor_fp32_bs1.onnx --noTF32 --saveEngine=actor_fp32_n5.engine
python3 build_ptq_engine.py --onnx actor_fp32_bs1.onnx --calib calib_obs.npz --out actor_int8_n5.engine --batch-size 1
```

> 导航恒用 batch=1；`--noTF32` 才是真 FP32 基线。把 4 个 `.engine` 拷进
> `~/wheeltec_robot/src/ppo_nav/models/`。

---

## 3. 部署包编译

```bash
scp -r deploy/ppo_nav/ wheeltec@<ip>:~/wheeltec_robot/src/
ssh wheeltec@<ip>
cd ~/wheeltec_robot/src/ppo_nav && find . \( -name '*.cpp' -o -name '*.hpp' \) -exec touch {} +
cd ~/wheeltec_robot && rm -rf build/ppo_nav
catkin_make -DCATKIN_WHITELIST_PACKAGES="ppo_nav" -j$(nproc)
source devel/setup.bash
catkin_make -DCATKIN_WHITELIST_PACKAGES=""
```

> 判据：输出必须出现 `Building CXX object ... planner_node`。TRT 的 deprecated 警告正常。

---

## 4. 跑分协议（每格都要做）

1. **锁频**：`sudo nvpmodel -m 0 && sudo jetson_clocks`
2. **起测量**：
   ```bash
   # 终端 A：GPU/算力（跑分期间一直开着，结束后 Ctrl+C 保存）
   tegrastats --interval 500 --logfile /home/wheeltec/gpu_<格>.log
   # 终端 B：延时/频率/急停测量节点
   python3 ~/wheeltec_robot/src/ppo_nav/scripts/measure_node.py _out:=/home/wheeltec/measure_<格>.csv
   ```
3. **启动导航**（按格改 `engine_path` 与参数）：
   ```bash
   # 例：格② INT8 + N=5
   rosparam set /planner/chunk_size 5
   rosparam set /planner/use_prev_action true
   roslaunch ppo_nav planner.launch engine_path:=/home/wheeltec/wheeltec_robot/src/ppo_nav/models/actor_int8_n5.engine
   ```
   （或用 launch 参数/临时改 `config/params.yaml`。N=1 的格：`chunk_size 1`、`use_prev_action false`。）
4. **固定测试道**：goal 设 `(5,5)`（odom 系），车前 1–2m 放一个障碍，起步低速
   （`/planner/max_linear 0.3`），每格重复 **N 次**（建议 ≥10 次），每次记录
   是否到达 / 碰撞 / 急停。
5. **结束**：Ctrl+C 停导航与测量，保存 `measure_<格>.csv` 与 `gpu_<格>.log`。

---

## 5. 每格要留的数据

| 数据 | 来源 |
|---|---|
| scan→决策延时 | `measure_*.csv` 的 `scan_age_ms` 列（mean/median/p99） |
| 推理耗时（量化项） | `measure_*.csv` 的 `fwd_ms` 列 |
| 推理频率 | measure 汇总的 `inference Hz` |
| 动作频率 | measure 汇总的 `action Hz` |
| 安全急停次数 | measure 汇总的 `safety stops` |
| GPU 占用 | `gpu_*.log`（tegrastats 的 GR3D 占用率） |
| 碰撞/到达 | 第 4.4 步人工记录 |

---

## 6. 上车验证顺序（先别松手刹）

1. 确认日志：`[TRT] engine loaded: input 113 elements, output 10 elements`（N=5）
   或 `input 105, output 2`（N=1）——**input 维数必须匹配**，否则 engine 建错/参数错。
2. `reverse_scan` 现场校验（障碍放车头一侧，看最近束 base 角 ≈ ±90°）。
3. `debug_log=true` 比对 obs/action 与 PC torch 输出（max abs diff 应 < 0.05）。
4. `max_linear` 保持 0.3 低速试跑，逐级加障碍。

---

## 7. 已知注意点

- N=5 模型 obs 是 **113 维**（100 lidar + 3 goal + 10 prev），N=1 是 **105 维**；
  引擎 input 维数对不上会停车，见第 6.1 条。
- N=5 的 prev 通道由 planner 累积最近 5 步动作（`prev_history_`），无需手动干预。
- 决策已改雷达帧驱动（≈12Hz），N=5 的开环窗口 ≈ 5×83ms = 0.42s。
- `build_ptq_engine.py` 里“预期 105”的提示对 N=5 无害（只是 print，按 ONNX 实际维度建）。

## 8. ⚠️ 训练质量已知问题（熵坍缩）与重训配方

83ms 步长下每集 ~500-900 步，而 `n_steps=1024` 现在只覆盖 ~1-2 集（旧 0.3s 是 ~10 集），
配合 `ent_coef=0` 会在 ~130-150k 步出现**熵坍缩**（策略 std→0.5、rollout 成功率下滑）。
观测：N=1 在 165k 时 rollout 跌到 ~36%；N=5 在 40k 时仍健康（std 0.97、eval 70%）。

**务必用 `best_model.zip` 导出（EvalCallback 按 mean reward 保留峰值），不要用 `ppo_model.zip`。**

若需重训（提高质量），用 `--ent-coef` / `--n-steps` 一键重跑（无需改代码）：
```bash
d:/anaconda/envs/rl_env/python.exe src/train_ppo.py --steps 300000 --device cuda --name ppo_83ms_v2 --chunk 1 --ent-coef 0.01 --n-steps 4096
d:/anaconda/envs/rl_env/python.exe src/train_ppo.py --steps 300000 --device cuda --name ppo_83ms_v2 --chunk 5 --ent-coef 0.01 --n-steps 4096
```

⚠️ **更严重的问题（2026-08-25 实测发现）**：EvalCallback 按 **mean reward** 保存 best_model，
而熵坍缩后模型会在 1800 步里“徘徊+刷 goal shaping 奖励”，导致 **mean reward 反升、success 归零**，
best_model 会被坍缩模型覆盖（N=1 在 180k 时 reward 383 > 80k 的 355，但 success 0%）。
结论：**当前 best_model 已被 0% 成功率模型覆盖，N=1 需重训**（用上面的 v2 命令）。
若沿用 mean reward 指标，重训后仍可能重演；稳妥做法是后续把 EvalCallback 换成按 success 保存。
