# INT8 量化 vs FP32 碰撞率对照（ir-sim 仿真）

- checkpoint: `checkpoints/ppo_mw_N1/best_model.zip`
- 校准集: `export/ppo_mw/calib_obs.npz`
- episodes: 300（配对，同起点/无激光噪声/静态障碍 eval_world）

## 输出一致性（校准集 256）

| max-abs-diff | mean-abs-diff | cos |
|---|---|---|
| 0.00875 | 0.00185 | 0.999879693 |

## 碰撞/成功率

| 条件 | success | collision | timeout |
|---|---|---|---|
| fp32 | 41.67% | 17.00% | 41.33% |
| int8 | 41.33% | 18.00% | 40.67% |

净效应(量化后 碰撞->成功 减 成功->碰撞): -1 集 / 300 集
平均首步 max-abs 动作扰动: 0.00327
