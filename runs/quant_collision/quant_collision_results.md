# INT8 Quantization vs FP32 Collision-Rate Comparison (ir-sim simulation)

- checkpoint: `checkpoints/ppo_mw_N1/best_model.zip`
- calibration set: `export/ppo_mw/calib_obs.npz`
- episodes: 300 (paired, same starts / no laser noise / static-obstacle eval_world)

## Output Consistency (calibration set 256)

| max-abs-diff | mean-abs-diff | cos |
|---|---|---|
| 0.00875 | 0.00185 | 0.999879693 |

## Collision / Success Rate

| Condition | success | collision | timeout |
|---|---|---|---|
| fp32 | 41.67% | 17.00% | 41.33% |
| int8 | 41.33% | 18.00% | 40.67% |

Net effect (post-quantization, collision→success minus success→collision): -1 episode / 300 episodes
Mean first-step max-abs action perturbation: 0.00327
