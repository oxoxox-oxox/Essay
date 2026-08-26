#!/usr/bin/env bash
# 消融矩阵训练：N=1（chunk=1，obs 105） + N=5（chunk=5，obs 113）
# 量化 INT8 引擎在板上构建，本脚本只负责把模型训好并落到 checkpoints/。
#
# 产物:
#   checkpoints/ppo_final_N1/{best_model,ppo_model}.zip
#   checkpoints/ppo_final_N5/{best_model,ppo_model}.zip
#   runs/ppo_final_N1/  runs/ppo_final_N5/  (evaluations.npz + tensorboard)
#
# 用法:
#   bash run_ablation_training.sh                  # 默认参数（essay_env）
#   STEPS=400000 bash run_ablation_training.sh     # 环境变量覆盖
#
# 防熵坍缩: 83ms 步长下 n_steps=1024 只覆盖 ~1-2 集易坍缩，故用 4096 + ent_coef 0.01。
# best_model 已由 SuccessRateEvalCallback 按 eval 成功率保存（train_ppo.py）。
set -euo pipefail
cd "$(dirname "$0")"

# ---- 可调参数（服务器上按需改）----
PYTHON="${PYTHON:-python}"            # essay_env 下用 python，或填绝对路径
NAME="${NAME:-ppo_final}"
STEPS="${STEPS:-300000}"
DEVICE="${DEVICE:-cuda}"
ENT_COEF="${ENT_COEF:-0.01}"
N_STEPS="${N_STEPS:-4096}"            # 83ms 步长下覆盖更多 episode，防熵坍缩

# ---- conda 激活 essay_env（PYTHON 已显式指定时跳过）----
if [[ "$PYTHON" != *python* && ! -x "$(command -v python)" ]]; then
  echo "[env] PYTHON 已指定非 python 路径，跳过 conda 激活"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  echo "[env] 已激活 $CONDA_PREFIX"
else
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ]; then
    # shellcheck disable=SC1091
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate essay_env
    echo "[env] 已激活 essay_env"
  else
    echo "[env] 未找到 conda，使用 PYTHON=$PYTHON（确保 essay_env 已激活或 PYTHON 指向其 python）"
  fi
fi

echo "=== [1/2] 训练 N=1 (chunk=1, obs 105) ==="
$PYTHON src/train_ppo.py --steps "$STEPS" --device "$DEVICE" \
    --name "$NAME" --chunk 1 --ent-coef "$ENT_COEF" --n-steps "$N_STEPS"

echo "=== [2/2] 训练 N=5 (chunk=5, obs 113) ==="
$PYTHON src/train_ppo.py --steps "$STEPS" --device "$DEVICE" \
    --name "$NAME" --chunk 5 --ent-coef "$ENT_COEF" --n-steps "$N_STEPS"

echo "=== 全部完成 ==="
ls -la checkpoints/${NAME}_N1 checkpoints/${NAME}_N5
