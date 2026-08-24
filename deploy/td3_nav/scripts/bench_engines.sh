#!/usr/bin/env bash
# 在 Jetson 上对比 actor 三种精度（INT8 PTQ / FP16 / FP32）的引擎性能（固定 batch=8）：
#   INT8  <- actor_fp32_bs8.onnx (--int8)   TRT 原生 PTQ。
#           若已用 build_ptq_engine.py 生成 actor_int8_bs8.engine（真实数据校准）则直接复用；
#           否则用 trtexec --int8 随机校准重建（仅测速，精度不可信）。
#   FP16  <- actor_fp32_bs8.onnx (--fp16)   TRT 自动转半精度
#   FP32  <- actor_fp32_bs8.onnx (--noTF32) 关 TF32 才是真 FP32 基线
#
# 用法(任意目录, 无需 cd, ONNX 默认取脚本相对目录 ../models/):
#     bash <package>/scripts/bench_engines.sh
# 可选环境变量:
#     WARMUP_MS=200 DURATION_S=5 EXTRA_FLAGS="--useCudaGraph" TRTEXEC=/path/to/trtexec
# 建议先锁性能模式让数字更稳:
#     sudo nvpmodel -m 0 && sudo jetson_clocks
#
# 说明: 引擎 batch=8 已写死（无需 --shapes），吞吐=每引擎样本数(8)的批量吞吐；
#       每样本延迟用 1000/(thr×8) 换算。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
BATCH="${BATCH:-8}"
resolve_onnx() {
  local f="$1"
  if [ -f "$f" ]; then
    echo "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
  elif [ -f "$MODELS_DIR/$f" ]; then
    echo "$MODELS_DIR/$f"
  else
    echo ""
  fi
}
ONNX_ACTOR_FP32="$(resolve_onnx "actor_fp32_bs${BATCH}.onnx")"
[ -n "$ONNX_ACTOR_FP32" ] || { echo "!! 找不到 actor FP32 onnx: 期望 $MODELS_DIR/actor_fp32_bs${BATCH}.onnx" >&2; exit 1; }
echo "ONNX (actor FP32, batch=${BATCH}) = $ONNX_ACTOR_FP32"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WARM="${WARMUP_MS:-200}"
DUR="${DURATION_S:-5}"
EXTRA="${EXTRA_FLAGS:-}"

RESULTS=()

get() { # $1=输出文本, $2=字段(thr/mean/p99)
  case "$2" in
    thr)  echo "$1" | grep -oE "Throughput: [0-9.]+"            | grep -oE "[0-9.]+$" ;;
    mean) echo "$1" | grep "GPU Compute Time" | grep -oE "mean = [0-9.]+"              | grep -oE "[0-9.]+$" ;;
    p99)  echo "$1" | grep "GPU Compute Time" | grep -oE "percentile\(99%\) = [0-9.]+" | grep -oE "[0-9.]+$" ;;
  esac
}

bench() {
  local name="$1" tag="$2" onnx="$3"; shift 3
  local eng="${name}_${tag}_bs${BATCH}.engine"
  if [ -f "$eng" ]; then
    echo "== 复用现有引擎 ${eng} =="
  else
    echo "== 构建 ${name}:${tag} =="
    if ! "$TRTEXEC" --onnx="$onnx" "$@" --saveEngine="$eng" >"$eng.build.log" 2>&1; then
      echo "  !! 构建 ${tag} 失败 (日志: $eng.build.log)"
      tail -n 20 "$eng.build.log" | sed 's/^/     /'
      rm -f "$eng.build.log"
      return
    fi
    rm -f "$eng.build.log"
  fi
  echo "== 基准 ${name}:${tag} =="
  local out t m p per
  out="$("$TRTEXEC" --loadEngine="$eng" --warmUp="$WARM" --duration="$DUR" --streams=1 $EXTRA 2>&1 \
      | grep -E "Throughput:|GPU Compute Time:" | sed -E 's/\[[^]]*\]//g')"
  echo "$out"
  t="$(get "$out" thr)"
  m="$(get "$out" mean)"
  p="$(get "$out" p99)"
  per="$(awk -v t="$t" -v b="$BATCH" 'BEGIN{printf "%.4f", 1000/(t*b)}')"
  RESULTS+=("$name|$tag|$t|$m|$p|$per")
  echo
}

echo "=== INT8 (PTQ) (actor) ==="
bench actor int8 "$ONNX_ACTOR_FP32" --int8
echo "=== FP16 (actor) ==="
bench actor fp16 "$ONNX_ACTOR_FP32" --fp16
echo "=== FP32 no TF32 (actor) ==="
bench actor fp32 "$ONNX_ACTOR_FP32" --noTF32

# ---- 汇总 ----
echo ""
echo "=== 汇总 (Throughput / GPU Compute Time / per-sample) ==="
printf "%-6s %-6s %14s %12s %10s %14s\n" "模型" "精度" "Throughput(qps)" "mean(ms)" "p99(ms)" "per-sample(ms)"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r name tag t m p per <<<"$r"
  printf "%-6s %-6s %14s %12s %10s %14s\n" "$name" "$tag" "$t" "$m" "$p" "$per"
done
