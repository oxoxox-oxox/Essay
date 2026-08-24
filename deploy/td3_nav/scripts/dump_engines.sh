#!/usr/bin/env bash
# 构建三种精度（FP32 noTF32 / FP16 / INT8）引擎并 dump 逐层精度分配 + 逐层耗时，
# 用于从根源排查 FP16 在 batch=8 下反慢的原因（README "下一步" 第 2 条）。
#
# 用法(任意目录, 无需 cd, ONNX 默认取脚本相对目录 ../models/):
#     bash <package>/scripts/dump_engines.sh
# 可选环境变量:
#     BATCH=8 WARMUP_MS=200 DURATION_S=5 TRTEXEC=/usr/src/tensorrt/bin/trtexec
#     OUT_DIR=<输出目录，默认 models/dump>
# 建议先锁性能模式让数字更稳:
#     sudo nvpmodel -m 0 && sudo jetson_clocks
#
# 产物（OUT_DIR 下）:
#     actor_<tag>_bs<BATCH>_dump.engine   详细 profiling 引擎（与 bench 用引擎名不同，不覆盖）
#     dump_<tag>.log                      完整 trtexec 输出（含 Layer Information 与 Profile）
#     dump_<tag>.clean.log                log 去除时间戳/ANSI 后的版本，方便解析
#
# 注意:
#   - 引擎必须用 --profilingVerbosity=detailed 构建才有 kernel/tactic 细节，
#     所以这里重新构建，不复用 bench 的 actor_*_bs8.engine。
#   - INT8 用 trtexec --int8 随机校准：只看结构/tactic/耗时参考，
#     不代表真实数据校准引擎（actor_int8_bs8.engine，见 build_ptq_engine.py）。
#   - 三个精度的推理实测吞吐/延迟请另跑 scripts/bench_engines.sh。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
BATCH="${BATCH:-8}"
OUT_DIR="${OUT_DIR:-$MODELS_DIR/dump}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WARM="${WARMUP_MS:-200}"
DUR="${DURATION_S:-5}"

mkdir -p "$OUT_DIR"

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
ONNX="$(resolve_onnx "actor_fp32_bs${BATCH}.onnx")"
[ -n "$ONNX" ] || { echo "!! 找不到 actor FP32 onnx: 期望 $MODELS_DIR/actor_fp32_bs${BATCH}.onnx" >&2; exit 1; }
echo "ONNX (actor FP32, batch=${BATCH}) = $ONNX"
echo "输出目录 = $OUT_DIR"
echo "TRTEXEC  = $TRTEXEC"
[ -x "$TRTEXEC" ] || echo "!! 注意: $TRTEXEC 不存在/不可执行，请设置 TRTEXEC 变量"

clean() { # $1=raw, $2=out：去 ANSI 与 [timestamp] 前缀
  sed -E 's/\x1b\[[0-9;]*[mK]//g; s/\[[0-9]{2}\/[0-9]{2}\/[0-9]{4}-[0-9:]+[0-9]\]//g' "$1" > "$2"
}

dump() { # $1=tag, $2..=trtexec 构建 flags
  local tag="$1"; shift
  local eng="$OUT_DIR/actor_${tag}_bs${BATCH}_dump.engine"
  local raw="$OUT_DIR/dump_${tag}.log"
  echo "== 构建 + dump ${tag} =="
  if ! "$TRTEXEC" --onnx="$ONNX" "$@" \
      --warmUp="$WARM" --duration="$DUR" \
      --profilingVerbosity=detailed --dumpLayerInfo --dumpProfile \
      --saveEngine="$eng" > "$raw" 2>&1; then
    echo "  !! ${tag} 构建/dump 失败 (完整日志: $raw)"
    tail -n 20 "$raw" | sed 's/^/     /'
    return 1
  fi
  clean "$raw" "$OUT_DIR/dump_${tag}.clean.log"
  echo "  engine: $eng"
  echo "  完整日志: $raw  (清理版: dump_${tag}.clean.log)"
  echo "  --- 汇总 ---"
  grep -E "Throughput:|GPU Compute Time:|Total Host Walltime" "$raw" | sed -E 's/\[[^]]*\]//g'
  echo "  --- Layer Information 行数 (层数/条目参考) ---"
  grep -cE "Layer" "$OUT_DIR/dump_${tag}.clean.log" || true
  echo ""
}

echo ""
echo "=== 1/3 FP32 (--noTF32, 真 FP32 基线) ==="
dump fp32 --noTF32
echo "=== 2/3 FP16 (--fp16) ==="
dump fp16 --fp16
echo "=== 3/3 INT8 (--int8, 随机校准, 仅结构/耗时参考) ==="
dump int8 --int8

echo ""
echo "=== 完成，产物目录: $OUT_DIR ==="
ls -lh "$OUT_DIR"
echo ""
echo "把整个 $OUT_DIR 目录（或至少 dump_fp32/dump_fp16/dump_int8 三个 log 和 clean.log）发回即可。"
