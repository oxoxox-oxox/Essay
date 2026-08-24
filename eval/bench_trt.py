"""核心验证：ONNX → TRT 三精度（FP32/FP16/INT8 PTQ）吞吐基准（PC，TRT 10.10，cuda conda env）。

用法（cuda 环境 python）:
    python eval/bench_trt.py --onnx export/mlptd3_1024_N5/actor_fp32_bs8.onnx \\
        --calib export/mlptd3_1024_N5/calib_obs.npz --batches 1,8,32

说明：
    - FP32 引擎显式 clear TF32（真 FP32 基线，对应 trtexec --noTF32）
    - INT8 用真实 obs 校准（IInt8EntropyCalibrator2，256 条）
    - 测速口径：warmup 后连续 enqueue N 次 + 一次 cudaDeviceSynchronize（含 launch 开销的端到端引擎延迟）
    - 每样本延迟 = 1000/(吞吐×batch)；多轮取中位数抑制 GPU boost 噪声
    - 同时输出 INT8/FP16 vs FP32 的输出精度对比（max-abs-diff / 余弦相似度）
"""
import argparse
import ctypes
import ctypes.util
import os
import statistics
import time

import numpy as np
import tensorrt as trt

# --------------------------------------------------------------------------- #
# cudart（来自学习笔记 12.1）
# --------------------------------------------------------------------------- #
def load_cudart():
    # 先按裸名尝试（PATH 有效时）；Python 3.8+ ctypes 常搜不到 PATH 里的 DLL，需全路径兜底
    for n in [ctypes.util.find_library("cudart"), "cudart64_12.dll", "cudart64_13.dll"]:
        if not n:
            continue
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    import glob

    for pat in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\cudart64*.dll",
        r"D:\TensorRT\TensorRT-*\lib\cudart64*.dll",
    ]:
        for path in glob.glob(pat):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise SystemExit("无法加载 cudart（未找到 cudart64*.dll，请确认 CUDA Toolkit 已安装）")


lib = load_cudart()
lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cudaMalloc.restype = ctypes.c_int
lib.cudaFree.argtypes = [ctypes.c_void_p]
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
lib.cudaDeviceSynchronize.restype = ctypes.c_int
H2D, D2H = 1, 2
logger = trt.Logger(trt.Logger.WARNING)


def cuda_alloc(nbytes):
    p = ctypes.c_void_p()
    lib.cudaMalloc(ctypes.byref(p), nbytes)
    return int(p.value)


def cuda_memcpy_h2d(dst, arr):
    return lib.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, H2D)


def cuda_memcpy_d2h(arr, src):
    return lib.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), arr.nbytes, D2H)


# --------------------------------------------------------------------------- #
# 建引擎（来自学习笔记 12.2）+ 校准器（4.2）
# --------------------------------------------------------------------------- #
class ObsCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, obs: np.ndarray, batch_size: int):
        super().__init__()
        self.obs = np.ascontiguousarray(obs, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.pos = 0
        self._dev_ptr = cuda_alloc(self.batch_size * self.obs.shape[1] * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.pos >= len(self.obs):
            return None
        arr = np.array(
            self.obs[self.pos : self.pos + self.batch_size], dtype=np.float32, copy=True, order="C"
        )
        self.pos += self.batch_size
        cuda_memcpy_h2d(self._dev_ptr, arr)
        return [self._dev_ptr] * max(1, len(names))

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, buf):
        pass


def build(onnx_path: str, precision: str, calibrator=None) -> bytes:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(f"ONNX 解析失败: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)  # 真 FP32（对应 --noTF32）
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    else:
        raise ValueError(precision)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit(f"建引擎失败: {precision}")
    return bytes(ser)


# --------------------------------------------------------------------------- #
# Runner（来自学习笔记 4.3）
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, engine_bytes: bytes):
        rt = trt.Runtime(logger)
        self.engine = rt.deserialize_cuda_engine(engine_bytes)
        self.ctx = self.engine.create_execution_context()
        self.dev, self.host = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            npdtype = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF: np.float16,
                trt.DataType.INT8: np.int8,
                trt.DataType.INT32: np.int32,
            }[dtype]
            self.host[name] = np.zeros(shape, dtype=npdtype)
            self.dev[name] = cuda_alloc(self.host[name].nbytes)
            self.ctx.set_tensor_address(name, self.dev[name])
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.binds = [self.dev[self.engine.get_tensor_name(i)] for i in range(self.engine.num_io_tensors)]
        self.batch = shape[0] if len(shape) > 0 else 1

    def infer(self, x: np.ndarray) -> np.ndarray:
        self.host[self.in_name][:] = x
        cuda_memcpy_h2d(self.dev[self.in_name], self.host[self.in_name])
        self.ctx.execute_v2(self.binds)
        cuda_memcpy_d2h(self.host[self.out_name], self.dev[self.out_name])
        return self.host[self.out_name].copy()

    def bench(self, x: np.ndarray, iters: int, rounds: int) -> dict:
        for _ in range(min(100, iters)):  # warmup
            self.ctx.execute_v2(self.binds)
        lib.cudaDeviceSynchronize()
        per_round = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                self.ctx.execute_v2(self.binds)
            lib.cudaDeviceSynchronize()
            dt = (time.perf_counter() - t0) / iters * 1000.0  # ms / 次推理
            per_round.append(dt)
        med = statistics.median(per_round)
        return {
            "rounds_ms": per_round,
            "mean_ms": statistics.mean(per_round),
            "median_ms": med,
            "throughput_qps": 1000.0 / med,
            "per_sample_ms": med / self.batch,
        }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="TRT 三精度吞吐基准（PC / TRT 10.10）")
    ap.add_argument("--onnx", required=True, help="ONNX 基底路径（不含 _bsN 后缀，脚本自动拼 _bs{b}.onnx）")
    ap.add_argument("--calib", required=True, help="INT8 校准 obs (.npz, key='obs')")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--precisions", default="fp32,fp16,int8")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--acc-obs", type=int, default=64, help="精度对比用 obs 条数")
    args = ap.parse_args()

    obs = np.load(args.calib)["obs"].astype(np.float32)  # (N, 113)
    batches = [int(b) for b in args.batches.split(",")]
    precisions = args.precisions.split(",")

    # 精度对比用固定输入（真实 obs 前 acc-obs 条，避免与校准完全同分布的重叠顾虑）
    acc_in = np.ascontiguousarray(obs[: args.acc_obs], dtype=np.float32)

    print(f"onnx={args.onnx}")
    print(f"calib obs: {obs.shape}  batches={batches}  precisions={precisions}\n")

    results = {}  # (batch, prec) -> dict
    outputs = {}  # (batch, prec) -> np.ndarray
    engines = {}  # (batch, prec) -> engine bytes

    for b in batches:
        # 同一模型各 batch 的 ONNX 是分开导出的：基底路径 + _bs{b}.onnx
        onnx_b = f"{args.onnx}_bs{b}.onnx"
        if not os.path.exists(onnx_b):
            print(f"[skip] 缺少 {onnx_b}（用 export_onnx.py --batch {b} 导出）")
            continue
        print(f"===== batch={b} =====")
        for prec in precisions:
            calib = ObsCalibrator(obs, b) if prec == "int8" else None
            t0 = time.perf_counter()
            eng = build(onnx_b, prec, calib)
            dt_build = time.perf_counter() - t0
            engines[(b, prec)] = eng
            run = Runner(eng)
            x = np.ascontiguousarray(acc_in[:b], dtype=np.float32)
            out = run.infer(x)
            m = run.bench(x, args.iters, args.rounds)
            results[(b, prec)] = m
            outputs[(b, prec)] = out
            print(
                f"  {prec:5s} build={dt_build:5.1f}s  median={m['median_ms']*1000:7.1f}us  "
                f"thr={m['throughput_qps']:9.0f} qps  per-sample={m['per_sample_ms']*1000:7.2f}us"
            )

    # ---- 汇总表 ----
    print("\n===== 汇总（per-sample = 1000/(thr×batch)，单位 us；INT8/FP16 vs FP32 为每样本延迟变化率，负=更快） =====")
    header = "batch | " + " | ".join(f"{p:>10s}" for p in precisions) + " | INT8 vs FP32 | FP16 vs FP32"
    print(header)
    print("-" * len(header))
    for b in batches:
        if (b, "fp32") not in results:
            continue
        cells = []
        for p in precisions:
            if (b, p) in results:
                cells.append(f"{results[(b, p)]['per_sample_ms']*1000:8.2f}us")
            else:
                cells.append(f"{'-':>10s}")
        r8 = results[(b, "int8")]["per_sample_ms"] / results[(b, "fp32")]["per_sample_ms"]
        r16 = results[(b, "fp16")]["per_sample_ms"] / results[(b, "fp32")]["per_sample_ms"]
        print(f"{b:5d} | " + " | ".join(cells) + f" | {(r8-1):+8.1%} | {(r16-1):+8.1%}")

    # ---- 精度对比（INT8/FP16 vs FP32 输出；用随机 obs，避免与校准数据循环论证） ----
    print("\n===== 输出精度对比（64 条随机 obs，非校准数据） =====")
    rng = np.random.default_rng(0)
    acc_x = rng.uniform(0.0, 1.0, size=(args.acc_obs, obs.shape[1])).astype(np.float32)
    b = batches[-1]  # 用最大 batch 的引擎
    if (b, "fp32") in engines:
        out = {}
        for p in precisions:
            if (b, p) not in engines:
                continue
            run = Runner(engines[(b, p)])
            chunks = [run.infer(acc_x[i : i + b]) for i in range(0, len(acc_x), b)]
            out[p] = np.concatenate(chunks)[: len(acc_x)]
        for p in ("fp16", "int8"):
            if p not in out:
                continue
            a, c = out["fp32"], out[p]
            mae = float(np.max(np.abs(a - c)))
            cos = float(
                np.dot(a.ravel(), c.ravel())
                / (np.linalg.norm(a.ravel()) * np.linalg.norm(c.ravel()) + 1e-12)
            )
            print(f"  batch={b} {p:5s} vs FP32: max-abs-diff={mae:.6f}  cos={cos:.6f}")


if __name__ == "__main__":
    main()
