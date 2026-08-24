#!/usr/bin/env python3
"""在 Jetson 上用真实 obs 校准构建 INT8 引擎（TRT 原生 PTQ）。

校准数据由 PC 端 `python train/make_calib_data.py` 生成（buffer 采样 obs, .npz, key='obs'）。
运行前确认 python3 有 tensorrt 与 numpy：
    python3 -c "import tensorrt, numpy"

用法（任意目录均可，默认路径相对本脚本所在 models/ 目录解析）:
    python3 build_ptq_engine.py --onnx actor_fp32_bs8.onnx --calib calib_obs.npz \
        --out actor_int8_bs8.engine [--batch-size 8] [--cache calib.cache]

诊断选项:
    --verbose  打印校准回调每次调用的详细日志
"""

import argparse
import ctypes
import ctypes.util
import os
import sys
import traceback

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VERBOSE = False

CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDART = None


def _load_cudart():
    """加载 CUDA runtime，用于 cudaMalloc / cudaMemcpy / cudaFree。

    校准器的 get_batch 必须返回 device（GPU）内存指针；返回 host 指针会导致
    "illegal memory access"（TRT 把 bindings 当 CUDA 指针直接读写）。
    """
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    names = []
    found = ctypes.util.find_library("cudart")
    if found:
        names.append(found)
    names += ["libcudart.so", "libcudart.so.12", "libcudart.so.11", "libcudart.so.8"]
    for n in names:
        try:
            _CUDART = ctypes.CDLL(n)
            break
        except OSError:
            continue
    if _CUDART is None:
        raise SystemExit("[PTQ] 无法加载 libcudart（请确认 CUDA runtime 已安装）")
    _CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _CUDART.cudaMalloc.restype = ctypes.c_int
    _CUDART.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _CUDART.cudaMemcpy.restype = ctypes.c_int
    _CUDART.cudaFree.argtypes = [ctypes.c_void_p]
    _CUDART.cudaFree.restype = ctypes.c_int
    return _CUDART


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[DBG] {msg}", flush=True)


class ObsCalibrator(trt.IInt8EntropyCalibrator2):
    """从内存 obs 数组分批喂给 EntropyCalibrator2。

    注意：用 IInt8EntropyCalibrator2 作基类（tensorrt python 绑定未导出 ICalibrator）。
    """

    def __init__(self, obs: np.ndarray, batch_size: int = 8, cache: str = "calib.cache"):
        super().__init__()
        self.obs = np.ascontiguousarray(obs, dtype=np.float32)  # (N, obs_dim)
        self.batch_size = int(batch_size)
        self.pos = 0
        self.cache_path = cache
        self._cache = None
        self._last_batch = None
        self._calls = 0
        self._dev_ptr = None
        self._dev_nbytes = self.batch_size * self.obs.shape[1] * self.obs.itemsize

        cudart = _load_cudart()
        dev_ptr = ctypes.c_void_p()
        err = cudart.cudaMalloc(ctypes.byref(dev_ptr), self._dev_nbytes)
        if err != 0:
            raise SystemExit(f"[PTQ] cudaMalloc 失败: err={err}")
        self._dev_ptr = int(dev_ptr.value)

        print(
            f"[DBG] calibrator init: obs.shape={self.obs.shape} dtype={self.obs.dtype} "
            f"batch_size={self.batch_size} cache_exists={os.path.exists(cache)} "
            f"device_buf={hex(self._dev_ptr)} ({self._dev_nbytes} bytes)",
            flush=True,
        )
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                self._cache = f.read()
            log(f"read cache {len(self._cache)} bytes")

    def free(self) -> None:
        if self._dev_ptr is not None:
            cudart = _load_cudart()
            cudart.cudaFree(ctypes.c_void_p(self._dev_ptr))
            self._dev_ptr = None
            log("cudaFree device buffer")

    def get_batch_size(self) -> int:
        log(f"get_batch_size -> {self.batch_size}")
        return self.batch_size

    def get_batch(self, names):
        self._calls += 1
        log(f"get_batch #{self._calls}: names={names!r} pos={self.pos}/{len(self.obs)}")
        if self.pos >= len(self.obs):
            log("  -> None (EOF)")
            return None
        batch = self.obs[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        log(
            f"  slice shape={batch.shape} dtype={batch.dtype} "
            f"c_contig={batch.flags['C_CONTIGUOUS']}"
        )
        try:
            # 必须返回 C 连续、显式拷贝的数组（不能被 GC，见 self._last_batch）。
            arr = np.array(batch, dtype=np.float32, copy=True, order="C")
            log(
                f"  arr shape={arr.shape} dtype={arr.dtype} "
                f"c_contig={arr.flags['C_CONTIGUOUS']} id={hex(id(arr))}"
            )
        except Exception:
            print("[DBG] get_batch 内创建 numpy 数组失败:", flush=True)
            traceback.print_exc()
            return None

        # TRT 8.x python 绑定的 trampoline（pyInt8.cpp:58-64）要求 get_batch
        # 返回"每个输入的数据内存指针"的整型列表（std::vector<size_t>），
        # 它会被直接 memcpy 成 void* bindings。返回 numpy array 本身会导致
        # pybind cast numpy->size_t 失败，报
        #   "Exception caught in get_batch(): Unable to cast Python instance to C++ type"。
        #
        # 且该指针必须是 DEVICE(GPU) 内存指针：校准会真正在 GPU 上执行网络，
        # 把 host 指针塞进 bindings 会触发 "illegal memory access"。
        # 这里把 host 数据拷入预分配的 cudaMalloc 缓冲区，返回其 device 指针。
        n_in = max(1, len(names))
        self._last_batch = [arr] * n_in
        cudart = _load_cudart()
        err = cudart.cudaMemcpy(
            ctypes.c_void_p(self._dev_ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            arr.nbytes,  # 最后一批可能不足 batch_size，用实际字节数避免越界读
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        if err != 0:
            print(f"[DBG] cudaMemcpy H2D 失败: err={err}", flush=True)
            return None
        ptrs = [self._dev_ptr] * n_in
        log(f"  -> 返回 device 指针列表 len={n_in} ptrs={[hex(p) for p in ptrs]}")
        return ptrs

    def read_calibration_cache(self):
        log(f"read_calibration_cache -> {0 if self._cache is None else len(self._cache)} bytes")
        return self._cache

    def write_calibration_cache(self, buf) -> None:
        log(f"write_calibration_cache: {len(buf)} bytes -> {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            f.write(buf)


def build_engine(onnx_path: str, obs: np.ndarray, out_path: str,
                 batch_size: int = 8, cache: str = "calib.cache") -> None:
    print(f"[DBG] python: {sys.version.split()[0]}", flush=True)
    print(f"[DBG] numpy : {np.__version__}", flush=True)
    print(f"[DBG] trt   : {trt.__version__}", flush=True)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(f"[PTQ] onnx 解析失败: {onnx_path}")
    print("[DBG] onnx 解析成功", flush=True)

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = ObsCalibrator(obs, batch_size=batch_size, cache=cache)
    config.int8_calibrator = calibrator
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except AttributeError:  # TRT 老版本
        config.max_workspace_size = 1 << 30

    for i in range(network.num_inputs):
        d = network.get_input(i).shape
        print(f"[PTQ] network input '{network.get_input(i).name}': {list(d)}", flush=True)

    print("[DBG] 调用 build_serialized_network ...", flush=True)
    serialized = None
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"[PTQ] build_serialized_network 抛异常: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    print(f"[DBG] build_serialized_network 返回: {serialized!r}", flush=True)

    if serialized is None:
        print("[DBG] 尝试回退 builder.build_engine() ...", flush=True)
        try:
            serialized = builder.build_engine(network, config)
            print(f"[DBG] build_engine 返回: {serialized!r}", flush=True)
        except Exception as e2:
            print(f"[PTQ] build_engine 也抛异常: {type(e2).__name__}: {e2}", flush=True)
            traceback.print_exc()

    if serialized is None:
        calibrator.free()
        raise SystemExit(
            "[PTQ] INT8 引擎构建失败（两种 API 均返回 None/抛异常）。"
            f"校准器共被调用 get_batch {calibrator._calls} 次。"
        )

    with open(out_path, "wb") as f:
        f.write(serialized)
    print(f"[PTQ] INT8 engine saved to {out_path}")

    calibrator.free()


def main() -> None:
    ap = argparse.ArgumentParser(description="TRT INT8 真实数据校准（PTQ）")
    ap.add_argument("--onnx", default=None, help="FP32 ONNX（默认 <models>/actor_fp32_bs8.onnx）")
    ap.add_argument("--calib", default=None, help="校准数据 npz（默认 <models>/calib_obs.npz）")
    ap.add_argument("--out", default=None, help="输出 INT8 engine（默认 <models>/actor_int8_bs8.engine）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache", default=None, help="校准缓存（默认 <models>/calib.cache）")
    ap.add_argument("--verbose", action="store_true", help="打印校准回调详细日志")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    onnx_path = args.onnx or os.path.join(SCRIPT_DIR, "actor_fp32_bs8.onnx")
    calib_path = args.calib or os.path.join(SCRIPT_DIR, "calib_obs.npz")
    out_path = args.out or os.path.join(SCRIPT_DIR, "actor_int8_bs8.engine")
    cache_path = args.cache or os.path.join(SCRIPT_DIR, "calib.cache")

    if not os.path.exists(onnx_path):
        raise SystemExit(f"[PTQ] onnx 不存在: {onnx_path}")
    if not os.path.exists(calib_path):
        raise SystemExit(f"[PTQ] 校准数据不存在: {calib_path}（PC 端先跑 train/make_calib_data.py）")

    obs = np.load(calib_path)["obs"]
    print(f"[PTQ] calib obs: {obs.shape} ({obs.shape[0]} 条)")
    build_engine(onnx_path, obs, out_path, args.batch_size, cache_path)


if __name__ == "__main__":
    main()
