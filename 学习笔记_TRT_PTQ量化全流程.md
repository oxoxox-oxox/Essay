# TensorRT PTQ 量化全流程 · 学习笔记

> 本笔记把「在 PC 上验证 ONNX → TensorRT INT8 PTQ 量化可行性」这一整套工作完整拆解，目标是让你**只看这一份就能复现 + 理解**。
> 配套仓库（当前版本）：`export/ppo_mw/build_ptq_engine.py`（板上建 INT8 引擎脚本）、`train/export_ppo_onnx.py`（导出 ONNX）。
> 本笔记最初随已移除的 TD3 流水线编写（当时配套 `deploy/td3_nav/`、`train/export_onnx.py`），脚本位置已更新为 PPO 版。
> 实测环境：Windows + conda env `cuda` + TensorRT 10.10.0.31 + CUDA Toolkit 12.8 + RTX 5070（Blackwell sm_120）。

---

## 目录

1. [背景：一条推理部署链路](#1-背景一条推理部署链路)
2. [三种精度与 PTQ 校准](#2-三种精度与-ptq-校准)
3. [工具链与环境](#3-工具链与环境)
4. [完整 PTQ 流程（最小闭环）](#4-完整-ptq-流程最小闭环)
5. [踩坑清单（详解）](#5-踩坑清单详解)
6. [核心概念](#6-核心概念)
7. [诊断与验证方法](#7-诊断与验证方法)
8. [实测结果解读（案例）](#8-实测结果解读案例)
9. [测量噪声与定性定量](#9-测量噪声与定性定量)
10. [方法论十条](#10-方法论十条)
11. [继续学习路径](#11-继续学习路径)
12. [附录：可复用代码片段](#12-附录可复用代码片段)

---

## 1. 背景：一条推理部署链路

### 1.1 链路全景

模型要上机器人，不会直接拿 PyTorch 跑，标准链路是：

```
PyTorch checkpoint (.pt)
        │  torch.onnx.export
        ▼
ONNX（中间表示：算子 + 权重 + 图连接）
        │  TensorRT OnnxParser + build
        ▼
TensorRT 引擎 (.engine，针对某块 GPU 优化过的二进制)
```

三个环节各是什么：

- **ONNX**：跨框架、跨硬件的"模型描述文件"。里面是算子（Gemm/Relu/Tanh）、权重（initializer）、图结构。它不依赖 PyTorch 或 TensorRT。
- **TensorRT（TRT）**：NVIDIA 的推理加速器。拿到 ONNX 后做四件事：
  1. **算子融合**：把 Conv+BN+Relu 合成一个 kernel，减少显存读写次数。
  2. **精度选择**：哪一层用 FP32 / FP16 / INT8。
  3. **kernel（tactic）选择**：同一个 Gemm 有几十种 CUDA 实现，TRT 现场实测挑最快的。
  4. **内存布局优化**：把数据排成 GPU 喜欢的格式。
- **.engine 文件**：和 **TRT 版本 + GPU 架构强绑定**。PC 上建的引擎**不能搬到 Jetson 用**，换板子/换 TRT 版本必须重建。这是最容易被忽略的一条。

### 1.2 为什么 ONNX 这一步要"纯 FP32"

本项目刻意导出**无量化信息的纯 FP32 ONNX**，然后让 TensorRT 在板上自己降精度。原因：如果导出时就带量化节点（QAT/QDQ），TRT 会把它当成独立算子处理，反而可能拆坏融合、插入多余格式转换（旧版实测 INT8 反而慢 2.3×）。结论是：**量化的决定权交给 TRT，导出一个"干净"的 FP32 模型即可**。

---

## 2. 三种精度与 PTQ 校准

### 2.1 精度对比

| 精度 | 每个数位数 | 需要校准吗 | 精度损失 | 速度 |
|---|---|---|---|---|
| FP32 | 32 bit | 否 | 基线 | 基准 |
| FP16 | 16 bit | 否 | 几乎无损 | 更快（Tensor Core） |
| INT8 | 8 bit | **需要校准** | 可能有损 | 最快 |

### 2.2 TF32 是个坑

Ampere 及之后的 GPU 上，**FP32 默认其实会用 TF32**（19 位的"伪 FP32"）走 Tensor Core，比真 FP32 快但精度略低。所以测"真 FP32 基线"必须显式关掉：

```python
config.clear_flag(trt.BuilderFlag.TF32)   # 对应 trtexec 的 --noTF32
```

否则你的 FP32 基线被偷偷加速，和 INT8 的对比就不公平了。

### 2.3 为什么 INT8 需要"校准"（PTQ 的核心）

- **权重（weight）是静态的**：训练完就固定，可以直接算出量化范围。
- **激活值（activation）是动态的**：每层输出多大，取决于输入数据分布，你不知道。
- **校准（calibration）** 就是：喂一批**有代表性的输入** → 统计每层激活的数值范围 → 算出 scale（例如 `scale = max_abs / 127`）→ 推理时把浮点缩放成 INT8。

这就是 **PTQ（Post-Training Quantization，训练后量化）**：训练完不动权重，只用少量数据校准激活范围。与之相对的是 **QAT（训练中量化）**，训练时就让模型适应量化，精度更高但成本大。

### 2.4 校准算法

- **Entropy（KL 散度）**：找"信息损失最小"的 scale，默认最稳，本项目用它。
- **MinMax**：直接取 min/max，简单，但怕分布里的离群值（一个极大值就把范围拉大，浪费精度）。

校准数据必须**代表真实部署分布**（本项目从 replay buffer 采样真实 obs，而不是用随机高斯噪声），这是精度不掉的根基。

---

## 3. 工具链与环境

### 3.1 conda 环境

- 环境名 `cuda`，Python 3.10。
- 关键包：`tensorrt 10.10.0.31`、`onnx 1.22`、`numpy 2.2`、`torch 2.13.0+cpu`。
- ⚠️ **torch 是 CPU-only**（`torch.cuda.is_available() == False`）。这决定了后面校准器**不能用 torch 的 GPU 张量**，要用 ctypes 直接调 CUDA runtime。

### 3.2 TensorRT 的安装形态（Windows）

pip 装的是 **Python 绑定**（`tensorrt.cp310-win_amd64.pyd`），真正的库是 DLL：

```
D:\TensorRT\TensorRT-10.10.0.31\lib\nvinfer_10.dll        # 核心库
D:\TensorRT\TensorRT-10.10.0.31\lib\nvinfer_plugin_10.dll # 插件
D:\TensorRT\TensorRT-10.10.0.31\lib\nvonnxparser_10.dll  # ONNX 解析器
D:\TensorRT\TensorRT-10.10.0.31\bin\trtexec.exe          # 命令行基准工具
```

**注意 TRT 10 的 DLL 带 `_10` 后缀**（老版本是 `nvinfer.dll`）。`import tensorrt` 能成功，是因为这些目录在 PATH 里。

### 3.3 用 ctypes 调 CUDA runtime

没有 torch-GPU / pycuda 时，用 ctypes 直接加载 CUDA 运行时库：

```python
import ctypes, ctypes.util
lib = ctypes.CDLL(ctypes.util.find_library("cudart") or "cudart64_12.dll")
lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cudaMalloc.restype = ctypes.c_int
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
lib.cudaFree.argtypes = [ctypes.c_void_p]
lib.cudaDeviceSynchronize.restype = ctypes.c_int
```

几个核心 API：
- `cudaMalloc(&ptr, size)`：在 GPU 显存上分配 `size` 字节，返回 device 指针。
- `cudaMemcpy(dst, src, size, kind)`：拷数据，`kind=1` 是 host→device（H2D），`kind=2` 是 device→host（D2H）。
- `cudaFree(ptr)`：释放显存。
- `cudaDeviceSynchronize()`：等 GPU 把队列里的活干完（测速必须用它）。

---

## 4. 完整 PTQ 流程（最小闭环）

> 核心方法论：**先跑通最小闭环（建引擎→跑推理→比输出→测速），再谈优化**。

### 4.1 建引擎

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# EXPLICIT_BATCH：显式 batch 维度（现代标准写法）
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)
with open("actor.onnx", "rb") as f:
    ok = parser.parse(f.read())          # 失败时用 parser.get_error(i) 看原因

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB 工作区上限

config.clear_flag(trt.BuilderFlag.TF32)   # 真 FP32 基线
config.set_flag(trt.BuilderFlag.FP16)     # FP16
config.set_flag(trt.BuilderFlag.INT8)     # INT8
config.int8_calibrator = calibrator       # INT8 必须挂校准器

serialized = builder.build_serialized_network(network, config)
engine_bytes = bytes(serialized)          # TRT 10 返回 IHostMemory，不是 bytes！
with open("x.engine", "wb") as f:
    f.write(engine_bytes)
```

### 4.2 校准器（最关键的坑）

```python
class ObsCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, obs, batch_size):
        super().__init__()
        self.obs = np.ascontiguousarray(obs, dtype=np.float32)   # (N, obs_dim)
        self.batch_size = batch_size
        self.pos = 0
        # 预先在 GPU 上分配一个 batch 的缓冲
        self._dev_ptr = cuda_alloc(self.batch_size * self.obs.shape[1] * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.pos >= len(self.obs):
            return None                 # 返回 None = 数据喂完了
        arr = np.array(self.obs[self.pos:self.pos + self.batch_size],
                       dtype=np.float32, copy=True, order="C")
        self.pos += self.batch_size
        cuda_memcpy_h2d(self._dev_ptr, arr)      # 把 host 数据拷到 GPU
        return [self._dev_ptr] * max(1, len(names))  # 返回 device 指针的 int 列表

    def read_calibration_cache(self):
        return None                     # 没有缓存就返回 None

    def write_calibration_cache(self, buf):
        with open("calib.cache", "wb") as f:
            f.write(buf)                # 校准结果缓存，下次可复用
```

**为什么 get_batch 必须返回 device 指针**：TRT 校准时**真的在 GPU 上跑网络**，把你返回的指针当 CUDA 指针直接读。返回 host（CPU）指针 → 它去 GPU 地址空间读 → `illegal memory access` 崩溃。这是本项目踩过的最疼的坑（详见 `5）。

### 4.3 推理

```python
class Runner:
    def __init__(self, engine_path):
        rt = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.dev, self.host = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            npdtype = {trt.DataType.FLOAT: np.float32,
                       trt.DataType.HALF: np.float16,
                       trt.DataType.INT8: np.int8,
                       trt.DataType.INT32: np.int32}[dtype]
            self.host[name] = np.zeros(shape, dtype=npdtype)
            self.dev[name] = cuda_alloc(self.host[name].nbytes)
            self.ctx.set_tensor_address(name, self.dev[name])   # 告诉引擎数据在显存哪

    def run(self, x):
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        self.host[in_name][:] = x
        cuda_memcpy_h2d(self.dev[in_name], self.host[in_name])  # 输入拷进 GPU
        binds = [self.dev[self.engine.get_tensor_name(i)]
                 for i in range(self.engine.num_io_tensors)]
        self.ctx.execute_v2(binds)                              # 真正执行
        cuda_memcpy_d2h(self.host[out_name], self.dev[out_name]) # 输出拷回 CPU
        return self.host[out_name].copy()
```

**要点**：数据必须在显存。所以推理前 H2D、推理后 D2H。这是 CUDA 编程的基本模型——CPU 和 GPU 各有独立内存空间，要显式搬运。

### 4.4 测速（正确姿势）

```python
def bench(self, x, warmup=100, iters=500):
    in_name = self.engine.get_tensor_name(0)
    self.host[in_name][:] = x
    cuda_memcpy_h2d(self.dev[in_name], self.host[in_name])
    binds = [...]
    for _ in range(warmup):
        self.ctx.execute_v2(binds)        # 预热：让 GPU 上电、缓存就位
    lib.cudaDeviceSynchronize()           # 等 GPU 干完，再开始计时
    t0 = time.perf_counter()
    for _ in range(iters):
        self.ctx.execute_v2(binds)
    lib.cudaDeviceSynchronize()           # 等 GPU 干完，再停止计时
    return (time.perf_counter() - t0) / iters * 1000.0   # 单位 ms
```

四个要点：**warmup**（否则第一次异常慢）、**synchronize**（否则计的是"塞进队列"的时间）、**多次取平均**、然后算：

- **throughput（吞吐）** = `1000 / latency_ms × batch`（每秒多少样本）
- **per-sample latency（每样本延迟）** = `latency_ms / batch`

---

## 5. 踩坑清单（详解）

| # | 坑 | 机理 | 规避 |
|---|---|---|---|
| 1 | 校准器返回 host 指针 | TRT 把它当 CUDA 指针用 | `cudaMalloc` + `cudaMemcpy`，返回 device 指针 int |
| 2 | FP32 基线被 TF32 污染 | Ampere+ 默认开 TF32 | `clear_flag(TF32)` / `--noTF32` |
| 3 | TRT 10 返回 `IHostMemory` 非 bytes | API 变更 | `bytes(serialized)` |
| 4 | dtype 枚举 vs 字符串 | `str(DataType.FLOAT)` = `'DataType.FLOAT'` | 用 `trt.DataType.FLOAT` 做字典 key |
| 5 | 旧校准 API deprecated | TRT 10.1 起废弃 | 还能用，但留意未来迁移到显式量化 Q/DQ |
| 6 | 微秒级测速噪声 | launch-bound + WDDM + GPU boost | 定性结论为主，正式测速用 trtexec + 锁频 |
| 7 | `model.pt` vs `model_best.pt` | 训练不同阶段 | 部署锁定 + cos 相似度校验 |
| 8 | `conda run` 的 GBK 编码崩溃 | 脚本打印非 GBK 字符 | 用 env python 直跑 + `PYTHONIOENCODING=utf-8` |

**逐个展开：**

**坑 1（最重要）**：详见 `4.2。一句话：TRT 把 `get_batch` 返回值当成 GPU 指针整数列表直接 `memcpy` 成 `void* bindings`，host 指针会触发 `illegal memory access`。

**坑 3**：老版本 `build_serialized_network` 返回 `bytes`，TRT 10 返回 `IHostMemory` 对象。`len()` 会报 `object has no len()`，要 `bytes(ser)` 转换（或 `memoryview(ser).tobytes()`）。

**坑 4**：`engine.get_tensor_dtype()` 返回枚举 `trt.DataType.FLOAT`，它的 `str()` 是 `'DataType.FLOAT'` 而不是 `'float32'`。所以不能用字符串映射，要用枚举做 key。

**坑 5**：TRT 10.1 起 `IInt8EntropyCalibrator2` + `config.int8_calibrator` 这套"经典校准"被标记 `DeprecationWarning`，官方推荐"显式量化"（在 ONNX 里加 Q/DQ 节点）。但 **deprecated ≠ 删除**，10.10 里照常能用。工程常识：废弃 API 往往还能活好几个大版本，但要心里有数它终将被移除。

**坑 6**：详见 `9。

**坑 7**：`model.pt` 是训练最后一个 epoch 的权重，`model_best.pt` 是训练中峰值（最好）的权重，两者是**不同训练阶段的模型**。本项目部署用的 bs8/bs1 全部来自 `model_best.pt`，但导出脚本默认用 `model.pt`——一旦混用，权重就不一致了。这是被"一致性校验"（`7.1）抓出来的。

**坑 8**：`conda run -n env python xxx.py` 会拦截子进程输出并重新打印到控制台，Windows 上控制台默认 GBK 编码，脚本若打印了 GBK 无法编码的字符（如某些 ONNX 图内容、特殊符号），conda 自己先崩溃。解法：直接调 `D:\anaconda\envs\cuda\python.exe` 并加 `-X utf8`。

---

## 6. 核心概念

### 6.1 launch-bound（启动受限）

这是本次所有"量化无收益"结论的**根因**。

- 模型极小（约 20 万 MACs），单次推理 0.05–0.08 ms。
- GPU 启动一个 kernel、把任务塞进队列的固定开销本身就约 10–30 µs。
- **砍计算量（量化）没用，因为瓶颈不是计算，是启动开销。**

类比：送 5 块钱外卖，骑手取餐的固定成本比饭钱还高，你把做饭做快没用。

**判断方法**：batch=1 时三档精度打平、甚至 FP32 略快 → 就是 launch-bound 的典型特征。

### 6.2 Reformat cast（格式转换层）与算子融合

- INT8 和 FP32 交界处，数据要从一种格式转成另一种，这个"转换层"本身耗时间。
- 模型小、转换层多时，省下的计算时间全被转换吃掉了。
- README 里旧 CNN 结构 INT8 反而慢 2.3×，就是这个机理：卷积层被迫保 FP32，FC 层的收益被 FP32↔INT8 边界的 Reformat cast 抵消。
- **这正是 v2 改成纯 MLP（全图 GEMM）的原因**：GEMM 能整体降精度，少插转换层。

可以用 Engine Inspector（`7.4）直接看到 INT8 引擎里的 `Reformatting CopyNode` 层。

### 6.3 batch 才是杠杆

batch 从 1→8→32，每样本延迟降了约 27 倍。原因：**一次 kernel 启动的开销被摊到 32 个样本上**。

但注意矛盾点：**ROS 实时部署恒为 batch=1**（实时单步推理，不能等攒够 8 帧）。所以"batch 提速"只对吞吐基准有意义，对实时导航没意义。真正的实时杠杆是 **action chunking**（见 `8.4）。

### 6.4 量化误差的度量

比较两个输出（比如 INT8 vs FP32）的差异，常用三个指标：

- **max abs（最大绝对误差）**：`np.abs(y8 - y32).max()`，最坏情况。
- **mean abs（平均绝对误差）**：`np.abs(y8 - y32).mean()`，平均情况。
- **cosine similarity（余弦相似度）**：方向一致性，越接近 1 越像。

```python
def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
```

**cos 还有个妙用**：当两个模型输出 `cos ≈ 1.0`（如 1.000000000），说明它们是**同一套权重**——这是快速做"权重同源校验"的利器。

---

## 7. 诊断与验证方法

### 7.1 权重同源校验（cos 相似度）

跨 batch 对比前，先确认"比的是同一个东西"。做法：把 A 模型和 B 模型跑同一批输入，算输出 cos。`cos=1.0` → 同权重；`cos=0.56` → 权重完全不同。

本项目实例：导出 bs32 后，发现 bs32 和部署 bs8 输出 cos=0.56 → 追查发现 bs8 来自 `model_best.pt`、bs32 默认来自 `model.pt` → 改用 `model_best.pt` 重导后 cos=1.000000000。

### 7.2 checkpoint 版本管理

- 部署要**显式锁定**用哪个 checkpoint（`model.pt` / `model_best.pt`），别依赖脚本默认值。
- 导出命令显式传参：`--ckpt checkpoints/mlptd3_N5/model_best.pt`。

### 7.3 ONNX 图对比

要判断"两个 ONNX 是否等价"，别肉眼看文件，用 onnx 库结构化解包：

```python
import onnx, hashlib
from onnx import numpy_helper, helper

m = onnx.load(path)
# 比：ir_version、opset、producer、节点序列、节点属性、初始值器(权重)
for n in m.graph.node:
    print(n.op_type, n.name, list(n.input), list(n.output))
    for a in n.attribute:
        print("  ", a.name, helper.get_attribute_value(a))
for it in m.graph.initializer:
    arr = numpy_helper.to_array(it)
    print(it.name, arr.shape, arr.dtype,
          hashlib.sha256(arr.tobytes()).hexdigest()[:16])  # 权重指纹
```

本项目结论：部署 bs8 和重导 bs8 **节点/属性/权重完全一致**，唯一差异是 `producer` 字段（`pytorch 2.11.0` vs `2.13.0`）——说明图结构没变，之前测速差异不是图差异造成的。

### 7.4 Engine Inspector（TRT 10 诊断利器）

看引擎里每层叫什么、融了谁、精度是什么、哪些是格式转换层：

```python
insp = engine.create_engine_inspector()
# 层名列表（JSON 格式只有层名）
names = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
# 每层详细信息（ONELINE 格式含融合信息）
for i in range(len(names)):
    print(insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
```

本项目 INT8 引擎的 8 层（注意层名里的信息）：

```
reshape_before_/net/net.0/Gemm
/net/net.0/Gemm + (Unnamed Layer* 4) [ElementWise] + /net/net.1/Relu
Reformatting CopyNode for Input Tensor 0 to /net/net.2/Gemm + ... [ElementWise] + /net/net.3/Relu
/net/net.2/Gemm + (Unnamed Layer* 10) [ElementWise] + /net/net.3/Relu
Reformatting CopyNode for Input Tensor 0 to /head/Gemm + (Unnamed Layer* 16) [ElementWise]
/head/Gemm + (Unnamed Layer* 16) [ElementWise]
PWN(/Tanh)
squeeze_after_/Tanh + /Reshape
```

读法：
- `Gemm + ElementWise + Relu` = 三个算子被**融合**进一层了（省读写）。
- `Reformatting CopyNode` = **格式转换层**（INT8 精度带来的额外开销，量化"无收益"的显微镜证据）。
- `reshape_before_...` / `squeeze_after_...` = 输入/输出的形状调整层。
- 而 FP32 引擎只有 4 层（3×Gemm + Tanh），Reshape/Constant 被折叠掉了。

---

## 8. 实测结果解读（案例）

### 8.1 量化精度损失（256 条真实 obs，输出 tanh ∈ [-1,1]）

| 对比 | max abs | mean abs | cos |
|---|---|---|---|
| INT8 vs FP32 | ~5e-4 | ~5e-5 | **0.99999999** |
| FP16 vs FP32 | ~1e-3 | ~1e-4 | 0.99999997 |

→ 对这个 MLP 模型，**INT8 PTQ 精度损失可忽略**（相对误差约 0.01%）。

### 8.2 速度（RTX 5070，注意是 PC 不是板子）

| batch | FP32 每样本 | INT8 每样本 | INT8 vs FP32 |
|---|---|---|---|
| 1 | ~0.065 ms | ~0.052 ms | 噪声区间 |
| 8 | ~0.0099 ms | ~0.0074 ms | +10~35%（噪声大） |
| 32 | ~0.0024 ms | ~0.0019 ms | +20% 左右 |

**解读**：
1. 每样本延迟随 batch 单调下降（bs1→bs32 约 27 倍），这是稳定复现的定性结论。
2. INT8 相对 FP32 的收益只有 10–35%，且微秒级测量噪声大，**不能当精确数字**。
3. batch=1（实时场景）下三精度打平——**量化对实时导航没帮助**。

### 8.3 结论的"可迁移性"

- **精度侧**：可参考。量化误差主要由模型结构决定，与 GPU 架构关系不大。
- **速度侧**：不可直接迁移。PC（Blackwell sm_120 / TRT 10.10）与板子（Orin Nano Ampere sm_87 / TRT 8.5）绝对数字、引擎文件都不同，必须板上重建。
- **定性规律**：可参考。小模型 launch-bound、batch 是杠杆、INT8 收益有限——这些大概率在板子上同样成立。

### 8.4 与 action chunking 的关系

> README 的核心叙事：「量化降低单次推理延迟」×「action chunking 降低推理频率」双正交手段。

关键理解：
- **量化**：让"每次推理更快"。但对 tiny 模型基本无效（launch-bound）。
- **chunking（N=5）**：一次输出 5 步动作，开环执行，**推理频率降为 1/5**。这才是实打实的延时/频率收益。
- 代价：chunk 越长，开环越久，闭环质量（成功率/轨迹误差）可能退化。

所以真正有效的杠杆是 **chunking**，量化只是锦上添花（且常常添不上）。

---

## 9. 测量噪声与定性定量

### 9.1 为什么两次测的结果不一样

同模型两次测，INT8 收益从 +12.5% 跳到 +34%，甚至 bs1 从"更慢"变"更快"。三个噪声源：

1. **桌面共享 GPU（WDDM）**：浏览器、桌面合成都在抢 GPU，时钟在 boost/降频间跳。
2. **基准参数不同**：warmup 50 vs 100、iters 300 vs 500。
3. **构建顺序效应**：INT8 构建要 25 秒（重负载），紧接着测速时 GPU 还在 boost 态，数字偏快。

### 9.2 两条铁律

1. **百分比是"两个噪声数的比值"**，噪声会被放大，别当真。
2. **结论分两层**：
   - 定性结论（batch↑→延迟↓、INT8 精度无损）→ 稳定复现，可信；
   - 定量结论（INT8 快 12% 还是 34%）→ 噪声内，不可信。

### 9.3 正式测速怎么做

- 用 `trtexec`（TRT 自带命令行基准）固定 `--warmup`、`--duration`、`--streams`。
- 板上锁频（`jetson_clocks` / `nvpmodel`）。
- 多轮取**中位数**而不是均值（抗偶发慢帧）。

---

## 10. 方法论十条

1. **最小闭环优先**：先跑通"建引擎→跑推理"的 20 行，再谈优化。
2. **只读侦察 → 再动手**：先搞清楚环境有什么、文档怎么写、工具链在哪。
3. **假设驱动**：数字对不上时，先列假设（权重不同？图不同？噪声？），逐条取证排除。
4. **负结果也是结果**：INT8 无收益这个"没用的结论"反而最有信息量——它指向 launch 开销，而非计算。
5. **定性 vs 定量分开**：稳定趋势才下结论，噪声里的百分比别当真。
6. **同源校验**：跨实验对比前，先 cos=1.0 确认比的是同一个东西。
7. **锁版本**：部署必须显式指定 checkpoint、TRT 版本、导出参数，别依赖默认值。
8. **deprecated ≠ 删除**：废弃 API 还能用，但要留意迁移方向。
9. **理解边界**：引擎文件、绝对速度都绑硬件/版本，别跨机迁移。
10. **精度与延迟成对记录**：单看任何一边都会得出错误结论。

---

## 11. 继续学习路径

1. **动手跑一遍**：把 `export/ppo_mw/build_ptq_engine.py` 在 PC 上跑通，逐行改、看报错。
2. **读官方文档**：NVIDIA TensorRT Developer Guide 的 "Working with INT8" 章节（校准器/scale 的权威来源）。
3. **看懂 Engine Inspector**：用 `7.4 的代码打印你自己的引擎，读懂"融合"和"Reformat cast"。
4. **补 CUDA 基础**：host/device 内存、`cudaMemcpy`、kernel launch 开销——理解 launch-bound 的前提。
5. **做一次标准测速**：学 `trtexec` 参数，体会它和"手写计时"的区别。
6. **理解 PTQ 细节**：scale 怎么从校准数据算出来、Entropy 算法在做什么、为什么校准数据要代表真实分布。

---

## 12. 附录：可复用代码片段

### 12.1 cudart 加载 + 显存操作

```python
import ctypes, ctypes.util

def load_cudart():
    names = [ctypes.util.find_library("cudart"), "cudart64_12.dll", "cudart64_13.dll"]
    for n in names:
        if not n: continue
        try:
            lib = ctypes.CDLL(n); break
        except OSError: pass
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    return lib

lib = load_cudart()
H2D, D2H = 1, 2

def cuda_alloc(nbytes):
    p = ctypes.c_void_p()
    lib.cudaMalloc(ctypes.byref(p), nbytes)
    return int(p.value)

def cuda_memcpy_h2d(dst, arr):
    return lib.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, H2D)

def cuda_memcpy_d2h(arr, src):
    return lib.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), arr.nbytes, D2H)
```

### 12.2 建引擎（三种精度）

```python
def build(onnx_path, precision, calibrator=None):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    ser = builder.build_serialized_network(network, config)
    return bytes(ser)
```

### 12.3 余弦相似度

```python
def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
```

### 12.4 引擎逐层信息

```python
import json
insp = engine.create_engine_inspector()
layers = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
for i in range(len(layers)):
    print(insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
```

---

> 一句话总结这份笔记：**一条链路（pt→ONNX→engine）、一个核心（PTQ 校准）、一个坑（device 指针）、一个根因（launch-bound）、一套方法（假设驱动 + 同源校验 + 定性定量分离）。**
