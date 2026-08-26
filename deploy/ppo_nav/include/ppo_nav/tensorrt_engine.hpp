#ifndef PPO_NAV_TENSORRT_ENGINE_HPP_
#define PPO_NAV_TENSORRT_ENGINE_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace ppo_nav {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

/// 固定输入/输出的 TensorRT 引擎封装（obs/action 元素数由引擎绑定维度决定）。
/// N=1 模型: [1,105] -> [1,2]；N=5 模型: [1,113] -> [1,10]（μ，机器人侧需 clip 到 [-1,1] 再映射速度）。
class TrtEngine {
 public:
  TrtEngine() = default;
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  /// 从 .engine 文件反序列化并分配 CUDA buffer。
  bool load(const std::string& engine_path);

  /// 同步推理: obs(输入维度) -> action(输出维度)。
  bool forward(const std::vector<float>& obs, std::vector<float>& action) const;

  int inputDim() const { return input_dim_; }
  int outputDim() const { return output_dim_; }

 private:
  bool initBuffers();

  TrtLogger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  void* input_buf_ = nullptr;
  void* output_buf_ = nullptr;
  int input_dim_ = 0;
  int output_dim_ = 0;
  int input_idx_ = -1;
  int output_idx_ = -1;
};

}  // namespace ppo_nav

#endif  // PPO_NAV_TENSORRT_ENGINE_HPP_
