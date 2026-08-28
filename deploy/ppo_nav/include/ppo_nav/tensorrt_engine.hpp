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

/// Fixed input/output TensorRT engine wrapper (obs/action element counts determined by the engine binding dims).
/// N=1 model: [1,105] -> [1,2]; N=5 model: [1,113] -> [1,10] (μ, needs clipping to [-1,1] robot-side before velocity mapping).
class TrtEngine {
 public:
  TrtEngine() = default;
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  /// Deserialize from a .engine file and allocate the CUDA buffers.
  bool load(const std::string& engine_path);

  /// Synchronous inference: obs(input dims) -> action(output dims).
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
