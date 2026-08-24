#include "td3_nav/tensorrt_engine.hpp"

#include <ros/ros.h>

#include <fstream>
#include <vector>

namespace td3 {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    ROS_WARN("[TRT] %s", msg);
  } else {
    ROS_INFO("[TRT] %s", msg);
  }
}

TrtEngine::~TrtEngine() {
  if (input_buf_) cudaFree(input_buf_);
  if (output_buf_) cudaFree(output_buf_);
  if (stream_) cudaStreamDestroy(stream_);
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();
}

bool TrtEngine::load(const std::string& engine_path) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) {
    ROS_ERROR("[TRT] cannot open engine file: %s", engine_path.c_str());
    return false;
  }
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.empty()) {
    ROS_ERROR("[TRT] engine file empty: %s", engine_path.c_str());
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    ROS_ERROR("[TRT] createInferRuntime failed");
    return false;
  }
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size(), nullptr);
  if (!engine_) {
    ROS_ERROR("[TRT] deserializeCudaEngine failed");
    return false;
  }
  context_ = engine_->createExecutionContext();
  if (!context_) {
    ROS_ERROR("[TRT] createExecutionContext failed");
    return false;
  }

  input_idx_ = engine_->getBindingIndex("obs");
  output_idx_ = engine_->getBindingIndex("action");
  if (input_idx_ < 0 || output_idx_ < 0) {
    ROS_ERROR("[TRT] bindings 'obs'/'action' not found in engine");
    return false;
  }

  auto numel = [&](int idx) {
    nvinfer1::Dims d = engine_->getBindingDimensions(idx);
    int n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= d.d[i];
    return n;
  };
  input_dim_ = numel(input_idx_);
  output_dim_ = numel(output_idx_);
  ROS_INFO("[TRT] engine loaded: input %d elements, output %d elements", input_dim_, output_dim_);
  return initBuffers();
}

bool TrtEngine::initBuffers() {
  if (cudaSetDevice(0) != cudaSuccess) {
    ROS_ERROR("[TRT] cudaSetDevice(0) failed");
    return false;
  }
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    ROS_ERROR("[TRT] cudaStreamCreate failed");
    return false;
  }
  if (cudaMalloc(&input_buf_, input_dim_ * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&output_buf_, output_dim_ * sizeof(float)) != cudaSuccess) {
    ROS_ERROR("[TRT] cudaMalloc failed");
    return false;
  }
  return true;
}

bool TrtEngine::forward(const std::vector<float>& obs, std::vector<float>& action) const {
  if (static_cast<int>(obs.size()) != input_dim_) return false;
  action.assign(output_dim_, 0.f);
  void* bindings[2] = {input_buf_, output_buf_};
  if (cudaMemcpyAsync(input_buf_, obs.data(), input_dim_ * sizeof(float),
                      cudaMemcpyHostToDevice, stream_) != cudaSuccess)
    return false;
  if (!context_->enqueueV2(bindings, stream_, nullptr)) return false;
  if (cudaMemcpyAsync(action.data(), output_buf_, output_dim_ * sizeof(float),
                      cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
    return false;
  return cudaStreamSynchronize(stream_) == cudaSuccess;
}

}  // namespace td3
