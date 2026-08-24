# Essay plan

## Basic knowledge

- Model quantization

- TD3

- Action chunking

## Problem

End-to-end navigation models running on resource-limited robots suffer from low inference speed, because the onboard computation resources are constrained. There is an urgent need to reduce the computation required to run the model on the robot.

## Method

We combine two orthogonal techniques to speed up the TD3 navigation model:

1. **Quantization**: quantize the model (e.g., FP32 -> INT8) to reduce the per-inference latency by accelerating the convolution forward pass. Quantization is done as TRT-native PTQ (INT8 calibration on the target device from a pure FP32 ONNX).

2. **Action chunking**: let the model output an action sequence of length N at once, and let the robot execute the whole sequence open-loop before the model runs again. This reduces the inference frequency (from once per control step to once per N steps).

Since the two techniques affect different parts of the computation, their speed-ups can be combined. The problem is solved by reducing both the latency of each inference and the total number of inferences.

## Experiment

Method: TD3 with both quantization (INT8 PTQ) and action chunking. To
verify the contribution of each technique and their combination, we run an
ablation study of the full method:

- E4 (full method): TD3 with quantization + action chunking (INT8 PTQ, N > 1)

- E3: action chunking only (quantization removed, FP32, N > 1)

- E2: quantization only (action chunking removed, INT8 PTQ, N = 1)

- E1: neither technique (FP32, N = 1)

To study the trade-off between computation savings and navigation quality, we
additionally evaluate different chunk lengths N (e.g., N = 1 / 5 / 10).

### Metrics

- Inference latency (ms) and inference frequency (Hz)

- Navigation success rate and trajectory error, to verify that speed-up does not degrade navigation quality

- Safety-related events (e.g., collisions), especially in dynamic environments

### Environment / hardware

- Test in both static and dynamic environments

- Specify the target hardware platform (e.g., Jetson, Raspberry Pi) and the quantization framework (PTQ / QAT, TensorRT / ONNX)

## Related work

- Model quantization methods

- Action chunking / multi-step action prediction (e.g., Diffusion Policy, ACT)

- Temporal abstraction / macro-actions in reinforcement learning

- TD3

- Other work on improving computation speed on resource-limited devices
