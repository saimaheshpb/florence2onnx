#include "model/tensor_operation_registry.h"
#include <onnxruntime_cxx_api.h>

namespace florence2 {

// Initialize static member with TopK model bytes
const std::vector<uint8_t> TensorOperationRegistry::TOP_K_MODEL_BYTES = {
    8,  10, 18,  0,  58,  73, 10,  18,  10, 1,  120, 10,  1,  107, 18, 1, 118,
    18, 1,  105, 34, 4,   84, 111, 112, 75, 18, 1,   116, 90, 9,   10, 1, 120,
    18, 4,  10,  2,  8,   1,  90,  15,  10, 1,  107, 18,  10, 10,  8,  8, 7,
    18, 4,  10,  2,  8,   1,  98,  9,   10, 1,  118, 18,  4,  10,  2,  8, 1,
    98, 9,  10,  1,  105, 18, 4,   10,  2,  8,  7,   66,  2,  16,  21};

std::unique_ptr<Ort::Session>
TensorOperationRegistry::top_k_session(Ort::SessionOptions &session_options,
                                       Ort::Env &env) {
  return std::make_unique<Ort::Session>(
      env, TOP_K_MODEL_BYTES.data(), TOP_K_MODEL_BYTES.size(), session_options);
}

std::vector<Ort::Value> TensorOperationRegistry::call_top_k(
    Ort::Session *session, const std::vector<float> &x,
    const std::vector<int64_t> &k, const std::vector<int64_t> &x_dimensions) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensors
  Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(x.data()), x.size(), x_dimensions.data(),
      x_dimensions.size());

  std::vector<int64_t> k_dimensions = {1}; // k is always a scalar
  Ort::Value k_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, const_cast<int64_t *>(k.data()), k.size(),
      k_dimensions.data(), k_dimensions.size());

  // Setup inputs
  std::vector<const char *> input_names = {"k", "x"};
  std::vector<const char *> output_names = {"v", "i"};
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(k_tensor));
  input_tensors.push_back(std::move(x_tensor));

  // Run inference
  return session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                      input_tensors.data(), input_tensors.size(),
                      output_names.data(), output_names.size());
}

} // namespace florence2