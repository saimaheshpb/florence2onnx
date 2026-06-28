#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <utility>
#include <vector>

namespace florence2 {

class ILogitsSampler {
public:
  virtual ~ILogitsSampler() = default;
  virtual std::vector<std::pair<int64_t, double>>
  sample(int batch_idx, const std::vector<float> &logits,
         const std::vector<int64_t> &dimensions) = 0;
};

class BeamSearchSampler : public ILogitsSampler {
public:
  BeamSearchSampler(Ort::Session *top_k_session, int top_k, int num_beams);

  std::vector<std::pair<int64_t, double>>
  sample(int batch_idx, const std::vector<float> &logits,
         const std::vector<int64_t> &dimensions) override;

private:
  static std::vector<float> softmax(const std::vector<float> &arr);

  Ort::Session *top_k_session_; // Non-owning pointer
  int top_k_;
  int num_beams_;
};

} // namespace florence2