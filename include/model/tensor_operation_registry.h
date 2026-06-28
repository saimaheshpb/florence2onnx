#pragma once

#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"

namespace florence2 {

class TensorOperationRegistry {
public:
    // Get session for TopK operations
    static std::unique_ptr<Ort::Session> top_k_session(Ort::SessionOptions& session_options, Ort::Env& env);
    
    // Call TopK operation
    static std::vector<Ort::Value> call_top_k(
        Ort::Session* session, 
        const std::vector<float>& x,
        const std::vector<int64_t>& k,
        const std::vector<int64_t>& x_dimensions);

private:
    // Hardcoded TopK model bytes from C# implementation
    static const std::vector<uint8_t> TOP_K_MODEL_BYTES;
};

} // namespace florence2