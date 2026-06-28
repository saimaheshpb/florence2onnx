#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include "post_processing/quantization_mode.h"
#include "shared_types.h"  // Add this include

namespace florence2 {

class BoxQuantizer {
public:
    BoxQuantizer(QuantizerMode mode, std::pair<int, int> bins);

    std::vector<BoundingBox<int>> quantize(
        const std::vector<BoundingBox<float>>& boxes,
        std::pair<int, int> size);

    std::vector<BoundingBox<float>> dequantize(
        const std::vector<BoundingBox<int>>& boxes,
        std::pair<int, int> size);

private:
    QuantizerMode mode_;
    std::pair<int, int> bins_;
};

} // namespace florence2