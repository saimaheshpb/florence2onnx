#include "post_processing/box_quantizer.h"
#include <stdexcept>

namespace florence2 {

BoxQuantizer::BoxQuantizer(QuantizerMode mode, std::pair<int, int> bins)
    : mode_(mode)
    , bins_(bins) {
}

std::vector<BoundingBox<int>> BoxQuantizer::quantize(
    const std::vector<BoundingBox<float>>& boxes,
    std::pair<int, int> size) 
{
    auto [bins_w, bins_h] = bins_;
    auto [size_w, size_h] = size;
    float size_per_bin_w = static_cast<float>(size_w) / bins_w;
    float size_per_bin_h = static_cast<float>(size_h) / bins_h;

    std::vector<BoundingBox<int>> quantized_boxes;
    quantized_boxes.reserve(boxes.size());

    for (const auto& box : boxes) {
        switch (mode_) {
            case QuantizerMode::Floor: {
                quantized_boxes.emplace_back(
                    std::clamp(static_cast<int>(std::floor(box.xmin / size_per_bin_w)), 0, bins_w - 1),
                    std::clamp(static_cast<int>(std::floor(box.ymin / size_per_bin_h)), 0, bins_h - 1),
                    std::clamp(static_cast<int>(std::floor(box.xmax / size_per_bin_w)), 0, bins_w - 1),
                    std::clamp(static_cast<int>(std::floor(box.ymax / size_per_bin_h)), 0, bins_h - 1)
                );
                break;
            }
            default:
                throw std::invalid_argument("Incorrect quantization type.");
        }
    }

    return quantized_boxes;
}

std::vector<BoundingBox<float>> BoxQuantizer::dequantize(
    const std::vector<BoundingBox<int>>& boxes,
    std::pair<int, int> size) 
{
    auto [bins_w, bins_h] = bins_;
    auto [size_w, size_h] = size;
    float size_per_bin_w = static_cast<float>(size_w) / bins_w;
    float size_per_bin_h = static_cast<float>(size_h) / bins_h;

    std::vector<BoundingBox<float>> dequantized_boxes;
    dequantized_boxes.reserve(boxes.size());

    for (const auto& box : boxes) {
        switch (mode_) {
            case QuantizerMode::Floor: {
                dequantized_boxes.emplace_back(
                    (box.xmin + 0.5f) * size_per_bin_w,
                    (box.ymin + 0.5f) * size_per_bin_h,
                    (box.xmax + 0.5f) * size_per_bin_w,
                    (box.ymax + 0.5f) * size_per_bin_h
                );
                break;
            }
            default:
                throw std::invalid_argument("Incorrect quantization type.");
        }
    }

    return dequantized_boxes;
}

} // namespace florence2