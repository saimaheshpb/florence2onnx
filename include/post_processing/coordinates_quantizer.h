#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include "post_processing/quantization_mode.h"
#include "shared_types.h"  // Add this include

namespace florence2 {

class CoordinatesQuantizer {
public:
    CoordinatesQuantizer(QuantizerMode mode, std::pair<int, int> bins)
        : mode_(mode)
        , bins_(bins) {}
    
    std::vector<Coordinates<int>> quantize(
        const std::vector<Coordinates<float>>& coordinates,
        std::pair<float, float> size) 
    {
        auto [bins_w, bins_h] = bins_;
        auto [size_w, size_h] = size;
        float size_per_bin_w = size_w / bins_w;
        float size_per_bin_h = size_h / bins_h;

        std::vector<Coordinates<int>> quantized;
        quantized.reserve(coordinates.size());

        for (const auto& coord : coordinates) {
            switch (mode_) {
                case QuantizerMode::Floor: {
                    int quantized_x = std::clamp(static_cast<int>(std::floor(coord.x / size_per_bin_w)), 0, bins_w - 1);
                    int quantized_y = std::clamp(static_cast<int>(std::floor(coord.y / size_per_bin_h)), 0, bins_h - 1);
                    quantized.emplace_back(quantized_x, quantized_y);
                    break;
                }
                default:
                    throw std::invalid_argument("Invalid quantization mode");
            }
        }

        return quantized;
    }
        
    std::vector<Coordinates<float>> dequantize(
        const std::vector<Coordinates<int>>& coordinates,
        std::pair<float, float> size) 
    {
        auto [bins_w, bins_h] = bins_;
        auto [size_w, size_h] = size;
        float size_per_bin_w = size_w / bins_w;
        float size_per_bin_h = size_h / bins_h;

        std::vector<Coordinates<float>> dequantized;
        dequantized.reserve(coordinates.size());

        for (const auto& coord : coordinates) {
            switch (mode_) {
                case QuantizerMode::Floor: {
                    float dequantized_x = (coord.x + 0.5f) * size_per_bin_w;
                    float dequantized_y = (coord.y + 0.5f) * size_per_bin_h;
                    dequantized.emplace_back(dequantized_x, dequantized_y);
                    break;
                }
                default:
                    throw std::invalid_argument("Invalid quantization mode");
            }
        }

        return dequantized;
    }
        
private:
    QuantizerMode mode_;
    std::pair<int, int> bins_;
};

} // namespace florence2