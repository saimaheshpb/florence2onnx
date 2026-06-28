#pragma once

#include <vector>
#include <memory>
#include <array>
#include <utility>  // for std::pair

namespace florence2 {

struct CLIPConfig {
    std::array<float, 3> image_mean;
    int64_t image_seq_length;
    std::array<float, 3> image_std;
    float rescale_factor;
    int crop_height;
    int crop_width;
};

class CLIPImageProcessor {
public:
    explicit CLIPImageProcessor(const CLIPConfig& config);

    // Returns normalized pixel values as a vector and original image sizes
    // pixel_values shape: [batch_size][channels][height][width]
    std::pair<std::vector<float>, std::vector<std::pair<int, int>>> 
    preprocess(const std::vector<std::vector<uint8_t>>& image_data);
    const CLIPConfig& get_config() const { return config_; }

private:
    CLIPConfig config_;

    // Helper to normalize a single pixel's channels
    std::array<float, 3> normalize_pixel(uint8_t b, uint8_t g, uint8_t r) const;
};

} // namespace florence2