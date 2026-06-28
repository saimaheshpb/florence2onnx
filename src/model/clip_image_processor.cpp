#include "model/clip_image_processor.h"
#include "opencv2/opencv.hpp"
#include <stdexcept>

namespace florence2 {

CLIPImageProcessor::CLIPImageProcessor(const CLIPConfig& config)
    : config_(config) {
}

std::array<float, 3> CLIPImageProcessor::normalize_pixel(uint8_t b, uint8_t g, uint8_t r) const {
    return {
        ((b * config_.rescale_factor) - config_.image_mean[2]) / config_.image_std[2],
        ((g * config_.rescale_factor) - config_.image_mean[1]) / config_.image_std[1],
        ((r * config_.rescale_factor) - config_.image_mean[0]) / config_.image_std[0]
    };
}

std::pair<std::vector<float>, std::vector<std::pair<int, int>>> 
CLIPImageProcessor::preprocess(const std::vector<std::vector<uint8_t>>& image_data) {
    if (image_data.empty()) {
        throw std::runtime_error("No images provided");
    }

    const size_t batch_size = image_data.size();
    // Correct tensor size calculation: batch_size * channels * height * width
    const size_t tensor_size = batch_size * 3 * config_.crop_height * config_.crop_width;
    std::vector<float> normalized_values(tensor_size);
    std::vector<std::pair<int, int>> image_sizes(batch_size);

    // Process each image in batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Load and resize image
        cv::Mat img = cv::imdecode(image_data[batch_idx], cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Failed to decode image " + std::to_string(batch_idx));
        }

        // Store resized dimensions
        image_sizes[batch_idx] = {config_.crop_width, config_.crop_height};

        // Resize image
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(config_.crop_width, config_.crop_height), 
                  0, 0, cv::INTER_CUBIC);

        // Calculate base offset for this batch
        const size_t batch_offset = batch_idx * 3 * config_.crop_height * config_.crop_width;

        // Process pixels
        for (int h = 0; h < config_.crop_height; ++h) {
            for (int w = 0; w < config_.crop_width; ++w) {
                const cv::Vec3b& pixel = resized.at<cv::Vec3b>(h, w);
                auto normalized = normalize_pixel(pixel[0], pixel[1], pixel[2]);

                // Calculate position in normalized_values array
                // For each channel:
                // Index = batch_offset + (channel_idx * H * W) + (h * W + w)
                const size_t pos = h * config_.crop_width + w;
                
                // Store in RGB order (OpenCV gives BGR)
                normalized_values[batch_offset + (0 * config_.crop_height * config_.crop_width) + pos] = normalized[2];  // R
                normalized_values[batch_offset + (1 * config_.crop_height * config_.crop_width) + pos] = normalized[1];  // G
                normalized_values[batch_offset + (2 * config_.crop_height * config_.crop_width) + pos] = normalized[0];  // B
            }
        }
    }

    // Add debug information
    std::cout << "Preprocessed tensor shape:" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Channels: 3" << std::endl;
    std::cout << "Height: " << config_.crop_height << std::endl;
    std::cout << "Width: " << config_.crop_width << std::endl;
    std::cout << "Total tensor size: " << normalized_values.size() << std::endl;

    return {normalized_values, image_sizes};
}

} // namespace florence2