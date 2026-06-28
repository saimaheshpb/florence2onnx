#include "model/clip_image_processor.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>

namespace florence2 {
namespace test {

class CLIPImageProcessorTest {
public:
    void run_all_tests() {
        std::cout << "Starting CLIPImageProcessor tests..." << std::endl;

        try {
            test_initialization();
            test_preprocessing();
            test_normalization_values();
            test_error_handling();

            std::cout << "\nAll CLIPImageProcessor tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_initialization() {
        std::cout << "\nTesting CLIPImageProcessor initialization..." << std::endl;
        
        CLIPConfig config{
            {0.485f, 0.456f, 0.406f},  // image_mean
            577,                        // image_seq_length
            {0.229f, 0.224f, 0.225f},  // image_std
            0.00392156862745098f,      // rescale_factor (1/255)
            768,                        // crop_height
            768                         // crop_width
        };

        CLIPImageProcessor processor(config);
        std::cout << "Initialization: OK" << std::endl;
    }

    void test_preprocessing() {
        std::cout << "\nTesting image preprocessing..." << std::endl;

        // Create test config with smaller dimensions for testing
        CLIPConfig config{
            {0.485f, 0.456f, 0.406f},  // image_mean
            577,                        // image_seq_length
            {0.229f, 0.224f, 0.225f},  // image_std
            0.00392156862745098f,      // rescale_factor
            224,                        // crop_height - smaller for testing
            224                         // crop_width - smaller for testing
        };

        CLIPImageProcessor processor(config);

        // Load a test image
        std::vector<uint8_t> image_data = load_test_image("./tests/test_image/image.png");
        std::vector<std::vector<uint8_t>> batch = {image_data};

        // Process image
        auto [pixel_values, image_sizes] = processor.preprocess(batch);

        // Verify output dimensions
        size_t expected_size = 1 * 3 * config.crop_height * config.crop_width;
        assert(pixel_values.size() == expected_size);
        assert(image_sizes.size() == 1);
        
        std::cout << "Preprocessing dimensions: OK" << std::endl;
    }

    void test_normalization_values() {
        std::cout << "\nTesting normalization values..." << std::endl;

        CLIPConfig config{
            {0.485f, 0.456f, 0.406f},
            577,
            {0.229f, 0.224f, 0.225f},
            0.00392156862745098f,
            224,
            224
        };

        CLIPImageProcessor processor(config);

        // Create a simple test image (solid color)
        std::vector<uint8_t> image_data = create_solid_color_image(255, 255, 255);
        std::vector<std::vector<uint8_t>> batch = {image_data};

        auto [pixel_values, image_sizes] = processor.preprocess(batch);

        // Check first pixel values (should be normalized according to config)
        float expected_b = ((255 * config.rescale_factor) - config.image_mean[0]) / config.image_std[0];
        float expected_g = ((255 * config.rescale_factor) - config.image_mean[1]) / config.image_std[1];
        float expected_r = ((255 * config.rescale_factor) - config.image_mean[2]) / config.image_std[2];

        // Allow small floating point differences
        const float epsilon = 1e-5f;
        assert(std::abs(pixel_values[0] - expected_b) < epsilon);
        assert(std::abs(pixel_values[config.crop_height * config.crop_width] - expected_g) < epsilon);
        assert(std::abs(pixel_values[2 * config.crop_height * config.crop_width] - expected_r) < epsilon);

        std::cout << "Normalization values: OK" << std::endl;
    }

    void test_error_handling() {
        std::cout << "\nTesting error handling..." << std::endl;

        CLIPConfig config{
            {0.485f, 0.456f, 0.406f},
            577,
            {0.229f, 0.224f, 0.225f},
            0.00392156862745098f,
            224,
            224
        };

        CLIPImageProcessor processor(config);

        // Test empty batch
        try {
            std::vector<std::vector<uint8_t>> empty_batch;
            processor.preprocess(empty_batch);
            assert(false && "Expected exception for empty batch");
        } catch (const std::runtime_error&) {
            std::cout << "Empty batch handling: OK" << std::endl;
        }

        // Test invalid image data
        try {
            std::vector<uint8_t> invalid_data = {1, 2, 3}; // Invalid image bytes
            std::vector<std::vector<uint8_t>> invalid_batch = {invalid_data};
            processor.preprocess(invalid_batch);
            assert(false && "Expected exception for invalid image data");
        } catch (const std::runtime_error&) {
            std::cout << "Invalid image data handling: OK" << std::endl;
        }
    }

    // Helper functions
    std::vector<uint8_t> load_test_image(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open test image: " + filename);
        }
        return std::vector<uint8_t>(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    }

    std::vector<uint8_t> create_solid_color_image(uint8_t r, uint8_t g, uint8_t b) {
        // Create a small 2x2 test image using OpenCV
        cv::Mat img(2, 2, CV_8UC3, cv::Scalar(b, g, r));
        std::vector<uint8_t> buffer;
        cv::imencode(".jpg", img, buffer);
        return buffer;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::CLIPImageProcessorTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}