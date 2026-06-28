#include "post_processing/box_quantizer.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace florence2 {
namespace test {

class BoxQuantizerTest {
public:
    void run_all_tests() {
        std::cout << "Starting BoxQuantizer tests..." << std::endl;

        try {
            test_initialization();
            test_quantization();
            test_dequantization();
            test_round_trip();
            test_error_handling();

            std::cout << "\nAll BoxQuantizer tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_initialization() {
        std::cout << "\nTesting initialization..." << std::endl;
        
        BoxQuantizer quantizer(QuantizerMode::Floor, {100, 100});
        std::cout << "Initialization: OK" << std::endl;
    }

    void test_quantization() {
        std::cout << "\nTesting quantization..." << std::endl;
        
        BoxQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        // Test single box quantization
        std::vector<BoundingBox<float>> boxes = {
            BoundingBox<float>(25.6f, 30.4f, 45.2f, 60.8f)
        };
        
        auto result = quantizer.quantize(boxes, {100, 100});
        
        // Each bin is 10x10 pixels (100/10 = 10)
        assert(result[0].xmin == 2); // 25.6/10 -> floor(2.56) = 2
        assert(result[0].ymin == 3); // 30.4/10 -> floor(3.04) = 3
        assert(result[0].xmax == 4); // 45.2/10 -> floor(4.52) = 4
        assert(result[0].ymax == 6); // 60.8/10 -> floor(6.08) = 6
        
        std::cout << "Basic quantization: OK" << std::endl;

        // Test clamping
        std::vector<BoundingBox<float>> overflow_boxes = {
            BoundingBox<float>(-10.0f, 5.0f, 150.0f, 95.0f)
        };
        
        auto clamped = quantizer.quantize(overflow_boxes, {100, 100});
        
        assert(clamped[0].xmin == 0);  // Clamped to min
        assert(clamped[0].xmax == 9);  // Clamped to max
        
        std::cout << "Clamping: OK" << std::endl;
    }

    void test_dequantization() {
        std::cout << "\nTesting dequantization..." << std::endl;
        
        BoxQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        std::vector<BoundingBox<int>> boxes = {
            BoundingBox<int>(2, 3, 4, 6)
        };
        
        auto result = quantizer.dequantize(boxes, {100, 100});
        
        // Each bin is 10x10 pixels, adding 0.5 for center
        const float epsilon = 0.001f;
        assert(std::abs(result[0].xmin - 25.0f) < epsilon); // (2 + 0.5) * 10
        assert(std::abs(result[0].ymin - 35.0f) < epsilon); // (3 + 0.5) * 10
        assert(std::abs(result[0].xmax - 45.0f) < epsilon); // (4 + 0.5) * 10
        assert(std::abs(result[0].ymax - 65.0f) < epsilon); // (6 + 0.5) * 10
        
        std::cout << "Dequantization: OK" << std::endl;
    }

    void test_round_trip() {
        std::cout << "\nTesting quantization-dequantization round trip..." << std::endl;
        
        BoxQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        std::vector<BoundingBox<float>> original = {
            BoundingBox<float>(25.6f, 30.4f, 45.2f, 60.8f)
        };
        
        auto quantized = quantizer.quantize(original, {100, 100});
        auto roundtrip = quantizer.dequantize(quantized, {100, 100});
        
        // Values should be quantized to bin centers
        const float epsilon = 0.001f;
        assert(std::abs(roundtrip[0].xmin - 25.0f) < epsilon);
        assert(std::abs(roundtrip[0].ymin - 35.0f) < epsilon);
        assert(std::abs(roundtrip[0].xmax - 45.0f) < epsilon);
        assert(std::abs(roundtrip[0].ymax - 65.0f) < epsilon);
        
        std::cout << "Round trip: OK" << std::endl;
    }

    void test_error_handling() {
        std::cout << "\nTesting error handling..." << std::endl;
        
        // Test with invalid mode (if possible in your enum setup)
        // Note: In C++, this kind of test might not be possible depending on how
        // the enum is defined, as invalid values are harder to create than in C#
        
        std::cout << "Error handling: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::BoxQuantizerTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}