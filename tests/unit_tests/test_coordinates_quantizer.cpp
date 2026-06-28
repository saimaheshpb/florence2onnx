#include "post_processing/coordinates_quantizer.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace florence2 {
namespace test {

class CoordinatesQuantizerTest {
public:
    void run_all_tests() {
        std::cout << "Starting CoordinatesQuantizer tests..." << std::endl;

        try {
            test_initialization();
            test_quantization();
            test_dequantization();
            test_round_trip();
            test_error_handling();

            std::cout << "\nAll CoordinatesQuantizer tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_initialization() {
        std::cout << "\nTesting initialization..." << std::endl;
        
        CoordinatesQuantizer quantizer(QuantizerMode::Floor, {100, 100});
        std::cout << "Initialization: OK" << std::endl;
    }

    void test_quantization() {
        std::cout << "\nTesting quantization..." << std::endl;
        
        CoordinatesQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        // Test coordinates quantization
        std::vector<Coordinates<float>> coords = {
            Coordinates<float>(25.6f, 30.4f)
        };
        
        auto result = quantizer.quantize(coords, {100.0f, 100.0f});
        
        // Each bin is 10x10 pixels (100/10 = 10)
        assert(result[0].x == 2); // 25.6/10 -> floor(2.56) = 2
        assert(result[0].y == 3); // 30.4/10 -> floor(3.04) = 3
        
        std::cout << "Basic quantization: OK" << std::endl;

        // Test clamping
        std::vector<Coordinates<float>> overflow_coords = {
            Coordinates<float>(-10.0f, 150.0f)
        };
        
        auto clamped = quantizer.quantize(overflow_coords, {100.0f, 100.0f});
        
        assert(clamped[0].x == 0);  // Clamped to min
        assert(clamped[0].y == 9);  // Clamped to max
        
        std::cout << "Clamping: OK" << std::endl;
    }

    void test_dequantization() {
        std::cout << "\nTesting dequantization..." << std::endl;
        
        CoordinatesQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        std::vector<Coordinates<int>> coords = {
            Coordinates<int>(2, 3)
        };
        
        auto result = quantizer.dequantize(coords, {100.0f, 100.0f});
        
        // Each bin is 10x10 pixels, adding 0.5 for center
        const float epsilon = 0.001f;
        assert(std::abs(result[0].x - 25.0f) < epsilon); // (2 + 0.5) * 10
        assert(std::abs(result[0].y - 35.0f) < epsilon); // (3 + 0.5) * 10
        
        std::cout << "Dequantization: OK" << std::endl;
    }

    void test_round_trip() {
        std::cout << "\nTesting quantization-dequantization round trip..." << std::endl;
        
        CoordinatesQuantizer quantizer(QuantizerMode::Floor, {10, 10});
        
        std::vector<Coordinates<float>> original = {
            Coordinates<float>(25.6f, 30.4f)
        };
        
        auto quantized = quantizer.quantize(original, {100.0f, 100.0f});
        auto roundtrip = quantizer.dequantize(quantized, {100.0f, 100.0f});
        
        // Values should be quantized to bin centers
        const float epsilon = 0.001f;
        assert(std::abs(roundtrip[0].x - 25.0f) < epsilon);
        assert(std::abs(roundtrip[0].y - 35.0f) < epsilon);
        
        std::cout << "Round trip: OK" << std::endl;
    }

    void test_error_handling() {
        std::cout << "\nTesting error handling..." << std::endl;
        
        try {
            // Test with empty vector
            CoordinatesQuantizer quantizer(QuantizerMode::Floor, {10, 10});
            std::vector<Coordinates<float>> empty_coords;
            auto result = quantizer.quantize(empty_coords, {100.0f, 100.0f});
            assert(result.empty());
            std::cout << "Empty input handling: OK" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error handling test failed: " << e.what() << std::endl;
            throw;
        }
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::CoordinatesQuantizerTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}