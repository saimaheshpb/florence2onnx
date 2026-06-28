#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <cmath>

class TensorNormalizeTest {
private:
    void test_normalize_one_one_to_zero_one() {
        std::cout << "\nTesting normalize [-1,1] to [0,1]..." << std::endl;
        
        // Test data in [-1,1] range
        std::vector<float> input = {
            -1.0f,     // Should become 0.0
            -0.5f,     // Should become 0.25
            0.0f,      // Should become 0.5
            0.5f,      // Should become 0.75
            1.0f       // Should become 1.0
        };
        
        auto result = florence2::TensorExtension::normalize_one_one_to_zero_one(
            input.data(), input.size());
        
        std::vector<float> expected = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        
        for (size_t i = 0; i < input.size(); i++) {
            std::cout << input[i] << " -> " << result[i] 
                     << " (expected: " << expected[i] << ")" << std::endl;
            assert(std::abs(result[i] - expected[i]) < 1e-5f);
            
            // Verify range
            assert(result[i] >= 0.0f && result[i] <= 1.0f);
        }
        
        std::cout << "Normalize [-1,1] to [0,1]: OK" << std::endl;
    }

    void test_normalize_zero_one_to_one_one() {
        std::cout << "\nTesting normalize [0,1] to [-1,1]..." << std::endl;
        
        // Test data in [0,1] range
        std::vector<float> input = {
            0.0f,      // Should become -1.0
            0.25f,     // Should become -0.5
            0.5f,      // Should become 0.0
            0.75f,     // Should become 0.5
            1.0f       // Should become 1.0
        };
        
        auto result = florence2::TensorExtension::normalize_zero_one_to_one_one(
            input.data(), input.size());
        
        std::vector<float> expected = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        
        for (size_t i = 0; i < input.size(); i++) {
            std::cout << input[i] << " -> " << result[i] 
                     << " (expected: " << expected[i] << ")" << std::endl;
            assert(std::abs(result[i] - expected[i]) < 1e-5f);
            
            // Verify range
            assert(result[i] >= -1.0f && result[i] <= 1.0f);
        }
        
        std::cout << "Normalize [0,1] to [-1,1]: OK" << std::endl;
    }

    void test_normalization_roundtrip() {
        std::cout << "\nTesting normalization roundtrip..." << std::endl;
        
        std::vector<float> original = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        
        // Convert to [0,1]
        auto to_zero_one = florence2::TensorExtension::normalize_one_one_to_zero_one(
            original.data(), original.size());
        
        // Convert back to [-1,1]
        auto back_to_one_one = florence2::TensorExtension::normalize_zero_one_to_one_one(
            to_zero_one.data(), to_zero_one.size());
        
        // Verify roundtrip
        for (size_t i = 0; i < original.size(); i++) {
            std::cout << original[i] << " -> " << to_zero_one[i] 
                     << " -> " << back_to_one_one[i] << std::endl;
            assert(std::abs(back_to_one_one[i] - original[i]) < 1e-5f);
        }
        
        std::cout << "Normalization roundtrip: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting Normalization tests..." << std::endl;
        
        try {
            test_normalize_one_one_to_zero_one();
            test_normalize_zero_one_to_one_one();
            test_normalization_roundtrip();
            
            std::cout << "\nAll normalization tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorNormalizeTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}