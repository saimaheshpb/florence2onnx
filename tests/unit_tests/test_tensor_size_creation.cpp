#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <numeric>

class TensorOnesTest {
private:
    void test_ones_float() {
        std::cout << "\nTesting ones_float tensor creation..." << std::endl;

        // Test 2D tensor
        std::vector<int64_t> dimensions = {2, 3}; // 2x3 tensor
        auto result = florence2::TensorExtension::ones_float(dimensions);

        // Verify size
        assert(result.size() == 6);
        std::cout << "Size check: OK" << std::endl;

        // Verify all values are 1.0f
        for (size_t i = 0; i < result.size(); ++i) {
            assert(std::abs(result[i] - 1.0f) < 1e-5f);
        }
        std::cout << "Values check: OK" << std::endl;

        // Test 3D tensor
        dimensions = {2, 2, 2}; // 2x2x2 tensor
        result = florence2::TensorExtension::ones_float(dimensions);
        
        // Verify size
        assert(result.size() == 8);
        
        // Verify all values
        for (size_t i = 0; i < result.size(); ++i) {
            assert(std::abs(result[i] - 1.0f) < 1e-5f);
        }

        // Test empty dimensions
        try {
            std::vector<int64_t> empty_dims;
            florence2::TensorExtension::ones_float(empty_dims);
            assert(false); // Should not reach here
        } catch (const std::invalid_argument& e) {
            std::cout << "Empty dimensions check: OK" << std::endl;
        }

        std::cout << "Ones float tensor: All tests OK" << std::endl;
    }

    void test_ones_long() {
        std::cout << "\nTesting ones_long tensor creation..." << std::endl;

        // Test default value (1)
        std::vector<int64_t> dimensions = {2, 3}; // 2x3 tensor
        auto result = florence2::TensorExtension::ones_long(dimensions);

        // Verify size
        assert(result.size() == 6);
        std::cout << "Size check: OK" << std::endl;

        // Verify all values are 1
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i] == 1);
        }
        std::cout << "Default value check: OK" << std::endl;

        // Test custom value
        int64_t custom_value = 5;
        result = florence2::TensorExtension::ones_long(dimensions, custom_value);
        
        // Verify all values are custom_value
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i] == custom_value);
        }
        std::cout << "Custom value check: OK" << std::endl;

        // Test 3D tensor with custom value
        dimensions = {2, 2, 2}; // 2x2x2 tensor
        custom_value = -1;
        result = florence2::TensorExtension::ones_long(dimensions, custom_value);
        
        // Verify size
        assert(result.size() == 8);
        
        // Verify all values
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i] == custom_value);
        }

        // Test empty dimensions
        try {
            std::vector<int64_t> empty_dims;
            florence2::TensorExtension::ones_long(empty_dims);
            assert(false); // Should not reach here
        } catch (const std::invalid_argument& e) {
            std::cout << "Empty dimensions check: OK" << std::endl;
        }

        std::cout << "Ones long tensor: All tests OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting Tensor Ones Operations tests..." << std::endl;

        try {
            test_ones_float();
            test_ones_long();

            std::cout << "\nAll tensor ones operation tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorOnesTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}