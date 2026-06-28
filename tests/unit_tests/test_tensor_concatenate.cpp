#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <numeric>

class TensorConcatTest {
private:
    void test_concatenate_axis1_float() {
        std::cout << "\nTesting float tensor concatenation along axis 1..." << std::endl;

        // Create first tensor (2x2x2)
        std::vector<float> tensor1_data = {
            // i=0
            1.0f, 2.0f,  // j=0
            3.0f, 4.0f,  // j=1
            // i=1
            5.0f, 6.0f,  // j=0
            7.0f, 8.0f   // j=1
        };
        std::vector<int64_t> tensor1_dims = {2, 2, 2};  // 2x2x2

        // Create second tensor (2x1x2)
        std::vector<float> tensor2_data = {
            // i=0
            9.0f, 10.0f,  // j=0
            // i=1
            11.0f, 12.0f  // j=0
        };
        std::vector<int64_t> tensor2_dims = {2, 1, 2};  // 2x1x2

        // Expected result (2x3x2)
        std::vector<float> expected = {
            // i=0
            1.0f, 2.0f,   // j=0
            3.0f, 4.0f,   // j=1
            9.0f, 10.0f,  // j=2
            // i=1
            5.0f, 6.0f,   // j=0
            7.0f, 8.0f,   // j=1
            11.0f, 12.0f  // j=2
        };

        auto result = florence2::TensorExtension::concatenate_axis1(
            tensor1_data.data(), tensor1_dims,
            tensor2_data.data(), tensor2_dims
        );

        // Verify result size
        assert(result.size() == expected.size());
        std::cout << "Result size check: OK" << std::endl;

        // Print actual and expected values for debugging
        std::cout << "Result values:" << std::endl;
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << "result[" << i << "] = " << result[i] 
                      << ", expected[" << i << "] = " << expected[i] << std::endl;
            assert(std::abs(result[i] - expected[i]) < 1e-5f);
        }

        std::cout << "Float tensor concatenation: OK" << std::endl;
    }

    void test_concatenate_axis1_int64() {
        std::cout << "\nTesting int64 tensor concatenation along axis 1..." << std::endl;

        // Create first tensor (2x2x2)
        std::vector<int64_t> tensor1_data = {
            // i=0
            1, 2,  // j=0
            3, 4,  // j=1
            // i=1
            5, 6,  // j=0
            7, 8   // j=1
        };
        std::vector<int64_t> tensor1_dims = {2, 2, 2};  // 2x2x2

        // Create second tensor (2x1x2)
        std::vector<int64_t> tensor2_data = {
            // i=0
            9, 10,   // j=0
            // i=1
            11, 12   // j=0
        };
        std::vector<int64_t> tensor2_dims = {2, 1, 2};  // 2x1x2

        // Expected result (2x3x2)
        std::vector<int64_t> expected = {
            // i=0
            1, 2,   // j=0
            3, 4,   // j=1
            9, 10,  // j=2
            // i=1
            5, 6,   // j=0
            7, 8,   // j=1
            11, 12  // j=2
        };

        auto result = florence2::TensorExtension::concatenate_axis1(
            tensor1_data.data(), tensor1_dims,
            tensor2_data.data(), tensor2_dims
        );

        // Verify result size
        assert(result.size() == expected.size());
        std::cout << "Result size check: OK" << std::endl;

        // Print actual and expected values for debugging
        std::cout << "Result values:" << std::endl;
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << "result[" << i << "] = " << result[i] 
                      << ", expected[" << i << "] = " << expected[i] << std::endl;
            assert(result[i] == expected[i]);
        }

        std::cout << "Int64 tensor concatenation: OK" << std::endl;
    }

    void test_error_cases() {
        std::cout << "\nTesting error cases..." << std::endl;

        // Test tensors with mismatched dimensions
        std::vector<float> tensor1_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> tensor1_dims = {2, 1, 2};

        std::vector<float> tensor2_data = {5.0f, 6.0f};
        std::vector<int64_t> tensor2_dims = {1, 1, 2};  // Different batch size

        try {
            auto result = florence2::TensorExtension::concatenate_axis1(
                tensor1_data.data(), tensor1_dims,
                tensor2_data.data(), tensor2_dims
            );
            // Should not reach here
            assert(false && "Expected dimension mismatch error");
        } catch (const std::invalid_argument& e) {
            std::cout << "Dimension mismatch error handling: OK" << std::endl;
        }
    }

public:
    void run_all_tests() {
        std::cout << "Starting Tensor Concatenation tests..." << std::endl;

        try {
            test_concatenate_axis1_float();
            test_concatenate_axis1_int64();
            test_error_cases();

            std::cout << "\nAll tensor concatenation tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorConcatTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}