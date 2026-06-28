#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

class TensorBasicTest {
private:
    void test_multiply_by_float() {
        std::cout << "\nTesting multiply by float..." << std::endl;
        
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        float multiplier = 2.0f;
        
        auto result = florence2::TensorExtension::multiply_by_float(
            input.data(), input.size(), multiplier);
        
        // Print and verify results
        for (size_t i = 0; i < input.size(); i++) {
            float expected = input[i] * multiplier;
            std::cout << input[i] << " * " << multiplier << " = " << result[i] 
                     << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Multiply by float: OK" << std::endl;
    }

    void test_divide_by_float() {
        std::cout << "\nTesting divide by float..." << std::endl;
        
        std::vector<float> input = {2.0f, 4.0f, 6.0f, 8.0f};
        float divisor = 2.0f;
        
        auto result = florence2::TensorExtension::divide_by_float(
            input.data(), input.size(), divisor);
        
        for (size_t i = 0; i < input.size(); i++) {
            float expected = input[i] / divisor;
            std::cout << input[i] << " / " << divisor << " = " << result[i] 
                     << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Divide by float: OK" << std::endl;
    }

    void test_subtract_float() {
        std::cout << "\nTesting subtract float..." << std::endl;
        
        std::vector<float> input = {3.0f, 4.0f, 5.0f, 6.0f};
        float value = 1.0f;
        
        auto result = florence2::TensorExtension::subtract_float(
            input.data(), input.size(), value);
        
        for (size_t i = 0; i < input.size(); i++) {
            float expected = input[i] - value;
            std::cout << input[i] << " - " << value << " = " << result[i] 
                     << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Subtract float: OK" << std::endl;
    }

    void test_add_tensors() {
        std::cout << "\nTesting add tensors..." << std::endl;
        
        std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> input2 = {0.1f, 0.2f, 0.3f, 0.4f};
        
        auto result = florence2::TensorExtension::add_tensors(
            input1.data(), input2.data(), input1.size());
        
        for (size_t i = 0; i < input1.size(); i++) {
            float expected = input1[i] + input2[i];
            std::cout << input1[i] << " + " << input2[i] << " = " << result[i] 
                     << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Add tensors: OK" << std::endl;
    }

    void test_sum_tensors() {
        std::cout << "\nTesting sum tensors..." << std::endl;
        
        std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> input2 = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<float> input3 = {10.0f, 20.0f, 30.0f, 40.0f};
        
        std::vector<const float*> inputs = {input1.data(), input2.data(), input3.data()};
        
        auto result = florence2::TensorExtension::sum_tensors(inputs, input1.size());
        
        for (size_t i = 0; i < input1.size(); i++) {
            float expected = input1[i] + input2[i] + input3[i];
            std::cout << input1[i] << " + " << input2[i] << " + " << input3[i] 
                     << " = " << result[i] << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Sum tensors: OK" << std::endl;
    }

    void test_subtract_tensors() {
        std::cout << "\nTesting subtract tensors..." << std::endl;
        
        std::vector<float> input1 = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> input2 = {1.0f, 2.0f, 3.0f, 4.0f};
        
        auto result = florence2::TensorExtension::subtract_tensors(
            input1.data(), input2.data(), input1.size());
        
        for (size_t i = 0; i < input1.size(); i++) {
            float expected = input1[i] - input2[i];
            std::cout << input1[i] << " - " << input2[i] << " = " << result[i] 
                     << " (expected: " << expected << ")" << std::endl;
            assert(std::abs(result[i] - expected) < 1e-5f);
        }
        std::cout << "Subtract tensors: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting Basic Tensor Operation tests..." << std::endl;
        
        try {
            test_multiply_by_float();
            test_divide_by_float();
            test_subtract_float();
            test_add_tensors();
            test_sum_tensors();
            test_subtract_tensors();
            
            std::cout << "\nAll basic tensor operation tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorBasicTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}