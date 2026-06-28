#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

class TensorInplaceTest {
private:
    void test_divide_by_inplace() {
        std::cout << "\nTesting divide by inplace..." << std::endl;
        
        std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
        float divisor = 2.0f;
        
        std::cout << "Original data: ";
        for (float val : data) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::divide_by_inplace(data.data(), data.size(), divisor);
        
        std::cout << "After division by " << divisor << ": ";
        for (size_t i = 0; i < data.size(); i++) {
            float expected = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}[i];
            std::cout << data[i] << " ";
            assert(std::abs(data[i] - expected) < 1e-5f);
        }
        std::cout << "\nDivide by inplace: OK" << std::endl;
    }

    void test_multiply_by_inplace() {
        std::cout << "\nTesting multiply by inplace..." << std::endl;
        
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        float multiplier = 2.0f;
        
        std::cout << "Original data: ";
        for (float val : data) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::multiply_by_inplace(data.data(), data.size(), multiplier);
        
        std::cout << "After multiplication by " << multiplier << ": ";
        for (size_t i = 0; i < data.size(); i++) {
            float expected = std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f}[i];
            std::cout << data[i] << " ";
            assert(std::abs(data[i] - expected) < 1e-5f);
        }
        std::cout << "\nMultiply by inplace: OK" << std::endl;
    }

    void test_abs_inplace() {
        std::cout << "\nTesting abs inplace..." << std::endl;
        
        std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
        
        std::cout << "Original data: ";
        for (float val : data) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::abs_inplace(data.data(), data.size());
        
        std::cout << "After abs: ";
        for (size_t i = 0; i < data.size(); i++) {
            float expected = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}[i];
            std::cout << data[i] << " ";
            assert(std::abs(data[i] - expected) < 1e-5f);
        }
        std::cout << "\nAbs inplace: OK" << std::endl;
    }

    void test_multiply_inplace() {
        std::cout << "\nTesting multiply inplace..." << std::endl;
        
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {2.0f, 2.0f, 2.0f, 2.0f};
        
        std::cout << "Original data: ";
        for (float val : data1) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::multiply_inplace(data1.data(), data2.data(), data1.size());
        
        std::cout << "After multiplication: ";
        for (size_t i = 0; i < data1.size(); i++) {
            float expected = std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f}[i];
            std::cout << data1[i] << " ";
            assert(std::abs(data1[i] - expected) < 1e-5f);
        }
        std::cout << "\nMultiply inplace: OK" << std::endl;
    }

    void test_add_inplace() {
        std::cout << "\nTesting add inplace..." << std::endl;
        
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {0.1f, 0.2f, 0.3f, 0.4f};
        
        std::cout << "Original data: ";
        for (float val : data1) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::add_inplace(data1.data(), data2.data(), data1.size());
        
        std::cout << "After addition: ";
        for (size_t i = 0; i < data1.size(); i++) {
            float expected = std::vector<float>{1.1f, 2.2f, 3.3f, 4.4f}[i];
            std::cout << data1[i] << " ";
            assert(std::abs(data1[i] - expected) < 1e-5f);
        }
        std::cout << "\nAdd inplace: OK" << std::endl;
    }

    void test_divide_inplace() {
        std::cout << "\nTesting divide inplace..." << std::endl;
        
        std::vector<float> data1 = {2.0f, 4.0f, 6.0f, 8.0f};
        std::vector<float> data2 = {2.0f, 2.0f, 2.0f, 2.0f};
        
        std::cout << "Original data: ";
        for (float val : data1) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::divide_inplace(data1.data(), data2.data(), data1.size());
        
        std::cout << "After division: ";
        for (size_t i = 0; i < data1.size(); i++) {
            float expected = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}[i];
            std::cout << data1[i] << " ";
            assert(std::abs(data1[i] - expected) < 1e-5f);
        }
        std::cout << "\nDivide inplace: OK" << std::endl;
    }

    void test_subtract_inplace() {
        std::cout << "\nTesting subtract inplace..." << std::endl;
        
        std::vector<float> data1 = {1.1f, 2.2f, 3.3f, 4.4f};
        std::vector<float> data2 = {0.1f, 0.2f, 0.3f, 0.4f};
        
        std::cout << "Original data: ";
        for (float val : data1) std::cout << val << " ";
        std::cout << std::endl;
        
        florence2::TensorExtension::subtract_inplace(data1.data(), data2.data(), data1.size());
        
        std::cout << "After subtraction: ";
        for (size_t i = 0; i < data1.size(); i++) {
            float expected = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}[i];
            std::cout << data1[i] << " ";
            assert(std::abs(data1[i] - expected) < 1e-5f);
        }
        std::cout << "\nSubtract inplace: OK" << std::endl;
    }

    void test_error_cases() {
        std::cout << "\nTesting error cases..." << std::endl;
        
        // Test division by zero
        std::vector<float> data = {1.0f, 2.0f};
        try {
            florence2::TensorExtension::divide_by_inplace(data.data(), data.size(), 0.0f);
            assert(false && "Should have thrown exception");
        }
        catch (const std::invalid_argument&) {
            std::cout << "Division by zero properly handled" << std::endl;
        }
        
        // Test division by zero in tensor
        std::vector<float> data1 = {1.0f, 2.0f};
        std::vector<float> data2 = {1.0f, 0.0f};
        try {
            florence2::TensorExtension::divide_inplace(data1.data(), data2.data(), data1.size());
            assert(false && "Should have thrown exception");
        }
        catch (const std::invalid_argument&) {
            std::cout << "Division by zero tensor properly handled" << std::endl;
        }
    }

public:
    void run_all_tests() {
        std::cout << "Starting Inplace Tensor Operation tests..." << std::endl;
        
        try {
            test_divide_by_inplace();
            test_multiply_by_inplace();
            test_abs_inplace();
            test_multiply_inplace();
            test_add_inplace();
            test_divide_inplace();
            test_subtract_inplace();
            test_error_cases();
            
            std::cout << "\nAll inplace tensor operation tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorInplaceTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}