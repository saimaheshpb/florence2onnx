#include <onnxruntime_cxx_api.h>
#include "helper/tensor_extension.h"
#include <iostream>
#include <vector>
#include <cassert>

class TensorCreationTest {
private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};

    void test_vector_to_tensor_conversion() {
        std::cout << "\nTesting vector to tensor conversion..." << std::endl;
        
        // Create test data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};  // 2x2 matrix
        
        std::cout << "Original data: ";
        for(const auto& val : data) std::cout << val << " ";
        std::cout << std::endl;

        try {
            // Create tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(data.data()),
                data.size(),
                shape.data(),
                shape.size()
            );

            // Verify tensor was created
            auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
            auto tensor_shape = tensor_info.GetShape();
            auto element_count = tensor_info.GetElementCount();
            
            std::cout << "Tensor created with:" << std::endl;
            std::cout << "- Shape: [" << tensor_shape[0] << ", " << tensor_shape[1] << "]" << std::endl;
            std::cout << "- Element count: " << element_count << std::endl;

            // Read back data from tensor
            const float* tensor_data = tensor.GetTensorData<float>();
            std::cout << "Data from tensor: ";
            for(size_t i = 0; i < element_count; i++) {
                std::cout << tensor_data[i] << " ";
                assert(std::abs(tensor_data[i] - data[i]) < 1e-5f);
            }
            std::cout << std::endl;

            // Test 2D access
            std::cout << "Accessing as 2D matrix:" << std::endl;
            for(int64_t i = 0; i < tensor_shape[0]; i++) {
                for(int64_t j = 0; j < tensor_shape[1]; j++) {
                    size_t index = i * tensor_shape[1] + j;
                    std::cout << "tensor[" << i << "][" << j << "] = " << tensor_data[index] << std::endl;
                }
            }

            std::cout << "Vector to tensor conversion: OK" << std::endl;
        }
        catch(const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            throw;
        }
    }

    void test_tensor_operations() {
        std::cout << "\nTesting tensor operations..." << std::endl;
        
        // Create input data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};

        try {
            // Create input tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(data.data()),
                data.size(),
                shape.data(),
                shape.size()
            );

            // Perform operation using TensorExtension
            const float* input_data = input_tensor.GetTensorData<float>();
            auto result_data = florence2::TensorExtension::multiply_by_float(
                input_data, 
                data.size(), 
                2.0f
            );

            // Create result tensor
            auto result_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(result_data.data()),
                result_data.size(),
                shape.data(),
                shape.size()
            );

            // Verify result
            const float* result_tensor_data = result_tensor.GetTensorData<float>();
            std::cout << "Result tensor data: ";
            for(size_t i = 0; i < data.size(); i++) {
                std::cout << result_tensor_data[i] << " ";
                assert(std::abs(result_tensor_data[i] - data[i] * 2.0f) < 1e-5f);
            }
            std::cout << std::endl;

            std::cout << "Tensor operations: OK" << std::endl;
        }
        catch(const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            throw;
        }
    }

public:
    void run_all_tests() {
        std::cout << "Starting Tensor Creation tests..." << std::endl;
        
        try {
            test_vector_to_tensor_conversion();
            test_tensor_operations();
            
            std::cout << "\nAll tensor creation tests passed successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TensorCreationTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}