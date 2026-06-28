#include "model/tensor_operation_registry.h"
#include <iostream>
#include <cassert>

namespace florence2 {
namespace test {

class TensorOperationRegistryTest {
public:
    void run_all_tests() {
        std::cout << "Starting TensorOperationRegistry tests..." << std::endl;

        try {
            test_top_k_session();
            test_call_top_k();

            std::cout << "\nAll TensorOperationRegistry tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_top_k_session() {
        std::cout << "\nTesting TopK session creation..." << std::endl;
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        
        auto session = TensorOperationRegistry::top_k_session(session_options, env);
        assert(session != nullptr);
        
        std::cout << "TopK session creation: OK" << std::endl;
    }

    void test_call_top_k() {
        std::cout << "\nTesting TopK operation..." << std::endl;
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        
        auto session = TensorOperationRegistry::top_k_session(session_options, env);
        
        // Test data
        std::vector<float> x = {1.0f, 5.0f, 2.0f, 4.0f, 3.0f};
        std::vector<int64_t> k = {3};  // get top 3
        std::vector<int64_t> x_dimensions = {1, 5};  // 1 batch, 5 values
        
        auto outputs = TensorOperationRegistry::call_top_k(
            session.get(), x, k, x_dimensions);
        
        assert(outputs.size() == 2);  // values and indices
        
        // Get results
        float* values = outputs[0].GetTensorMutableData<float>();
        int64_t* indices = outputs[1].GetTensorMutableData<int64_t>();
        
        // Check top 3 values
        assert(values[0] == 5.0f);  // highest
        assert(values[1] == 4.0f);  // second highest
        assert(values[2] == 3.0f);  // third highest
        
        // Check corresponding indices
        assert(indices[0] == 1);  // index of 5.0
        assert(indices[1] == 3);  // index of 4.0
        assert(indices[2] == 4);  // index of 3.0
        
        std::cout << "TopK operation: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::TensorOperationRegistryTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}