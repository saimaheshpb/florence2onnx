#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>

class TensorCreator
{
private:
    Ort::Env env;
    Ort::MemoryInfo memoryInfo;

public:
    TensorCreator()
        : env(ORT_LOGGING_LEVEL_WARNING, "TensorCreator"),
          memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

    Ort::Value create_tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    {
        std::cout << "\nCreating tensor..." << std::endl;

        size_t total_size = 1;
        for (int64_t dim : shape)
        {
            total_size *= dim;
        }

        // Ensure the data matches the shape size
        if (data.size() != total_size)
        {
            throw std::invalid_argument("Data size does not match tensor shape.");
        }

        // Create an Ort::Value tensor
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(data.data()), data.size(),
            const_cast<int64_t*>(shape.data()), shape.size());

        std::cout << "Tensor created successfully!" << std::endl;
        return tensor; // Return the tensor
    }

    void test_tensor_creation()
    {
        std::cout << "Testing tensor creation..." << std::endl;

        // Example data and shape
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> shape = {2, 3};

        // Create the tensor
        Ort::Value tensor = create_tensor(data, shape);

        // Extract and verify the tensor data
        float* tensor_data = tensor.GetTensorMutableData<float>();
        std::cout << "Tensor data: ";
        for (int i = 0; i < data.size(); ++i)
        {
            std::cout << tensor_data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Tensor creation test passed!" << std::endl;
    }
};

int main()
{
    try
    {
        TensorCreator creator;
        creator.test_tensor_creation();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
