#include "model/florence2.h"
#include <iostream>
#include <cassert>
#include <fstream>

namespace florence2
{
    namespace test
    {

        class Florence2ModelTest
        {
        public:
            void run_all_tests()
            {
                std::cout << "Starting Florence2Model inference tests..." << std::endl;

                try
                {
                    test_model_initialization();
                    test_tensor_concatenation();
                    test_basic_inference(); // We'll add this after initialization works
                    // test_batch_inference();
                    // test_error_handling();

                    std::cout << "\nAll Florence2Model tests passed!" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: " << e.what() << std::endl;
                    throw;
                }
            }

        private:
            void test_model_initialization()
            {
                std::cout << "\nTesting model initialization..." << std::endl;

                try
                {
                    // Create session options
                    Ort::SessionOptions session_options;
                    // Optional: Configure session options if needed
                    // session_options.SetIntraOpNumThreads(1);
                    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                    // Initialize model
                    Florence2Model model(model_path_, resource_path_, &session_options);
                    std::cout << "Model initialization: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Model initialization failed: " << e.what() << std::endl;
                    throw;
                }
            }

            // Helper functions
            std::vector<uint8_t> load_test_image(const std::string &filename)
            {
                std::ifstream file(filename, std::ios::binary);
                if (!file)
                {
                    throw std::runtime_error("Could not open test image: " + filename);
                }
                return std::vector<uint8_t>(
                    std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());
            }

            void test_tensor_concatenation()
            {
                std::cout << "\nTesting tensor concatenation..." << std::endl;

                // Create small test tensors
                DenseTensor<float> tensor1;
                tensor1.dimensions = {1, 2, 3};                      // Small dimensions for testing
                tensor1.data = std::vector<float>{1, 2, 3, 4, 5, 6}; // 1x2x3 tensor

                DenseTensor<float> tensor2;
                tensor2.dimensions = {1, 3, 3};                                     // Same as tensor1 except middle dimension
                tensor2.data = std::vector<float>{7, 8, 9, 10, 11, 12, 13, 14, 15}; // 1x3x3 tensor

                // Try concatenation
                std::vector<float> result = TensorExtension::concatenate_axis1(
                    tensor1.data.data(), tensor1.dimensions,
                    tensor2.data.data(), tensor2.dimensions);

                // Print dimensions and results
                std::cout << "Tensor1 dims: ";
                for (auto d : tensor1.dimensions)
                    std::cout << d << " ";
                std::cout << "\nTensor2 dims: ";
                for (auto d : tensor2.dimensions)
                    std::cout << d << " ";
                std::cout << "\nResult size: " << result.size() << std::endl;
            }

            void test_basic_inference()
            {
                std::cout << "\nTesting basic inference with captioning..." << std::endl;

                try
                {
                    // Create session options
                    Ort::SessionOptions session_options;

                    // Initialize model
                    Florence2Model model(model_path_, resource_path_, &session_options);

                    // Load test image
                    std::vector<std::vector<uint8_t>> batch;
                    auto image_data = load_test_image("./tests/test_image/top-gun-maverick-tom-cruise-action-movies-2020-movies-5k-8k-7000x5970-562.jpg");
                    batch.push_back(image_data);

                    // Run captioning inference
                    auto results = model.run(TaskType::MORE_DETAILED_CAPTION, batch, "");

                    // Verify results
                    assert(!results.empty() && "Results vector should not be empty");
                    assert(!results[0].pure_text.empty() && "Generated caption should not be empty");

                    // Print the generated caption
                    std::cout << "Generated caption: " << results[0].pure_text << std::endl;
                    std::cout << "Basic captioning inference: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Basic inference test failed: " << e.what() << std::endl;
                    throw;
                }
            }

        private:
            // Update these paths according to your setup
            const std::string model_path_ = "./models";
            const std::string resource_path_ = "./src/resources";
        };

    } // namespace test
} // namespace florence2

int main()
{
    try
    {
        florence2::test::Florence2ModelTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}