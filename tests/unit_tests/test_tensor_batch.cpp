#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>

class TensorBatchTest
{
private:
    void test_split_batch()
    {
        std::cout << "\nTesting split batch..." << std::endl;

        // Create test data with shape [2, 3] (2 batches, 3 values each)
        std::vector<int64_t> data = {1, 2, 3, 4, 5, 6};
        std::vector<int64_t> shape = {2, 3};

        auto result = florence2::TensorExtension::split_batch(data.data(), shape);

        // Should get 2 tensors, each with shape [1, 3]
        assert(result.size() == 2);
        assert(result[0].size() == 3);
        assert(result[1].size() == 3);

        // Print results
        std::cout << "First batch: ";
        for (const auto &val : result[0])
            std::cout << val << " ";
        std::cout << "\nSecond batch: ";
        for (const auto &val : result[1])
            std::cout << val << " ";
        std::cout << std::endl;

        // Verify values
        std::vector<int64_t> expected1 = {1, 2, 3};
        std::vector<int64_t> expected2 = {4, 5, 6};
        assert(result[0] == expected1);
        assert(result[1] == expected2);

        std::cout << "Split batch test: OK" << std::endl;
    }

    void test_split_batch_remove_dimension()
    {
        std::cout << "\nTesting split batch remove dimension..." << std::endl;

        // Create test data with shape [2, 2, 2]
        std::vector<int64_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {2, 2, 2};

        auto result = florence2::TensorExtension::split_batch_remove_leading_dimension(
            data.data(), shape);

        // Should get 2 tensors, each with shape [2, 2]
        assert(result.size() == 2);
        assert(result[0].size() == 4); // 2*2
        assert(result[1].size() == 4); // 2*2

        // Verify values
        std::vector<int64_t> expected1 = {1, 2, 3, 4};
        std::vector<int64_t> expected2 = {5, 6, 7, 8};
        assert(result[0] == expected1);
        assert(result[1] == expected2);

        std::cout << "Split batch remove dimension test: OK" << std::endl;
    }

    void test_join()
    {
        std::cout << "\nTesting join..." << std::endl;

        // Create test tensors
        std::vector<std::vector<float>> tensors = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}};
        std::vector<int64_t> input_dims = {1, 3}; // Each tensor is [1,3]

        auto result = florence2::TensorExtension::join(tensors, input_dims);

        // Should get [2,3] tensor
        assert(result.size() == 6);

        std::cout << "Joined result: ";
        for (float val : result)
            std::cout << val << " ";
        std::cout << std::endl;

        // Verify values
        std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        assert(result == expected);

        std::cout << "Join test: OK" << std::endl;
    }

    void test_repeat()
    {
        std::cout << "\nTesting repeat..." << std::endl;

        // Create test tensor data
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        std::vector<int64_t> input_dims = {1, 3};
        int count = 2;

        auto result = florence2::TensorExtension::repeat(
            data.data(), input_dims, count);

        // Should get [2,3] tensor
        assert(result.size() == data.size() * count);

        std::cout << "Repeated result: ";
        for (float val : result)
            std::cout << val << " ";
        std::cout << std::endl;

        // Verify values
        std::vector<float> expected = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
        assert(result == expected);

        std::cout << "Repeat test: OK" << std::endl;
    }

public:
    void run_all_tests()
    {
        std::cout << "Starting Batch Operation tests..." << std::endl;

        try
        {
            test_split_batch();
            test_split_batch_remove_dimension();
            test_join();
            test_repeat();

            std::cout << "\nAll batch operation tests passed!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main()
{
    try
    {
        TensorBatchTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}