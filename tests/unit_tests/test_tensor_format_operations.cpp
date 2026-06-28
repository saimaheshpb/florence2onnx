#include "helper/tensor_extension.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>

class TensorOperationsTest
{
private:
    void test_reorder_tensor()
    {
        std::cout << "\nTesting reorder tensor (RGB reorder)..." << std::endl;

        // Create a 1x3x2x2 tensor with RGB channels
        std::vector<float> input = {
            // R channel (first channel)
            1.0f, 2.0f, // Width values for height 0
            3.0f, 4.0f, // Width values for height 1

            // G channel (second channel)
            5.0f, 6.0f, // Width values for height 0
            7.0f, 8.0f, // Width values for height 1

            // B channel (third channel)
            9.0f, 10.0f, // Width values for height 0
            11.0f, 12.0f // Width values for height 1
        };

        // Output dimensions for BHWC format
        std::vector<int64_t> output_dims = {1, 2, 2, 3}; // [batch=1, height=2, width=2, channels=3]

        auto result = florence2::TensorExtension::reorder_tensor(input.data(), output_dims);

        // For each pixel position (h,w), we should have RGB values together
        std::vector<float> expected = {
            1.0f, 5.0f, 9.0f,  // (h=0,w=0): RGB values
            2.0f, 6.0f, 10.0f, // (h=0,w=1): RGB values
            3.0f, 7.0f, 11.0f, // (h=1,w=0): RGB values
            4.0f, 8.0f, 12.0f  // (h=1,w=1): RGB values
        };

        for (size_t i = 0; i < expected.size(); ++i)
        {
            std::cout << "Position " << i << ": " << result[i]
                      << " (expected: " << expected[i] << ")" << std::endl;
            assert(std::abs(result[i] - expected[i]) < 1e-5f);
        }

        std::cout << "Reorder tensor: OK" << std::endl;
    }
    void test_clip()
    {
        std::cout << "\nTesting clip with vector-sized chunks..." << std::endl;

        // Test with vector-size multiple and remainder
        std::vector<float> input = {
            -2.0f, -1.5f, -1.0f, -0.5f, // First vector chunk
            0.0f, 0.5f, 1.0f, 1.5f,     // Second vector chunk
            2.0f, 2.5f                  // Remainder
        };

        float min_value = -1.0f;
        float max_value = 1.0f;

        auto result = florence2::TensorExtension::clip(
            input.data(), input.size(), min_value, max_value);

        std::cout << "Clipping range: [" << min_value << ", " << max_value << "]" << std::endl;

        // Expected values after clipping
        std::vector<float> expected = {
            -1.0f, -1.0f, -1.0f, -0.5f, // First chunk clipped
            0.0f, 0.5f, 1.0f, 1.0f,     // Second chunk clipped
            1.0f, 1.0f                  // Remainder clipped
        };

        // Verify results
        for (size_t i = 0; i < input.size(); ++i)
        {
            std::cout << "Value " << input[i] << " clipped to " << result[i]
                      << " (expected: " << expected[i] << ")" << std::endl;
            assert(std::abs(result[i] - expected[i]) < 1e-5f);

            // Verify clipping constraints
            assert(result[i] >= min_value && result[i] <= max_value);
        }

        std::cout << "Clip test: OK" << std::endl;
    }

    void test_next_tensor()
    {
        std::cout << "\nTesting next_tensor random generation..." << std::endl;

        // Create random generator with fixed seed for reproducibility
        std::mt19937 random(42);

        size_t size = 1000; // Large enough for statistical tests
        float init_noise_sigma = 1.0f;

        auto result = florence2::TensorExtension::next_tensor(
            random, size, init_noise_sigma);

        // Verify size
        assert(result.size() == size);

        // Calculate statistical properties
        float sum = std::accumulate(result.begin(), result.end(), 0.0f);
        float mean = sum / size;

        float sq_sum = 0.0f;
        for (float val : result)
        {
            sq_sum += val * val;
        }
        float variance = (sq_sum / size) - (mean * mean);
        float std_dev = std::sqrt(variance);

        // For normal distribution with init_noise_sigma:
        // - Mean should be close to 0
        // - Standard deviation should be close to init_noise_sigma
        std::cout << "Statistical properties of generated values:" << std::endl;
        std::cout << "Mean: " << mean << " (should be close to 0)" << std::endl;
        std::cout << "Standard deviation: " << std_dev
                  << " (should be close to " << init_noise_sigma << ")" << std::endl;

        // Test different ranges
        size_t values_in_1sigma = 0;
        size_t values_in_2sigma = 0;
        size_t values_in_3sigma = 0;

        for (float val : result)
        {
            float abs_val = std::abs(val);
            if (abs_val <= init_noise_sigma)
                values_in_1sigma++;
            if (abs_val <= 2 * init_noise_sigma)
                values_in_2sigma++;
            if (abs_val <= 3 * init_noise_sigma)
                values_in_3sigma++;
        }

        // Print distribution statistics
        std::cout << "Values within 1 sigma: "
                  << (100.0f * values_in_1sigma / size) << "% (expect ~68%)" << std::endl;
        std::cout << "Values within 2 sigma: "
                  << (100.0f * values_in_2sigma / size) << "% (expect ~95%)" << std::endl;
        std::cout << "Values within 3 sigma: "
                  << (100.0f * values_in_3sigma / size) << "% (expect ~99.7%)" << std::endl;

        // Verify statistical properties
        assert(std::abs(mean) < 0.1f);                       // Mean should be very close to 0
        assert(std::abs(std_dev - init_noise_sigma) < 0.1f); // Std dev should match sigma

        // Verify normal distribution properties (rough checks)
        float percent_1sigma = 100.0f * values_in_1sigma / size;
        float percent_2sigma = 100.0f * values_in_2sigma / size;
        float percent_3sigma = 100.0f * values_in_3sigma / size;

        assert(percent_1sigma > 60.0f && percent_1sigma < 76.0f);  // ~68%
        assert(percent_2sigma > 90.0f && percent_2sigma < 98.0f);  // ~95%
        assert(percent_3sigma > 98.0f && percent_3sigma < 100.0f); // ~99.7%

        std::cout << "Next tensor test: OK" << std::endl;
    }

public:
    void run_all_tests()
    {
        std::cout << "Starting Tensor Operations tests..." << std::endl;

        try
        {
            test_reorder_tensor();
            test_clip();
            test_next_tensor();

            std::cout << "\nAll tensor operation tests passed!" << std::endl;
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
        TensorOperationsTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}