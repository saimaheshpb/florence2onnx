#pragma once
#include <vector>
#include <random>

namespace florence2
{

    class TensorExtension
    {
    public:
        // Basic Operations (non-mutating)
        static std::vector<float> multiply_by_float(const float *input_data, size_t size, float value);
        static std::vector<float> divide_by_float(const float *input_data, size_t size, float value);
        static std::vector<float> subtract_float(const float *input_data, size_t size, float value);
        static std::vector<float> add_tensors(const float *data1, const float *data2, size_t size);
        static std::vector<float> sum_tensors(const std::vector<const float *> &data_arrays, size_t size);
        static std::vector<float> subtract_tensors(const float *data1, const float *data2, size_t size);

        // Inplace Operations (mutating)
        static void divide_by_inplace(float *data, size_t size, float value);
        static void multiply_by_inplace(float *data, size_t size, float value);
        static void abs_inplace(float *data, size_t size);
        static void multiply_inplace(float *data1, const float *data2, size_t size);
        static void divide_inplace(float *data1, const float *data2, size_t size);
        static void add_inplace(float *data1, const float *data2, size_t size);
        static void subtract_inplace(float *data1, const float *data2, size_t size);

        // Tensor Operations
        static std::vector<float> reorder_tensor(const float *input_data, const std::vector<int64_t> &dimensions);
        static std::vector<float> clip(const float *data, size_t size, float min_value, float max_value);
        static std::vector<float> next_tensor(std::mt19937 &random, size_t size, float init_noise_sigma);

        // Batch Operations
        static std::vector<std::vector<int64_t>> split_batch(
            const int64_t *data,
            const std::vector<int64_t> &shape);

        static std::vector<std::vector<int64_t>> split_batch_remove_leading_dimension(
            const int64_t *data,
            const std::vector<int64_t> &shape);

        static std::vector<float> join(
            const std::vector<std::vector<float>> &tensors,
            const std::vector<int64_t> &input_dims,
            int axis = 0);

        static std::vector<float> repeat(
            const float *tensor_data,
            const std::vector<int64_t> &input_dims,
            int count,
            int axis = 0);

        // Normalization Operations
        static std::vector<float> normalize_one_one_to_zero_one(
            const float *data, size_t size);

        static std::vector<float> normalize_zero_one_to_one_one(
            const float *data, size_t size);

        // Creates a tensor filled with ones (float)
        static std::vector<float> ones_float(const std::vector<int64_t> &dimensions);

        // Creates a tensor filled with ones or specified value (long)
        static std::vector<int64_t> ones_long(const std::vector<int64_t> &dimensions,
                                              int64_t value = 1);

        static std::vector<float> concatenate_axis1(const float *tensor1_data, const std::vector<int64_t> &tensor1_dims, const float *tensor2_data, const std::vector<int64_t> &tensor2_dims);

        static std::vector<int64_t> concatenate_axis1(const int64_t *tensor1_data, const std::vector<int64_t> &tensor1_dims, const int64_t *tensor2_data, const std::vector<int64_t> &tensor2_dims);

    private:
        // Helper to calculate total size from dimensions
        static int64_t calculate_size(const std::vector<int64_t> &dimensions);
    };

} // namespace florence2