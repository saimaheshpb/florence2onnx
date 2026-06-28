#include "helper/tensor_extension.h"
#include <stdexcept>
#include <cmath>
#include <random>

namespace florence2
{

    std::vector<float> TensorExtension::multiply_by_float(const float *input_data,
                                                          size_t size, float value)
    {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = input_data[i] * value;
        }
        return result;
    }

    std::vector<float> TensorExtension::divide_by_float(const float *input_data,
                                                        size_t size, float value)
    {
        if (std::abs(value) < 1e-7f)
        {
            throw std::invalid_argument("Division by zero or near-zero value");
        }
        return multiply_by_float(input_data, size, 1.0f / value);
    }

    std::vector<float> TensorExtension::subtract_float(const float *input_data,
                                                       size_t size, float value)
    {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = input_data[i] - value;
        }
        return result;
    }

    std::vector<float> TensorExtension::add_tensors(const float *data1,
                                                    const float *data2,
                                                    size_t size)
    {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = data1[i] + data2[i];
        }
        return result;
    }

    std::vector<float> TensorExtension::sum_tensors(const std::vector<const float *> &data_arrays,
                                                    size_t size)
    {
        if (data_arrays.empty())
        {
            throw std::invalid_argument("No tensors to sum");
        }

        std::vector<float> result(size, 0.0f); // Initialize with zeros

        for (const float *data : data_arrays)
        {
            for (size_t i = 0; i < size; ++i)
            {
                result[i] += data[i];
            }
        }
        return result;
    }

    std::vector<float> TensorExtension::subtract_tensors(const float *data1,
                                                         const float *data2,
                                                         size_t size)
    {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = data1[i] - data2[i];
        }
        return result;
    }

    void TensorExtension::divide_by_inplace(float *data, size_t size, float value)
    {
        if (std::abs(value) < 1e-7f)
        {
            throw std::invalid_argument("Division by zero or near-zero value");
        }
        float inv_value = 1.0f / value;
        for (size_t i = 0; i < size; ++i)
        {
            data[i] *= inv_value; // Multiply by inverse is faster than division
        }
    }

    void TensorExtension::multiply_by_inplace(float *data, size_t size, float value)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] *= value;
        }
    }

    void TensorExtension::abs_inplace(float *data, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = std::abs(data[i]);
        }
    }

    void TensorExtension::multiply_inplace(float *data1, const float *data2, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data1[i] *= data2[i];
        }
    }

    void TensorExtension::divide_inplace(float *data1, const float *data2, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            if (std::abs(data2[i]) < 1e-7f)
            {
                throw std::invalid_argument("Division by zero or near-zero value at index " + std::to_string(i));
            }
            data1[i] /= data2[i];
        }
    }

    void TensorExtension::add_inplace(float *data1, const float *data2, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data1[i] += data2[i];
        }
    }

    void TensorExtension::subtract_inplace(float *data1, const float *data2, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data1[i] -= data2[i];
        }
    }

    std::vector<float> TensorExtension::reorder_tensor(const float *input_data,
                                                       const std::vector<int64_t> &dimensions)
    {
        // dimensions should be the output dimensions
        std::vector<float> result(dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3]);

        // Hardcoded for batch=1, channels=3 (RGB)
        for (int y = 0; y < dimensions[1]; y++)
        {
            for (int x = 0; x < dimensions[2]; x++)
            {
                // Write RGB channels in new order
                // input is in [0, channel, y, x] format
                // output is in [0, y, x, channel] format
                size_t out_idx = y * dimensions[2] * 3 + x * 3;

                // R channel
                result[out_idx] = input_data[y * dimensions[2] + x];
                // G channel
                result[out_idx + 1] = input_data[dimensions[1] * dimensions[2] + y * dimensions[2] + x];
                // B channel
                result[out_idx + 2] = input_data[2 * dimensions[1] * dimensions[2] + y * dimensions[2] + x];
            }
        }
        return result;
    }

    std::vector<float> TensorExtension::clip(const float *data,
                                             size_t size,
                                             float min_value,
                                             float max_value)
    {
        // Get vector size for SIMD operations
        const size_t vector_size = 4; // Typical SSE vector size for floats
        std::vector<float> result(size);

        // Process vector-sized chunks
        size_t vector_count = size / vector_size;
        for (size_t i = 0; i < vector_count; i++)
        {
            for (size_t j = 0; j < vector_size; j++)
            {
                size_t idx = i * vector_size + j;
                result[idx] = std::min(std::max(data[idx], min_value), max_value);
            }
        }

        // Handle remaining elements
        for (size_t i = vector_count * vector_size; i < size; i++)
        {
            result[i] = std::min(std::max(data[i], min_value), max_value);
        }

        return result;
    }

    std::vector<float> TensorExtension::next_tensor(std::mt19937 &random,
                                                    size_t size,
                                                    float init_noise_sigma)
    {
        std::vector<float> result(size);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < size; i++)
        {
            // Generate two uniform random numbers
            float u1 = dist(random);
            float u2 = dist(random);

            // Box-Muller transform to generate normal distribution
            float radius = std::sqrt(-2.0f * std::log(u1));
            float theta = 2.0f * M_PI * u2;
            float standard_normal_rand = radius * std::cos(theta);

            // Apply noise sigma
            result[i] = standard_normal_rand * init_noise_sigma;
        }

        return result;
    }

    std::vector<std::vector<int64_t>> TensorExtension::split_batch(
        const int64_t *data,
        const std::vector<int64_t> &shape)
    {

        const int64_t count = shape[0];
        auto dimensions = shape; // Copy original dimensions
        dimensions[0] = 1;       // Set batch size to 1 for each split

        int64_t new_length = 1;
        for (size_t i = 1; i < shape.size(); ++i)
        {
            new_length *= shape[i];
        }

        std::vector<std::vector<int64_t>> result;
        result.reserve(count);

        for (int64_t i = 0; i < count; ++i)
        {
            const int64_t *start = data + (i * new_length);
            result.emplace_back(start, start + new_length);
        }

        return result;
    }

    std::vector<std::vector<int64_t>> TensorExtension::split_batch_remove_leading_dimension(
        const int64_t *data,
        const std::vector<int64_t> &shape)
    {

        const int64_t count = shape[0];

        // Remove first dimension
        std::vector<int64_t> dimensions(shape.begin() + 1, shape.end());

        int64_t new_length = 1;
        for (size_t i = 1; i < shape.size(); ++i)
        {
            new_length *= shape[i];
        }

        std::vector<std::vector<int64_t>> result;
        result.reserve(count);

        for (int64_t i = 0; i < count; ++i)
        {
            const int64_t *start = data + (i * new_length);
            result.emplace_back(start, start + new_length);
        }

        return result;
    }

    std::vector<float> TensorExtension::join(
        const std::vector<std::vector<float>> &tensors,
        const std::vector<int64_t> &input_dims,
        int axis)
    {

        if (axis != 0)
        {
            throw std::invalid_argument("Only axis 0 is supported");
        }
        if (tensors.empty())
        {
            throw std::invalid_argument("No tensors to join");
        }

        // Calculate single tensor length and total length
        const size_t single_length = tensors[0].size();
        const size_t total_length = single_length * tensors.size();

        std::vector<float> result(total_length);

        // Copy each tensor's data
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            if (tensors[i].size() != single_length)
            {
                throw std::invalid_argument("All tensors must have the same size");
            }
            std::copy(tensors[i].begin(),
                      tensors[i].end(),
                      result.begin() + i * single_length);
        }

        return result;
    }

    std::vector<float> TensorExtension::repeat(
        const float *tensor_data,
        const std::vector<int64_t> &input_dims,
        int count,
        int axis)
    {

        if (axis != 0)
        {
            throw std::invalid_argument("Only axis 0 is supported");
        }

        // Calculate length of input tensor
        int64_t length = 1;
        for (const auto &dim : input_dims)
        {
            length *= dim;
        }

        // Create result buffer with repeated data
        std::vector<float> result(length * count);

        for (int i = 0; i < count; ++i)
        {
            std::copy(tensor_data,
                      tensor_data + length,
                      result.begin() + i * length);
        }

        return result;
    }

    std::vector<float> TensorExtension::normalize_one_one_to_zero_one(
        const float *data, size_t size)
    {
        std::vector<float> result(size);
#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {
            result[i] = data[i] / 2.0f + 0.5f;
        }
        return result;
    }

    std::vector<float> TensorExtension::normalize_zero_one_to_one_one(
        const float *data, size_t size)
    {
        std::vector<float> result(size);
#pragma omp parallel for
        for (size_t i = 0; i < size; i++)
        {
            result[i] = 2.0f * data[i] - 1.0f;
        }
        return result;
    }

    int64_t TensorExtension::calculate_size(const std::vector<int64_t> &dimensions)
    {
        if (dimensions.empty())
        {
            throw std::invalid_argument("Dimensions vector cannot be empty");
        }

        return std::accumulate(dimensions.begin(), dimensions.end(),
                               static_cast<int64_t>(1), std::multiplies<int64_t>());
    }

    std::vector<float> TensorExtension::ones_float(const std::vector<int64_t> &dimensions)
    {
        auto size = calculate_size(dimensions);
        return std::vector<float>(size, 1.0f);
    }

    std::vector<int64_t> TensorExtension::ones_long(const std::vector<int64_t> &dimensions,
                                                    int64_t value)
    {
        auto size = calculate_size(dimensions);
        return std::vector<int64_t>(size, value);
    }

    // Float version
    std::vector<float> TensorExtension::concatenate_axis1(
        const float *tensor1_data,
        const std::vector<int64_t> &tensor1_dims,
        const float *tensor2_data,
        const std::vector<int64_t> &tensor2_dims)
    {
        // Check dimensions compatibility
        if (tensor1_dims[0] != tensor2_dims[0] || tensor1_dims[2] != tensor2_dims[2])
        {
            throw std::invalid_argument("Tensor dimensions are not compatible for axis 1 concatenation");
        }

        // Calculate new dimensions
        std::vector<int64_t> output_dims = tensor1_dims;
        output_dims[1] += tensor2_dims[1];

        // Allocate result
        size_t total_size = output_dims[0] * output_dims[1] * output_dims[2];
        std::vector<float> result(total_size);

        // Copy data from first tensor
        for (int i = 0; i < tensor1_dims[0]; i++)
        {
            for (int j = 0; j < tensor1_dims[1]; j++)
            {
                for (int k = 0; k < tensor1_dims[2]; k++)
                {
                    size_t out_idx = i * (output_dims[1] * output_dims[2]) +
                                     j * output_dims[2] + k;
                    size_t in_idx = i * (tensor1_dims[1] * tensor1_dims[2]) +
                                    j * tensor1_dims[2] + k;
                    result[out_idx] = tensor1_data[in_idx];
                }
            }
        }

        // Copy data from second tensor
        for (int i = 0; i < tensor2_dims[0]; i++)
        {
            for (int j = 0; j < tensor2_dims[1]; j++)
            {
                for (int k = 0; k < tensor2_dims[2]; k++)
                {
                    size_t out_idx = i * (output_dims[1] * output_dims[2]) +
                                     (j + tensor1_dims[1]) * output_dims[2] + k;
                    size_t in_idx = i * (tensor2_dims[1] * tensor2_dims[2]) +
                                    j * tensor2_dims[2] + k;
                    result[out_idx] = tensor2_data[in_idx];
                }
            }
        }

        return result;
    }

    std::vector<int64_t> TensorExtension::concatenate_axis1(
        const int64_t *tensor1_data,
        const std::vector<int64_t> &tensor1_dims,
        const int64_t *tensor2_data,
        const std::vector<int64_t> &tensor2_dims)
    {
        // For 2D tensors (attention masks)
        if (tensor1_dims.size() != 2 || tensor2_dims.size() != 2 ||
            tensor1_dims[0] != tensor2_dims[0])
        {
            throw std::invalid_argument("Tensor dimensions are not compatible for axis 1 concatenation");
        }

        // Calculate new dimensions
        std::vector<int64_t> output_dims = tensor1_dims;
        output_dims[1] += tensor2_dims[1];

        // Allocate result
        size_t total_size = output_dims[0] * output_dims[1];
        std::vector<int64_t> result(total_size);

        // Copy data from first tensor
        for (int i = 0; i < tensor1_dims[0]; i++)
        {
            for (int j = 0; j < tensor1_dims[1]; j++)
            {
                size_t out_idx = i * output_dims[1] + j;
                size_t in_idx = i * tensor1_dims[1] + j;
                result[out_idx] = tensor1_data[in_idx];
            }
        }

        // Copy data from second tensor
        for (int i = 0; i < tensor2_dims[0]; i++)
        {
            for (int j = 0; j < tensor2_dims[1]; j++)
            {
                size_t out_idx = i * output_dims[1] + (j + tensor1_dims[1]);
                size_t in_idx = i * tensor2_dims[1] + j;
                result[out_idx] = tensor2_data[in_idx];
            }
        }

        return result;
    }

} // namespace florence2