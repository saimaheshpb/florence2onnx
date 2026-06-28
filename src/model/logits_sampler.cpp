#include "model/logits_sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace florence2
{
    BeamSearchSampler::BeamSearchSampler(
        Ort::Session *top_k_session,
        int top_k,
        int num_beams)
        : top_k_session_(top_k_session), top_k_(top_k), num_beams_(num_beams)
    {
    }

    std::vector<float> BeamSearchSampler::softmax(const std::vector<float> &arr)
    {
        float max_val = *std::max_element(arr.begin(), arr.end());

        std::vector<float> exps(arr.size());
        std::transform(arr.begin(), arr.end(), exps.begin(),
                      [max_val](float x) { return std::exp(x - max_val); });

        float sum_exps = std::accumulate(exps.begin(), exps.end(), 0.0f);

        std::vector<float> softmax_arr(arr.size());
        std::transform(exps.begin(), exps.end(), softmax_arr.begin(),
                      [sum_exps](float x) { return x / sum_exps; });

        return softmax_arr;
    }

    std::vector<std::pair<int64_t, double>> BeamSearchSampler::sample(
        int batch_idx,
        const std::vector<float> &logits,
        const std::vector<int64_t> &dimensions)
    {
        std::cout << "\nBeam Search Sampling Debug:" << std::endl;
        std::cout << "Input dimensions: ";
        for (const auto &dim : dimensions)
            std::cout << dim << " ";
        std::cout << std::endl;

        // Get vocab_size from the last dimension
        int64_t vocab_size = dimensions.back();
        int64_t k = (top_k_ > 0) ? std::min(top_k_, static_cast<int>(vocab_size)) : vocab_size;

        std::cout << "Using k value: " << k << " (vocab size: " << vocab_size << ")" << std::endl;

        // For 3D tensor [batch_size, sequence_length, vocab_size], get last sequence position
        // For 2D tensor [batch_size, vocab_size], get batch slice
        std::vector<float> batch_logits;
        if (dimensions.size() == 3) {
            int64_t seq_length = dimensions[1];
            int64_t batch_offset = batch_idx * seq_length * vocab_size;
            int64_t last_token_offset = batch_offset + (seq_length - 1) * vocab_size;
            batch_logits = std::vector<float>(
                logits.begin() + last_token_offset,
                logits.begin() + last_token_offset + vocab_size
            );
        } else {
            int64_t batch_offset = batch_idx * vocab_size;
            batch_logits = std::vector<float>(
                logits.begin() + batch_offset,
                logits.begin() + batch_offset + vocab_size
            );
        }

        // Create memory info for ONNX
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Prepare input tensor for top-k: [1, vocab_size]
        std::vector<int64_t> logits_dims = {1, vocab_size};
        Ort::Value logits_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            batch_logits.data(),
            batch_logits.size(),
            logits_dims.data(),
            logits_dims.size());

        // Prepare k tensor
        std::vector<int64_t> k_val = {k};
        Ort::Value k_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            k_val.data(),
            k_val.size(),
            std::vector<int64_t>{1}.data(),
            1);

        // Run top-k
        std::vector<const char *> input_names = {"x", "k"};
        std::vector<const char *> output_names = {"v", "i"};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(logits_tensor));
        input_tensors.push_back(std::move(k_tensor));

        std::cout << "Running top-k operation..." << std::endl;

        auto outputs = top_k_session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());

        // Get results
        float *values = outputs[0].GetTensorMutableData<float>();
        int64_t *indices = outputs[1].GetTensorMutableData<int64_t>();

        // Convert to probabilities using softmax
        std::vector<float> values_vec(values, values + k);
        std::vector<float> probabilities = softmax(values_vec);

        // Create result pairs (token_id, log_probability)
        std::vector<std::pair<int64_t, double>> result;
        result.reserve(num_beams_);

        for (int i = 0; i < std::min(num_beams_, static_cast<int>(k)); ++i) {
            result.emplace_back(
                indices[i],
                std::log(probabilities[i]));
        }

        std::cout << "Sampling complete. Selected " << result.size() << " tokens." << std::endl;
        return result;
    }

} // namespace florence2