#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include "downloader/florence_model_downloader.h"
#include "helper/tensor_extension.h"
#include "logits_processor.h"
#include "logits_sampler.h"
#include "model/clip_image_processor.h"
#include "model/model_source.h"
#include "post_processing/decoder_config.h"
#include "post_processing/post_processor.h"
#include "shared_types.h"
#include "stopping_criteria.h"
#include "tensor_operation_registry.h"
#include "tokenizer/florence2_tokenizer.h"

namespace florence2 {
template <typename T> struct DenseTensor {
  std::vector<T> data;
  std::vector<int64_t> dimensions;

  size_t size() const {
    size_t total = 1;
    for (auto dim : dimensions) {
      total *= dim;
    }
    return total;
  }
};

class Florence2Model {
public:
  Florence2Model(
      const std::string &model_path, const std::string &resource_path,
      Ort::SessionOptions *session_options); // TODO Rename Resource Path Name
  bool initialize_models(const std::string &model_path);
  ~Florence2Model() = default;

  std::vector<std::string> generate_loop_v2(
      const std::vector<float> &prefill_logits_data,
      const std::vector<int64_t> &prefill_logits_shape,
      const std::vector<std::vector<float>> &decoder_key_data,
      const std::vector<std::vector<int64_t>> &decoder_key_shapes,
      const std::vector<std::vector<float>> &decoder_value_data,
      const std::vector<std::vector<int64_t>> &decoder_value_shapes,
      const std::vector<std::vector<float>> &encoder_key_data,
      const std::vector<std::vector<int64_t>> &encoder_key_shapes,
      const std::vector<std::vector<float>> &encoder_value_data,
      const std::vector<std::vector<int64_t>> &encoder_value_shapes,
      const DenseTensor<float> &encoder_hidden_states,
      const DenseTensor<int64_t> &attention_mask,
      const Ort::RunOptions &run_options, const int64_t sequence_length);

  std::vector<std::string> generate_loop_v4(
      const std::vector<float> &prefill_logits_data,
      const std::vector<int64_t> &prefill_logits_shape,
      const std::vector<std::vector<float>> &decoder_key_data,
      const std::vector<std::vector<int64_t>> &decoder_key_shapes,
      const std::vector<std::vector<float>> &decoder_value_data,
      const std::vector<std::vector<int64_t>> &decoder_value_shapes,
      const std::vector<std::vector<float>> &encoder_key_data,
      const std::vector<std::vector<int64_t>> &encoder_key_shapes,
      const std::vector<std::vector<float>> &encoder_value_data,
      const std::vector<std::vector<int64_t>> &encoder_value_shapes,
      const DenseTensor<float> &encoder_hidden_states,
      const DenseTensor<int64_t> &attention_mask,
      const Ort::RunOptions &run_options, const int64_t sequence_length);

  std::vector<std::string> generate_loop_v5(
      const std::vector<float> &prefill_logits_data,
      const std::vector<int64_t> &prefill_logits_shape,
      const std::vector<std::vector<float>> &decoder_key_data,
      const std::vector<std::vector<int64_t>> &decoder_key_shapes,
      const std::vector<std::vector<float>> &decoder_value_data,
      const std::vector<std::vector<int64_t>> &decoder_value_shapes,
      const std::vector<std::vector<float>> &encoder_key_data,
      const std::vector<std::vector<int64_t>> &encoder_key_shapes,
      const std::vector<std::vector<float>> &encoder_value_data,
      const std::vector<std::vector<int64_t>> &encoder_value_shapes,
      const DenseTensor<float> &encoder_hidden_states,
      const DenseTensor<int64_t> &attention_mask,
      const Ort::RunOptions &run_options, const int64_t sequence_length);

  std::vector<std::string> generate_loop_v3(
      const std::vector<float> &prefill_logits_data,
      const std::vector<int64_t> &prefill_logits_shape,
      const std::vector<std::vector<float>> &decoder_key_data,
      const std::vector<std::vector<int64_t>> &decoder_key_shapes,
      const std::vector<std::vector<float>> &decoder_value_data,
      const std::vector<std::vector<int64_t>> &decoder_value_shapes,
      const std::vector<std::vector<float>> &encoder_key_data,
      const std::vector<std::vector<int64_t>> &encoder_key_shapes,
      const std::vector<std::vector<float>> &encoder_value_data,
      const std::vector<std::vector<int64_t>> &encoder_value_shapes,
      const DenseTensor<float> &encoder_hidden_states,
      const DenseTensor<int64_t> &attention_mask,
      const Ort::RunOptions &run_options, const int64_t sequence_length);

  std::vector<FlorenceResults>
  run(TaskType task, const std::vector<std::vector<uint8_t>> &img_data,
      const std::string &text_input = "");

private:
  std::vector<const char *> get_decoder_input_names() {
    std::vector<const char *> input_names = {
        "use_cache_branch", "inputs_embeds", "encoder_hidden_states",
        "encoder_attention_mask"};

    const char *past_kv_names[] = {
        "past_key_values.0.decoder.key", "past_key_values.0.decoder.value",
        "past_key_values.0.encoder.key", "past_key_values.0.encoder.value",
        "past_key_values.1.decoder.key", "past_key_values.1.decoder.value",
        "past_key_values.1.encoder.key", "past_key_values.1.encoder.value",
        "past_key_values.2.decoder.key", "past_key_values.2.decoder.value",
        "past_key_values.2.encoder.key", "past_key_values.2.encoder.value",
        "past_key_values.3.decoder.key", "past_key_values.3.decoder.value",
        "past_key_values.3.encoder.key", "past_key_values.3.encoder.value",
        "past_key_values.4.decoder.key", "past_key_values.4.decoder.value",
        "past_key_values.4.encoder.key", "past_key_values.4.encoder.value",
        "past_key_values.5.decoder.key", "past_key_values.5.decoder.value",
        "past_key_values.5.encoder.key", "past_key_values.5.encoder.value"};
    for (const auto &name : past_kv_names) {
      input_names.push_back(name);
    }
    return input_names;
  }

  std::vector<const char *> get_decoder_output_names() {
    std::vector<const char *> output_names = {"logits"};
    const char *present_names[] = {
        "present.0.decoder.key", "present.0.decoder.value",
        "present.0.encoder.key", "present.0.encoder.value",
        "present.1.decoder.key", "present.1.decoder.value",
        "present.1.encoder.key", "present.1.encoder.value",
        "present.2.decoder.key", "present.2.decoder.value",
        "present.2.encoder.key", "present.2.encoder.value",
        "present.3.decoder.key", "present.3.decoder.value",
        "present.3.encoder.key", "present.3.encoder.value",
        "present.4.decoder.key", "present.4.decoder.value",
        "present.4.encoder.key", "present.4.encoder.value",
        "present.5.decoder.key", "present.5.decoder.value",
        "present.5.encoder.key", "present.5.encoder.value"};
    for (const auto &name : present_names) {
      output_names.push_back(name);
    }
    return output_names;
  }
  std::vector<Ort::Value> encoder_past_key_values_;
  std::unique_ptr<Ort::Session>
      session_tensor_ops_; // For tensor operations like TopK
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,
                "Florence2Model"}; // ONNX Runtime environment
  std::pair<DenseTensor<int64_t>, DenseTensor<int64_t>>
  get_text_inputs(const std::vector<std::string> &sentences);

  std::vector<Ort::Value>
  init_past_key_values(const NormalizedConfig &config,
                       const Ort::MemoryInfo &memory_info,
                       const int64_t sequence_length);

  // Update merge_input_ids signature too
  std::pair<std::pair<DenseTensor<float>, DenseTensor<int64_t>>, const int64_t>
  merge_input_ids_with_image_features(
      const DenseTensor<float> &inputs_embeds,
      const DenseTensor<float> &image_features,
      const DenseTensor<int64_t> &attention_mask);

  std::shared_ptr<FlorenceModelDownloader> model_downloader_;
  static const std::unordered_map<TaskType, std::string>
      task_prompts_without_inputs_;
  static const std::unordered_map<TaskType, std::string>
      task_prompts_with_inputs_;

  std::unique_ptr<Ort::Session> create_session(IModelSource *source,
                                               IModelSource::Model model);

  std::string construct_prompts(TaskType task_type,
                                const std::string *text_input = nullptr);

  // Member variables for the model components
  std::shared_ptr<Florence2Tokenizer> tokenizer_;
  std::unique_ptr<CLIPImageProcessor> image_processor_;
  std::unique_ptr<Florence2PostProcessor> post_processor_;

  // ONNX Runtime sessions
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_decoder_merged_;
  std::unique_ptr<Ort::Session> session_embed_tokens_;
  std::unique_ptr<Ort::Session> session_encoder_;
  std::unique_ptr<Ort::Session> session_vision_encoder_;
  std::unique_ptr<Ort::Session> session_decoder_;

  std::vector<std::string>
  generate_loop(const DenseTensor<int64_t> &attention_mask,
                const DenseTensor<float> &encoder_outputs,
                const Ort::RunOptions &run_options, const int64_t merge_size);

  // Helper function for generation
  static std::string
  decode_single(const std::shared_ptr<Florence2Tokenizer> &tokenizer,
                const std::vector<int64_t> &token_ids);

  static std::string clean_up_tokenization(const std::string &text);

  void initialize_session_options();

  void verify_execution_provider(const Ort::Session &session);
};

} // namespace florence2