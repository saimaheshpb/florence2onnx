#include "model/florence2.h"
#ifdef ANDROID
#include "../../third_party/onnxruntime-android/include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include <android/log.h>
#define LOG_TAG "Florence2Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#include <cstdarg>
#define ANDROID_LOG_INFO 1
#define ANDROID_LOG_ERROR 2

inline void __android_log_print(int prio, const char *tag, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (prio == ANDROID_LOG_ERROR) {
        fprintf(stderr, "[ERROR][%s] ", tag);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
    } else {
        printf("[INFO][%s] ", tag);
        vprintf(fmt, args);
        printf("\n");
    }
    va_end(args);
}

#define LOG_TAG "Florence2Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif
#include <stdexcept>

namespace florence2 {
    const std::unordered_map<TaskType, std::string>
    Florence2Model::task_prompts_without_inputs_ = {
        {TaskType::OCR, "What is the text in the image?"},
        {
            TaskType::OCR_WITH_REGION,
            "What is the text in the image, with regions?"
        },
        {TaskType::CAPTION, "What does the image describe?"},
        {
            TaskType::DETAILED_CAPTION,
            "Describe in detail what is shown in the image."
        },
        {
            TaskType::MORE_DETAILED_CAPTION,
            "Describe with a paragraph what is shown in the image."
        },
        {TaskType::OD, "Locate the objects with category name in the image."},
        {
            TaskType::DENSE_REGION_CAPTION,
            "Locate the objects in the image, with their descriptions."
        },
        {
            TaskType::REGION_PROPOSAL,
            "Locate the region proposals in the image."
        }
    };

    const std::unordered_map<TaskType, std::string>
    Florence2Model::task_prompts_with_inputs_ = {
        {
            TaskType::CAPTION_TO_PHRASE_GROUNDING,
            "Locate the phrases in the caption: {}"
        },
        {
            TaskType::REFERRING_EXPRESSION_SEGMENTATION,
            "Locate {} in the image with mask"
        },
        {
            TaskType::REGION_TO_SEGMENTATION,
            "What is the polygon mask of region {}"
        },
        {TaskType::OPEN_VOCABULARY_DETECTION, "Locate {} in the image."},
        {TaskType::REGION_TO_CATEGORY, "What is the region {}?"},
        {TaskType::REGION_TO_DESCRIPTION, "What does the region {} describe?"},
        {TaskType::REGION_TO_OCR, "What text is in the region {}?"}
    };

    std::unique_ptr<Ort::Session>
    Florence2Model::create_session(IModelSource *source,
                                   IModelSource::Model model) {
        std::string model_path;
        if (source->try_get_model_path(model, model_path)) {
            return std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                                  session_options_);
        } else {
            auto model_bytes = source->get_model_bytes(model);
            return std::make_unique<Ort::Session>(env_, model_bytes.data(),
                                                  model_bytes.size(), session_options_);
        }
    }

    void Florence2Model::initialize_session_options() {
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC);
        session_options_.EnableCpuMemArena();
        // session_options_.EnableProfiling(
        //     "/data/data/com.yourpackage.florence2/files/profile.json");

#ifdef ANDROID
        // More specific NNAPI debug flags
        session_options_.AddConfigEntry("ep.nnapi.execution_preference",
                                        "sustained_speed");
        session_options_.AddConfigEntry("ep.nnapi.accelerator_name",
                                        "nnapi-reference");
        session_options_.AddConfigEntry("ep.nnapi.cpu_disabled", "true");
        session_options_.AddConfigEntry("ep.nnapi.fp16_enabled", "true");

        // Logging configuration
        session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
        session_options_.SetLogId("Florence2Model");

        try {
            uint32_t nnapi_flags = NNAPI_FLAG_USE_FP16 | // Enable FP16
                                   NNAPI_FLAG_USE_NCHW | // Use NCHW layout
                                   NNAPI_FLAG_CPU_DISABLED; //

            __android_log_print(ANDROID_LOG_INFO, "Florence2Native",
                                "Attempting NNAPI initialization with flags: 0x%x",
                                nnapi_flags);

            auto status = OrtSessionOptionsAppendExecutionProvider_Nnapi(
                session_options_, nnapi_flags);
            if (!status) {
                __android_log_print(ANDROID_LOG_INFO, "Florence2Native",
                                    "NNAPI initialized successfully with flags: 0x%x",
                                    nnapi_flags);
            } else {
                __android_log_print(ANDROID_LOG_ERROR, "Florence2Native",
                                    "NNAPI initialization failed with flags: 0x%x",
                                    nnapi_flags);
                throw std::runtime_error("NNAPI initialization failed");
            }
        } catch (const std::exception &e) {
            __android_log_print(ANDROID_LOG_ERROR, "Florence2Native",
                                "Failed to add NNAPI provider: %s", e.what());
            throw;
        }
#else
        // Logging configuration
        session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
        session_options_.SetLogId("Florence2Model");
#endif
    }

    bool Florence2Model::initialize_models(const std::string &model_path) {
        try {
            // Initialize session options first
            initialize_session_options();

            // Create and initialize downloader
            model_downloader_ = std::make_shared<FlorenceModelDownloader>(model_path);

            // Download models if needed
            bool success =
                    model_downloader_->download_models([](const IStatus &status) {
                        std::cout << status.get_message() << std::endl;
                    });

            if (!success || !model_downloader_->is_ready()) {
                std::cerr << "Model downloader initialization failed" << std::endl;
                return false;
            }

            // Initialize ONNX sessions using the downloaded models
            try {
                session_decoder_merged_ = create_session(
                    model_downloader_.get(), IModelSource::Model::DecoderModelMerged);
                std::cout << "Decoder Merged session created successfully" << std::endl;

                session_embed_tokens_ = create_session(model_downloader_.get(),
                                                       IModelSource::Model::EmbedTokens);
                std::cout << "Embed Tokens session created successfully" << std::endl;

                session_encoder_ = create_session(model_downloader_.get(),
                                                  IModelSource::Model::EncoderModel);
                std::cout << "Encoder session created successfully" << std::endl;

                session_vision_encoder_ = create_session(
                    model_downloader_.get(), IModelSource::Model::VisionEncoder);
                std::cout << "Vision Encoder session created successfully" << std::endl;

                session_decoder_ = create_session(model_downloader_.get(),
                                                  IModelSource::Model::DecoderModel);
                std::cout << "Decoder session created successfully" << std::endl;
                return true;
            } catch (const Ort::Exception &e) {
                std::cerr << "ONNX Runtime error during session creation: " << e.what()
                        << std::endl;
                return false;
            } catch (const std::exception &e) {
                std::cerr << "Failed to initialize ONNX sessions: " << e.what()
                        << std::endl;
                return false;
            }
        } catch (const std::exception &e) {
            std::cerr << "Global initialization error: " << e.what() << std::endl;
            return false;
        }
    }

    Florence2Model::Florence2Model(const std::string &model_path,
                                   const std::string &resource_path,
                                   Ort::SessionOptions *session_options) {
        try {
            // Initialize session options
            if (session_options) {
                session_options_ = std::move(*session_options);
            }

            session_tensor_ops_ =
                    TensorOperationRegistry::top_k_session(session_options_, env_);

            // First initialize models and sessions
            if (!initialize_models(model_path)) {
                throw std::runtime_error("Failed to initialize models");
            }

            // Set resource path for tokenizer initialization
            ResourceLoader::set_resource_path(resource_path);

            // Initialize tokenizer
            tokenizer_ = Florence2Tokenizer::init();
            if (!tokenizer_) {
                throw std::runtime_error("Failed to initialize tokenizer");
            }

            // Initialize image processor
            CLIPConfig config{
                {0.485f, 0.456f, 0.406f}, // image_mean
                577, // image_seq_length
                {0.229f, 0.224f, 0.225f}, // image_std
                0.00392156862745098f, // rescale_factor (1/255)
                500, // crop_height
                500 // crop_width
            };

            image_processor_ = std::make_unique<CLIPImageProcessor>(config);

            // Initialize post processor
            post_processor_ = std::make_unique<Florence2PostProcessor>();
        } catch (const std::exception &e) {
            throw std::runtime_error(
                std::string("Florence2Model initialization failed: ") + e.what());
        }
    }

    std::pair<DenseTensor<int64_t>, DenseTensor<int64_t> >
    Florence2Model::get_text_inputs(const std::vector<std::string> &sentences) {
        const size_t num_sentences = sentences.size();
        auto encoded = tokenizer_->encode(sentences);
        const size_t token_count = encoded[0].first.size();

        // Calculate total sizes
        size_t total_input_size = 0;
        size_t total_attention_size = 0;
        for (const auto &pair: encoded) {
            total_input_size += pair.first.size();
            total_attention_size += pair.second.size();
        }

        // Initialize tensors with proper dimensions
        DenseTensor<int64_t> input_ids;
        DenseTensor<int64_t> attention_mask;

        input_ids.dimensions = {
            static_cast<int64_t>(num_sentences),
            static_cast<int64_t>(token_count)
        };
        attention_mask.dimensions = input_ids.dimensions;

        input_ids.data.resize(total_input_size);
        attention_mask.data.resize(total_attention_size);

        // Use pointers for efficient copying
        int64_t *input_ptr = input_ids.data.data();
        int64_t *attention_ptr = attention_mask.data.data();

        for (const auto &[ids, mask]: encoded) {
            std::memcpy(input_ptr, ids.data(), ids.size() * sizeof(int64_t));
            std::memcpy(attention_ptr, mask.data(), mask.size() * sizeof(int64_t));

            input_ptr += ids.size();
            attention_ptr += mask.size();
        }

        return {std::move(input_ids), std::move(attention_mask)};
    }

    std::pair<std::pair<DenseTensor<float>, DenseTensor<int64_t> >, const int64_t>
    Florence2Model::merge_input_ids_with_image_features(
        const DenseTensor<float> &inputs_embeds,
        const DenseTensor<float> &image_features,
        const DenseTensor<int64_t> &attention_mask) {
        std::cout << "\n=== Merging Features Debug ===" << std::endl;

        // Debug input tensors
        std::cout << "Image Features:" << std::endl;
        std::cout << "- Dimensions: ";
        for (auto d: image_features.dimensions)
            std::cout << d << " ";
        std::cout << "\n- Data size: " << image_features.data.size() << std::endl;

        std::cout << "\nInput Embeds:" << std::endl;
        std::cout << "- Dimensions: ";
        for (auto d: inputs_embeds.dimensions)
            std::cout << d << " ";
        std::cout << "\n- Data size: " << inputs_embeds.data.size() << std::endl;

        std::cout << "\nAttention Mask:" << std::endl;
        std::cout << "- Dimensions: ";
        for (auto d: attention_mask.dimensions)
            std::cout << d << " ";
        std::cout << "\n- Data size: " << attention_mask.data.size() << std::endl;

        try {
            // Create image attention mask
            std::vector<int64_t> image_attention_dims = {
                image_features.dimensions[0], // batch_size
                image_features.dimensions[1] // seq_length
            };
            std::vector<int64_t> image_attention_mask =
                    TensorExtension::ones_long(image_attention_dims);

            std::cout << "\nImage Attention Mask:" << std::endl;
            std::cout << "- Dimensions: ";
            for (auto d: image_attention_dims)
                std::cout << d << " ";
            std::cout << "\n- Data size: " << image_attention_mask.size() << std::endl;

            // Concatenate embeddings
            std::cout << "\nAttempting embeddings concatenation..." << std::endl;
            DenseTensor<float> merged_embeds;
            merged_embeds.data = TensorExtension::concatenate_axis1(
                image_features.data.data(), image_features.dimensions,
                inputs_embeds.data.data(), inputs_embeds.dimensions);
            merged_embeds.dimensions = {
                image_features.dimensions[0],
                image_features.dimensions[1] +
                inputs_embeds.dimensions[1],
                image_features.dimensions[2]
            };
            std::cout << "Embeddings concatenation successful!" << std::endl;
            std::cout << "- Merged dimensions: ";
            for (auto d: merged_embeds.dimensions)
                std::cout << d << " ";
            std::cout << "\n- Merged data size: " << merged_embeds.data.size()
                    << std::endl;

            // Concatenate attention masks
            std::cout << "\nAttempting attention mask concatenation..." << std::endl;
            DenseTensor<int64_t> merged_attention;
            merged_attention.data = TensorExtension::concatenate_axis1(
                image_attention_mask.data(), image_attention_dims,
                attention_mask.data.data(), attention_mask.dimensions);
            merged_attention.dimensions = {
                image_features.dimensions[0],
                image_features.dimensions[1] +
                attention_mask.dimensions[1]
            };
            std::cout << "Attention mask concatenation successful!" << std::endl;
            std::cout << "- Merged dimensions: ";
            for (auto d: merged_attention.dimensions)
                std::cout << d << " ";
            std::cout << "\n- Merged data size: " << merged_attention.data.size()
                    << std::endl;

            return {
                {std::move(merged_embeds), std::move(merged_attention)},
                std::move(image_features.dimensions[1] + inputs_embeds.dimensions[1])
            };
        } catch (const std::exception &e) {
            std::cout << "\nError during concatenation: " << e.what() << std::endl;
            throw;
        }
    }

    std::string Florence2Model::decode_single(
        const std::shared_ptr<Florence2Tokenizer> &tokenizer,
        const std::vector<int64_t> &token_ids) {
        // Convert token IDs to tokens
        std::vector<std::string> tokens;
        tokens.reserve(token_ids.size());

        for (const auto &id: token_ids) {
            tokens.push_back(tokenizer->id_to_token(id));
        }

        // Join tokens and decode
        auto decoded = tokenizer->untokenize(tokens);
        auto joined = std::accumulate(decoded.begin(), decoded.end(), std::string());

        return clean_up_tokenization(joined);
    }

    std::string Florence2Model::clean_up_tokenization(const std::string &text) {
        std::string result = text;

        // Clean up tokenization artifacts
        const std::vector<std::pair<std::string, std::string> > replacements = {
            {" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","},
            {" ' ", ""}, {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"},
            {" 've", "'ve"}, {" 're", "'re"}
        };

        for (const auto &[from, to]: replacements) {
            size_t pos = 0;
            while ((pos = result.find(from, pos)) != std::string::npos) {
                result.replace(pos, from.length(), to);
                pos += to.length();
            }
        }

        return result;
    }

    std::string Florence2Model::construct_prompts(TaskType task_type,
                                                  const std::string *text_input) {
        // First check prompts without inputs
        auto it = task_prompts_without_inputs_.find(task_type);
        if (it != task_prompts_without_inputs_.end()) {
            return it->second;
        }

        // Then check prompts that require text input
        auto it_with_input = task_prompts_with_inputs_.find(task_type);
        if (it_with_input != task_prompts_with_inputs_.end()) {
            if (!text_input) {
                throw std::invalid_argument("Text input required for this task type");
            }

            // Replace {} with the text input
            std::string prompt = it_with_input->second;
            size_t pos = prompt.find("{}");
            if (pos != std::string::npos) {
                prompt.replace(pos, 2, *text_input);
            }
            return prompt;
        }

        throw std::invalid_argument("Unknown task type");
    }

    std::vector<float> load_binary(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }

        // Get file size
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read into vector
        std::vector<float> buffer(size / sizeof(float));
        file.read(reinterpret_cast<char *>(buffer.data()), size);

        return buffer;
    }

    std::vector<std::string> Florence2Model::generate_loop_v4(
        const std::vector<float> &prefill_logits_data,
        const std::vector<int64_t> &prefill_logits_shape,
        const std::vector<std::vector<float> > &decoder_key_data,
        const std::vector<std::vector<int64_t> > &decoder_key_shapes,
        const std::vector<std::vector<float> > &decoder_value_data,
        const std::vector<std::vector<int64_t> > &decoder_value_shapes,
        const std::vector<std::vector<float> > &encoder_key_data,
        const std::vector<std::vector<int64_t> > &encoder_key_shapes,
        const std::vector<std::vector<float> > &encoder_value_data,
        const std::vector<std::vector<int64_t> > &encoder_value_shapes,
        const DenseTensor<float> &encoder_hidden_states,
        const DenseTensor<int64_t> &attention_mask,
        const Ort::RunOptions &run_options, const int64_t sequence_length) {
        Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Cache configuration values
        const int64_t batch_size = static_cast<int64_t>(attention_mask.dimensions[0]);
        const int64_t max_length = GenerationConfig::get_max_length();
        const int64_t no_repeat_ngram_size =
                GenerationConfig::get_no_repeat_ngram_size();
        const int64_t num_beams = GenerationConfig::get_num_beams();
        const int64_t top_k = GenerationConfig::get_top_k();

        // Pre-allocate vectors
        std::vector<std::vector<int64_t> > all_input_ids(batch_size);
        for (auto &input_ids: all_input_ids) {
            input_ids.reserve(max_length);
        }
        std::vector<double> scores(batch_size, 0.0);
        std::vector<bool> is_done(batch_size, false);
        int done_count = 0;

        // Initialize sampler and processors
        BeamSearchSampler sampler(session_tensor_ops_.get(), top_k, num_beams);
        std::vector<std::unique_ptr<LogitsProcessor> > logits_processors;
        logits_processors.push_back(
            std::make_unique<NoRepeatNGramLogitsProcessor>(no_repeat_ngram_size));

        const int64_t bos_token_id = tokenizer_->token_to_id(
            *tokenizer_->get_tokens()->beginning_of_sequence());
        const int64_t eos_token_id =
                tokenizer_->token_to_id(*tokenizer_->get_tokens()->end_of_sequence());

        logits_processors.push_back(
            std::make_unique<ForcedBOSTokenLogitsProcessor>(bos_token_id));
        logits_processors.push_back(std::make_unique<ForcedEOSTokenLogitsProcessor>(
            max_length, eos_token_id));

        std::vector<std::unique_ptr<StoppingCriteria> > stopping_criteria;
        stopping_criteria.push_back(std::make_unique<MaxLengthCriteria>(max_length));

        // Persistent tensor buffers
        std::vector<Ort::Value> decoder_kv;
        decoder_kv.reserve(12); // 6 layers * 2 tensors

        // Create persistent decoder key/value tensors
        for (int i = 0; i < 6; ++i) {
            decoder_kv.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(decoder_key_data[i].data()),
                decoder_key_data[i].size(), decoder_key_shapes[i].data(),
                decoder_key_shapes[i].size()));
            decoder_kv.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(decoder_value_data[i].data()),
                decoder_value_data[i].size(), decoder_value_shapes[i].data(),
                decoder_value_shapes[i].size()));
        }

        // Initialize logits
        const float *logits_data = prefill_logits_data.data();
        std::vector<int64_t> logits_dims = prefill_logits_shape;

        // Pre-allocate persistent tensors for input and embed names
        std::vector<int64_t> input_buffer{bos_token_id};
        auto input_shape = std::vector<int64_t>{1, 1};

        Ort::MemoryInfo input_memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
        void *input_buffer_memory =
                Ort::AllocatorWithDefaultOptions().Alloc(sizeof(int64_t));
        Ort::Value input_tensor = Ort::Value::CreateTensor(
            input_memory, static_cast<int64_t *>(input_buffer_memory),
            sizeof(int64_t) / sizeof(int64_t), input_shape.data(),
            input_shape.size());

        static const std::vector<const char *> embed_input_names = {"input_ids"};
        static const std::vector<const char *> embed_output_names = {"inputs_embeds"};

        while (done_count < batch_size) {
            for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                if (is_done[batch_index])
                    continue;

                std::vector<float> last_logits(
                    logits_data + (logits_dims[2] * (logits_dims[1] - 1)),
                    logits_data + (logits_dims[2] * logits_dims[1]));

                for (const auto &processor: logits_processors) {
                    processor->process(batch_index, all_input_ids[batch_index], last_logits,
                                       logits_dims);
                }

                auto sampled_tokens =
                        sampler.sample(batch_index, last_logits, logits_dims);
                for (const auto &[token, score]: sampled_tokens) {
                    scores[batch_index] += score;
                    all_input_ids[batch_index].push_back(token);
                    break;
                }

                auto criterion_done = stopping_criteria[0]->call(
                    {all_input_ids[batch_index]}, {scores[batch_index]});
                if (criterion_done[0] ||
                    all_input_ids[batch_index].back() == eos_token_id) {
                    is_done[batch_index] = true;
                    ++done_count;
                }
            }

            if (done_count >= batch_size)
                break;

            // Update input tensor buffer with new token
            auto *input_data = static_cast<int64_t *>(input_buffer_memory);
            *input_data = all_input_ids[0].back();

            // Get embeddings using persistent input tensor
            auto next_embeds = session_embed_tokens_->Run(
                run_options, embed_input_names.data(), &input_tensor, 1,
                embed_output_names.data(), 1);

            // Prepare decoder inputs
            std::vector<Ort::Value> decoder_inputs;
            decoder_inputs.reserve(28); // 6 layers * 4 tensors + 4 inputs

            bool use_cache = true;
            decoder_inputs.push_back(Ort::Value::CreateTensor<bool>(
                memory_info, &use_cache, 1, std::vector<int64_t>{1}.data(), 1));

            decoder_inputs.push_back(std::move(next_embeds[0]));

            decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(encoder_hidden_states.data.data()),
                encoder_hidden_states.data.size(),
                encoder_hidden_states.dimensions.data(),
                encoder_hidden_states.dimensions.size()));

            decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t *>(attention_mask.data.data()),
                attention_mask.data.size(), attention_mask.dimensions.data(),
                attention_mask.dimensions.size()));

            // Append key/value pairs
            for (int i = 0; i < 6; ++i) {
                decoder_inputs.push_back(std::move(decoder_kv[i * 2]));
                decoder_inputs.push_back(std::move(decoder_kv[i * 2 + 1]));
                decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, const_cast<float *>(encoder_key_data[i].data()),
                    encoder_key_data[i].size(), encoder_key_shapes[i].data(),
                    encoder_key_shapes[i].size()));
                decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, const_cast<float *>(encoder_value_data[i].data()),
                    encoder_value_data[i].size(), encoder_value_shapes[i].data(),
                    encoder_value_shapes[i].size()));
            }

            auto decoder_outputs = session_decoder_merged_->Run(
                run_options, get_decoder_input_names().data(), decoder_inputs.data(),
                decoder_inputs.size(), get_decoder_output_names().data(),
                get_decoder_output_names().size());

            // Update states for next iteration
            logits_data = decoder_outputs[0].GetTensorData<float>();
            logits_dims = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

            // Update KV cache
            for (int i = 0; i < 6; ++i) {
                decoder_kv[i * 2] = std::move(decoder_outputs[i * 4 + 1]);
                decoder_kv[i * 2 + 1] = std::move(decoder_outputs[i * 4 + 2]);
            }
        }

        // Free allocated memory
        Ort::AllocatorWithDefaultOptions().Free(input_buffer_memory);

        // Decode results
        std::vector<std::string> results;
        results.reserve(batch_size);
        for (const auto &input_ids: all_input_ids) {
            results.push_back(decode_single(tokenizer_, input_ids));
        }
        return results;
    }

    std::vector<std::string> Florence2Model::generate_loop_v3(
        const std::vector<float> &prefill_logits_data,
        const std::vector<int64_t> &prefill_logits_shape,
        const std::vector<std::vector<float> > &decoder_key_data,
        const std::vector<std::vector<int64_t> > &decoder_key_shapes,
        const std::vector<std::vector<float> > &decoder_value_data,
        const std::vector<std::vector<int64_t> > &decoder_value_shapes,
        const std::vector<std::vector<float> > &encoder_key_data,
        const std::vector<std::vector<int64_t> > &encoder_key_shapes,
        const std::vector<std::vector<float> > &encoder_value_data,
        const std::vector<std::vector<int64_t> > &encoder_value_shapes,
        const DenseTensor<float> &encoder_hidden_states,
        const DenseTensor<int64_t> &attention_mask,
        const Ort::RunOptions &run_options, const int64_t sequence_length) {
        // std::cout << "\nStarting generation loop v3..." << std::endl;

        Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t batch_size = static_cast<int64_t>(attention_mask.dimensions[0]);
        const int64_t max_length = GenerationConfig::get_max_length();
        const int64_t no_repeat_ngram_size =
                GenerationConfig::get_no_repeat_ngram_size();
        const int64_t num_beams = GenerationConfig::get_num_beams();
        const int64_t top_k = GenerationConfig::get_top_k();

        std::vector<std::vector<int64_t> > all_input_ids(batch_size);
        std::vector<double> scores(batch_size, 0.0);
        std::vector<bool> is_done(batch_size, false);

        // Initialize beam search sampler
        BeamSearchSampler sampler(session_tensor_ops_.get(), top_k, num_beams);

        // Initialize logits processors
        std::vector<std::unique_ptr<LogitsProcessor> > logits_processors;
        logits_processors.push_back(
            std::make_unique<NoRepeatNGramLogitsProcessor>(no_repeat_ngram_size));

        auto bos_token = tokenizer_->get_tokens()->beginning_of_sequence();
        if (!bos_token) {
            throw std::runtime_error("Beginning of sequence token not found");
        }
        logits_processors.push_back(std::make_unique<ForcedBOSTokenLogitsProcessor>(
            tokenizer_->token_to_id(*bos_token)));

        auto eos_token = tokenizer_->get_tokens()->end_of_sequence();
        if (!eos_token) {
            throw std::runtime_error("End of sequence token not found");
        }
        const int64_t eos_token_id = tokenizer_->token_to_id(*eos_token);
        logits_processors.push_back(std::make_unique<ForcedEOSTokenLogitsProcessor>(
            max_length, eos_token_id));

        // Initialize stopping criteria
        std::vector<std::unique_ptr<StoppingCriteria> > stopping_criteria;
        stopping_criteria.push_back(std::make_unique<MaxLengthCriteria>(max_length));

        // Store persistent copies
        std::vector<std::vector<float> > decoder_key_copies;
        std::vector<std::vector<float> > decoder_val_copies;
        std::vector<std::vector<float> > encoder_key_copies;
        std::vector<std::vector<float> > encoder_val_copies;

        // Make persistent copies
        for (int i = 0; i < 6; i++) {
            decoder_key_copies.push_back(decoder_key_data[i]);
            decoder_val_copies.push_back(decoder_value_data[i]);
            encoder_key_copies.push_back(encoder_key_data[i]);
            encoder_val_copies.push_back(encoder_value_data[i]);
        }

        // Get initial logits data
        const float *logits_data_ac = prefill_logits_data.data();
        float *logits_data = (float *) logits_data_ac;
        auto logits_dims = prefill_logits_shape;

        // Initialize decoder KVs
        std::vector<Ort::Value> decoder_kv;
        for (int i = 0; i < 6; i++) {
            decoder_kv.push_back(Ort::Value::CreateTensor<float>(
                memory_info, decoder_key_copies[i].data(), decoder_key_copies[i].size(),
                decoder_key_shapes[i].data(), decoder_key_shapes[i].size()));

            decoder_kv.push_back(Ort::Value::CreateTensor<float>(
                memory_info, decoder_val_copies[i].data(), decoder_val_copies[i].size(),
                decoder_value_shapes[i].data(), decoder_value_shapes[i].size()));
        }

        while (
            !std::all_of(is_done.begin(), is_done.end(), [](bool b) { return b; })) {
            // Process logits for each batch
            for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                if (is_done[batch_index]) {
                    continue;
                }

                std::vector<float> last_logits(
                    logits_data + (logits_dims[2] * (logits_dims[1] - 1)),
                    logits_data + (logits_dims[2] * logits_dims[1]));

                // Apply logits processors
                for (const auto &processor: logits_processors) {
                    processor->process(batch_index, all_input_ids[batch_index], last_logits,
                                       logits_dims);
                }

                // Use beam search sampler instead of greedy selection
                auto sampled_tokens =
                        sampler.sample(batch_index, last_logits, logits_dims);

                // Take best token from beam search
                for (const auto &[token, score]: sampled_tokens) {
                    // std::cout << "Generated token: " << token << ", score: " << score <<
                    // std::endl;
                    scores[batch_index] += score;
                    all_input_ids[batch_index].push_back(token);

                    // Break after taking the best token
                    break;
                }

                // Check stopping criteria
                auto criterion_done = stopping_criteria[0]->call(
                    {all_input_ids[batch_index]}, {scores[batch_index]});
                is_done[batch_index] = is_done[batch_index] || criterion_done[0];

                if (all_input_ids[batch_index].back() == eos_token_id) {
                    is_done[batch_index] = true;
                }
            }

            if (std::all_of(is_done.begin(), is_done.end(), [](bool b) { return b; })) {
                break;
            }

            // Get embeddings for next token
            std::vector<int64_t> next_input_dims = {1, 1};
            std::vector<int64_t> next_input_data = {
                all_input_ids[0].back()
            }; // Using last generated token

            std::vector<const char *> embed_input_names = {"input_ids"};
            std::vector<const char *> embed_output_names = {"inputs_embeds"};

            std::vector<Ort::Value> embed_inputs;
            embed_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, next_input_data.data(), next_input_data.size(),
                next_input_dims.data(), next_input_dims.size()));

            auto next_embeds = session_embed_tokens_->Run(
                run_options, embed_input_names.data(), embed_inputs.data(),
                embed_inputs.size(), embed_output_names.data(),
                embed_output_names.size());

            // Setup decoder inputs
            std::vector<Ort::Value> decoder_inputs;
            bool use_cache = true;
            decoder_inputs.push_back(Ort::Value::CreateTensor<bool>(
                memory_info, &use_cache, 1, std::vector<int64_t>{1}.data(), 1));

            decoder_inputs.push_back(std::move(next_embeds[0]));

            std::vector<float> encoder_data = encoder_hidden_states.data;
            decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, encoder_data.data(), encoder_data.size(),
                encoder_hidden_states.dimensions.data(),
                encoder_hidden_states.dimensions.size()));

            std::vector<int64_t> attention_data = attention_mask.data;
            decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_data.data(), attention_data.size(),
                attention_mask.dimensions.data(), attention_mask.dimensions.size()));

            // Add KVs layer by layer
            for (int i = 0; i < 6; i++) {
                decoder_inputs.push_back(std::move(decoder_kv[i * 2]));
                decoder_inputs.push_back(std::move(decoder_kv[i * 2 + 1]));

                decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, encoder_key_copies[i].data(),
                    encoder_key_copies[i].size(), encoder_key_shapes[i].data(),
                    encoder_key_shapes[i].size()));

                decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info, encoder_val_copies[i].data(),
                    encoder_val_copies[i].size(), encoder_value_shapes[i].data(),
                    encoder_value_shapes[i].size()));
            }

            // Run decoder
            auto decoder_outputs = session_decoder_merged_->Run(
                run_options, get_decoder_input_names().data(), decoder_inputs.data(),
                decoder_inputs.size(), get_decoder_output_names().data(),
                get_decoder_output_names().size());

            // Update for next iteration
            logits_data = (float *) decoder_outputs[0].GetTensorData<float>();
            logits_dims = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

            decoder_kv.clear();
            for (int i = 0; i < 6; i++) {
                decoder_kv.push_back(std::move(decoder_outputs[i * 4 + 1]));
                decoder_kv.push_back(std::move(decoder_outputs[i * 4 + 2]));
            }
        }

        std::vector<std::string> results;
        for (const auto &input_ids: all_input_ids) {
            results.push_back(decode_single(tokenizer_, input_ids));
        }
        return results;
    }

    std::vector<FlorenceResults>
    Florence2Model::run(TaskType task,
                        const std::vector<std::vector<uint8_t> > &img_data,
                        const std::string &text_input) {
        try {
            if (img_data.empty()) {
                throw std::runtime_error("No images provided");
            }

            // Create run options and memory info
            Ort::RunOptions run_options;
            Ort::MemoryInfo memory_info =
                    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            encoder_past_key_values_.clear();

            // Step 1: Process text inputs
            auto prompts = std::vector<std::string>(
                img_data.size(),
                construct_prompts(task, text_input.empty() ? nullptr : &text_input));
            std::cout << "Image Data Size: " << img_data.size() << std::endl;
            auto [input_ids, attention_mask] = get_text_inputs(prompts);

            for (auto x: prompts)
                std::cout << x << std::endl;

            std::cout << "Encoded text:" << std::endl;
            std::cout << "Input IDs size: " << input_ids.size() << std::endl;
            std::cout << "Tokens: ";
            for (int64_t id: input_ids.data) {
                std::cout << id << " ";
            }
            std::cout << std::endl;

            // Step 2: Get text embeddings
            std::vector<const char *> embed_input_names = {"input_ids"};
            std::vector<const char *> embed_output_names = {"inputs_embeds"};
            std::vector<Ort::Value> text_embed_inputs;

            text_embed_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data.data(), input_ids.data.size(),
                input_ids.dimensions.data(), input_ids.dimensions.size()));

            auto text_embeds = session_embed_tokens_->Run(
                run_options, embed_input_names.data(), text_embed_inputs.data(),
                text_embed_inputs.size(), embed_output_names.data(),
                embed_output_names.size());

            // Step 3: Process image inputs
            auto [pixel_values, img_sizes] = image_processor_->preprocess(img_data);

            const auto &config = image_processor_->get_config();
            std::vector<int64_t> pixel_dims = {
                static_cast<int64_t>(img_data.size()), 3,
                static_cast<int64_t>(config.crop_height),
                static_cast<int64_t>(config.crop_width)
            };

            // Validate image tensor size
            size_t expected_size =
                    img_data.size() * 3 * config.crop_height * config.crop_width;
            if (pixel_values.size() != expected_size) {
                throw std::runtime_error("Preprocessed image tensor size mismatch");
            }

            // Step 4: Run vision encoder
            std::vector<const char *> vision_input_names = {"pixel_values"};
            std::vector<const char *> vision_output_names = {"image_features"};
            std::vector<Ort::Value> vision_inputs;

            vision_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, pixel_values.data(), pixel_values.size(),
                pixel_dims.data(), pixel_dims.size()));

            auto vision_outputs = session_vision_encoder_->Run(
                run_options, vision_input_names.data(), vision_inputs.data(),
                vision_inputs.size(), vision_output_names.data(),
                vision_output_names.size());

            // Step 5: Convert Ort::Values to DenseTensors for merging
            DenseTensor<float> text_embeds_tensor;
            {
                auto tensor_info = text_embeds[0].GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                text_embeds_tensor.dimensions.assign(shape.begin(), shape.end());
                const float *data = text_embeds[0].GetTensorData<float>();
                size_t size = tensor_info.GetElementCount();
                text_embeds_tensor.data.assign(data, data + size);
            }

            DenseTensor<float> vision_features_tensor;
            {
                auto tensor_info = vision_outputs[0].GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                vision_features_tensor.dimensions.assign(shape.begin(), shape.end());
                const float *data = vision_outputs[0].GetTensorData<float>();
                size_t size = tensor_info.GetElementCount();
                vision_features_tensor.data.assign(data, data + size);
            }

            // Step 6: Merge text and vision features
            auto merger = merge_input_ids_with_image_features(
                text_embeds_tensor, vision_features_tensor, attention_mask);

            auto merged_embeds = merger.first.first;
            auto merged_attention = merger.first.second;
            auto merge_size = merger.second;

            // Step 7: Run encoder with merged tensors
            std::vector<const char *> encoder_input_names = {
                "attention_mask",
                "inputs_embeds"
            };
            std::vector<const char *> encoder_output_names = {"last_hidden_state"};
            std::vector<Ort::Value> encoder_inputs;

            encoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, merged_attention.data.data(), merged_attention.data.size(),
                merged_attention.dimensions.data(),
                merged_attention.dimensions.size()));

            encoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, merged_embeds.data.data(), merged_embeds.data.size(),
                merged_embeds.dimensions.data(), merged_embeds.dimensions.size()));

            auto encoder_outputs = session_encoder_->Run(
                run_options, encoder_input_names.data(), encoder_inputs.data(),
                encoder_inputs.size(), encoder_output_names.data(),
                encoder_output_names.size());

            // Step 8: Convert encoder output for generation
            DenseTensor<float> encoder_tensor;
            {
                auto tensor_info = encoder_outputs[0].GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                encoder_tensor.dimensions.assign(shape.begin(), shape.end());
                const float *data = encoder_outputs[0].GetTensorData<float>();
                size_t size = tensor_info.GetElementCount();
                encoder_tensor.data.assign(data, data + size);
            }

            // Step 9: Prefill stage
            auto eos_token = tokenizer_->get_tokens()->end_of_sequence();
            if (!eos_token) {
                throw std::runtime_error("End of sequence token not found");
            }
            const int64_t decoder_start_token_id = tokenizer_->token_to_id(*eos_token);

            // Create initial decoder input
            DenseTensor<int64_t> initial_input_ids;
            initial_input_ids.dimensions = {static_cast<int64_t>(img_data.size()), 1};
            initial_input_ids.data =
                    std::vector<int64_t>(img_data.size(), decoder_start_token_id);

            // Get embeddings for initial input
            std::vector<Ort::Value> initial_embed_inputs;
            initial_embed_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, initial_input_ids.data.data(),
                initial_input_ids.data.size(), initial_input_ids.dimensions.data(),
                initial_input_ids.dimensions.size()));

            auto initial_embeds = session_embed_tokens_->Run(
                run_options, embed_input_names.data(), initial_embed_inputs.data(),
                initial_embed_inputs.size(), embed_output_names.data(),
                embed_output_names.size());

            auto embed_info = initial_embeds[0].GetTensorTypeAndShapeInfo();
            auto embed_shape = embed_info.GetShape();

            const float *embed_data = initial_embeds[0].GetTensorData<float>();
            size_t total_elements = embed_info.GetElementCount();

            std::cout << "First 10 values: ";
            for (size_t i = 0; i < 10 && i < total_elements; i++) {
                std::cout << embed_data[i] << " ";
            }
            std::cout << std::endl;

            // Run prefill stage
            std::vector<const char *> prefill_input_names = {
                "inputs_embeds", "encoder_hidden_states", "encoder_attention_mask"
            };
            std::vector<const char *> prefill_output_names = {"logits"};

            // Add output names for present key values
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
                "present.5.encoder.key", "present.5.encoder.value"
            };
            for (const auto &name: present_names) {
                prefill_output_names.push_back(name);
            }

            std::vector<Ort::Value> prefill_inputs;
            prefill_inputs.push_back(std::move(initial_embeds[0]));
            prefill_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, encoder_tensor.data.data(), encoder_tensor.data.size(),
                encoder_tensor.dimensions.data(), encoder_tensor.dimensions.size()));
            prefill_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, merged_attention.data.data(), merged_attention.data.size(),
                merged_attention.dimensions.data(),
                merged_attention.dimensions.size()));

            auto prefill_outputs = session_decoder_->Run(
                run_options, prefill_input_names.data(), prefill_inputs.data(),
                prefill_inputs.size(), prefill_output_names.data(),
                prefill_output_names.size());

            // Store encoder KVs for generation loop
            std::vector<std::vector<float> > encoder_key_data;
            std::vector<std::vector<int64_t> > encoder_key_shapes;
            std::vector<std::vector<float> > encoder_value_data;
            std::vector<std::vector<int64_t> > encoder_value_shapes;
            for (int i = 0; i < 6; ++i) {
                auto &encoder_key = prefill_outputs[i * 4 + 3];
                auto &encoder_val = prefill_outputs[i * 4 + 4];

                auto key_info = encoder_key.GetTensorTypeAndShapeInfo();
                auto val_info = encoder_val.GetTensorTypeAndShapeInfo();

                auto key_shape = key_info.GetShape();
                auto val_shape = val_info.GetShape();

                std::vector<float> key_data(key_info.GetElementCount());
                std::vector<float> val_data(val_info.GetElementCount());

                memcpy(key_data.data(), encoder_key.GetTensorData<float>(),
                       key_info.GetElementCount() * sizeof(float));
                memcpy(val_data.data(), encoder_val.GetTensorData<float>(),
                       val_info.GetElementCount() * sizeof(float));

                encoder_key_data.push_back(std::move(key_data));
                encoder_key_shapes.push_back(key_shape);
                encoder_value_data.push_back(std::move(val_data));
                encoder_value_shapes.push_back(val_shape);
            }

            // Extract data from prefill outputs
            std::vector<float> prefill_logits_data;
            std::vector<int64_t> prefill_logits_shape;
            {
                auto tensor_info = prefill_outputs[0].GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                prefill_logits_shape = shape;
                size_t count = tensor_info.GetElementCount();
                const float *data = prefill_outputs[0].GetTensorData<float>();
                prefill_logits_data.assign(data, data + count);
            }

            // Extract decoder KV data
            std::vector<std::vector<float> > decoder_key_data;
            std::vector<std::vector<int64_t> > decoder_key_shapes;
            std::vector<std::vector<float> > decoder_value_data;
            std::vector<std::vector<int64_t> > decoder_value_shapes;

            for (int i = 0; i < 6; i++) {
                auto &decoder_key = prefill_outputs[i * 4 + 1];
                auto &decoder_value = prefill_outputs[i * 4 + 2];

                auto key_info = decoder_key.GetTensorTypeAndShapeInfo();
                auto value_info = decoder_value.GetTensorTypeAndShapeInfo();

                auto key_shape = key_info.GetShape();
                auto value_shape = value_info.GetShape();

                std::vector<float> key_data(key_info.GetElementCount());
                std::vector<float> value_data(value_info.GetElementCount());

                memcpy(key_data.data(), decoder_key.GetTensorData<float>(),
                       key_info.GetElementCount() * sizeof(float));
                memcpy(value_data.data(), decoder_value.GetTensorData<float>(),
                       value_info.GetElementCount() * sizeof(float));

                decoder_key_data.push_back(std::move(key_data));
                decoder_key_shapes.push_back(key_shape);
                decoder_value_data.push_back(std::move(value_data));
                decoder_value_shapes.push_back(value_shape);
            }

            // Call generate_loop_v2 with extracted data
            auto raw_results = generate_loop_v4(
                prefill_logits_data, prefill_logits_shape, decoder_key_data,
                decoder_key_shapes, decoder_value_data, decoder_value_shapes,
                encoder_key_data, encoder_key_shapes, encoder_value_data,
                encoder_value_shapes, encoder_tensor, merged_attention, run_options,
                merge_size);

            // Step 11: Post-process results
            std::vector<FlorenceResults> final_results;
            final_results.reserve(raw_results.size());

            for (size_t i = 0; i < raw_results.size(); ++i) {
                final_results.push_back(post_processor_->PostProcessGeneration(
                    raw_results[i], task, img_sizes[i]));
            }

            return final_results;
        } catch (const Ort::Exception &e) {
            throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Error in run: ") + e.what());
        }
    }
} // namespace florence2
