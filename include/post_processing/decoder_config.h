#pragma once

namespace florence2 {

class GenerationConfig {
public:
    static inline int get_no_repeat_ngram_size() { return 2; }
    static inline int get_num_beams() { return 1; }
    static inline int get_max_length() { return 100; }
    static inline int get_top_k() { return 1; }
};

class NormalizedConfig {
public:
    NormalizedConfig() 
        : num_decoder_layers_(6)
        , num_decoder_heads_(12)
        , decoder_hidden_size_(768)
        , num_encoder_layers_(6)
        , num_encoder_heads_(12)
        , encoder_hidden_size_(768) 
    {}

    int get_num_decoder_layers() const { return num_decoder_layers_; }
    int get_num_decoder_heads() const { return num_decoder_heads_; }
    int get_decoder_hidden_size() const { return decoder_hidden_size_; }
    int get_num_encoder_layers() const { return num_encoder_layers_; }
    int get_num_encoder_heads() const { return num_encoder_heads_; }
    int get_encoder_hidden_size() const { return encoder_hidden_size_; }

private:
    int num_decoder_layers_;
    int num_decoder_heads_;
    int decoder_hidden_size_;
    int num_encoder_layers_;
    int num_encoder_heads_;
    int encoder_hidden_size_;
};

} // namespace florence2