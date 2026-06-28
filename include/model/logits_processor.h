#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <limits>

namespace florence2 {

class LogitsProcessor {
public:
    virtual ~LogitsProcessor() = default;
    virtual void process(int batch_id, const std::vector<int64_t>& input_ids, 
                        std::vector<float>& logits, const std::vector<int64_t>& dimensions) = 0;
    // Helper function to get batch slice
    static std::vector<float> get_batch_slice(int batch_id, 
                                            const std::vector<float>& tensor,
                                            const std::vector<int64_t>& dimensions);
};

class NoRepeatNGramLogitsProcessor : public LogitsProcessor {
public:
    explicit NoRepeatNGramLogitsProcessor(int no_repeat_ngram_size);
    
    void process(int batch_id, const std::vector<int64_t>& input_ids, 
                std::vector<float>& logits, 
                const std::vector<int64_t>& dimensions) override;

private:
    // Custom hash for vector<int64_t>
    struct VectorHash {
        size_t operator()(const std::vector<int64_t>& v) const {
            std::hash<int64_t> hasher;
            size_t seed = 0;
            for (int64_t i : v) {
                seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    using NGramMap = std::unordered_map<std::vector<int64_t>, 
                                      std::vector<int64_t>, 
                                      VectorHash>;

    NGramMap get_ngrams(int batch_id, const std::vector<int64_t>& prev_input_ids);
    std::vector<int64_t> get_generated_ngrams(int batch_id, 
                                            const NGramMap& banned_ngrams,
                                            const std::vector<int64_t>& prev_input_ids);
    std::vector<int64_t> calc_banned_ngram_tokens(int batch_id,
                                                const std::vector<int64_t>& prev_input_ids);

    int no_repeat_ngram_size_;
};

class ForcedBOSTokenLogitsProcessor : public LogitsProcessor {
public:
    explicit ForcedBOSTokenLogitsProcessor(int bos_token_id);
    
    void process(int batch_id, const std::vector<int64_t>& input_ids,
                std::vector<float>& logits, 
                const std::vector<int64_t>& dimensions) override;

private:
    int bos_token_id_;
};

class ForcedEOSTokenLogitsProcessor : public LogitsProcessor {
public:
    ForcedEOSTokenLogitsProcessor(int max_length, int eos_token_id);
    
    void process(int batch_id, const std::vector<int64_t>& input_ids,
                std::vector<float>& logits, 
                const std::vector<int64_t>& dimensions) override;

private:
    int max_length_;
    int eos_token_id_;
};

} // namespace florence2