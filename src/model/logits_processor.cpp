#include "model/logits_processor.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace florence2 {

std::vector<float> LogitsProcessor::get_batch_slice(
    int batch_id, 
    const std::vector<float>& tensor,
    const std::vector<int64_t>& dimensions) 
{
    if (dimensions.size() == 2) {
        // For 2D tensor [batch_size, vocab_size]
        size_t start = batch_id * dimensions[1];
        size_t size = dimensions[1];
        return std::vector<float>(tensor.begin() + start, tensor.begin() + start + size);
    }
    else if (dimensions.size() == 3 && dimensions[1] == 1) {
        // For 3D tensor with sequence_length=1 [batch_size, 1, vocab_size]
        size_t start = batch_id * dimensions[2];
        size_t size = dimensions[2];
        return std::vector<float>(tensor.begin() + start, tensor.begin() + start + size);
    }
    else if (dimensions.size() == 3) {
        // For general 3D tensor [batch_size, sequence_length, vocab_size]
        // Get the last sequence position
        size_t vocab_size = dimensions[2];
        size_t seq_length = dimensions[1];
        size_t start = batch_id * seq_length * vocab_size + (seq_length - 1) * vocab_size;
        return std::vector<float>(tensor.begin() + start, tensor.begin() + start + vocab_size);
    }
    else {
        throw std::runtime_error("Unsupported tensor dimensions in get_batch_slice");
    }
}

// NoRepeatNGramLogitsProcessor implementation
NoRepeatNGramLogitsProcessor::NoRepeatNGramLogitsProcessor(int no_repeat_ngram_size)
    : no_repeat_ngram_size_(no_repeat_ngram_size) {
}

NoRepeatNGramLogitsProcessor::NGramMap 
NoRepeatNGramLogitsProcessor::get_ngrams(int batch_id, const std::vector<int64_t>& prev_input_ids) {
    int64_t cur_len = prev_input_ids.size();
    std::vector<std::vector<int64_t>> ngrams;

    // Generate all possible ngrams
    for (int j = 0; j < cur_len + 1 - no_repeat_ngram_size_; ++j) {
        std::vector<int64_t> ngram(no_repeat_ngram_size_);
        for (int k = 0; k < no_repeat_ngram_size_; ++k) {
            ngram[k] = prev_input_ids[j + k];
        }
        ngrams.push_back(ngram);
    }

    // Create ngram map
    NGramMap generated_ngram;
    for (const auto& ngram : ngrams) {
        std::vector<int64_t> prev_ngram(ngram.begin(), ngram.end() - 1);
        generated_ngram[prev_ngram].push_back(ngram.back());
    }

    return generated_ngram;
}

std::vector<int64_t> NoRepeatNGramLogitsProcessor::get_generated_ngrams(
    int batch_id,
    const NGramMap& banned_ngrams,
    const std::vector<int64_t>& prev_input_ids) 
{
    // Get the last n-1 tokens
    std::vector<int64_t> ngram_idx(
        prev_input_ids.begin() + (prev_input_ids.size() + 1 - no_repeat_ngram_size_),
        prev_input_ids.end()
    );

    auto it = banned_ngrams.find(ngram_idx);
    return it != banned_ngrams.end() ? it->second : std::vector<int64_t>();
}

std::vector<int64_t> NoRepeatNGramLogitsProcessor::calc_banned_ngram_tokens(
    int batch_id,
    const std::vector<int64_t>& prev_input_ids) 
{
    if (prev_input_ids.size() + 1 < no_repeat_ngram_size_) {
        return std::vector<int64_t>();
    }

    auto generated_ngrams = get_ngrams(batch_id, prev_input_ids);
    return get_generated_ngrams(batch_id, generated_ngrams, prev_input_ids);
}

void NoRepeatNGramLogitsProcessor::process(
    int batch_id,
    const std::vector<int64_t>& input_ids,
    std::vector<float>& logits,
    const std::vector<int64_t>& dimensions) 
{
    auto banned_tokens = calc_banned_ngram_tokens(batch_id, input_ids);
    
    size_t vocab_size = dimensions.back();
    size_t seq_length = dimensions[1];
    size_t last_token_offset = batch_id * seq_length * vocab_size + (seq_length - 1) * vocab_size;
    
    for (int64_t token : banned_tokens) {
        logits[last_token_offset + token] = -std::numeric_limits<float>::infinity();
    }
}

// ForcedBOSTokenLogitsProcessor implementation
ForcedBOSTokenLogitsProcessor::ForcedBOSTokenLogitsProcessor(int bos_token_id)
    : bos_token_id_(bos_token_id) {
}

void ForcedBOSTokenLogitsProcessor::process(
    int batch_id,
    const std::vector<int64_t>& input_ids,
    std::vector<float>& logits,
    const std::vector<int64_t>& dimensions) 
{
    if (input_ids.size() == 1) {
        size_t vocab_size = dimensions.back();
        size_t seq_length = dimensions[1];
        size_t last_token_offset = batch_id * seq_length * vocab_size + (seq_length - 1) * vocab_size;
        
        // Fill with negative infinity
        std::fill(logits.begin() + last_token_offset,
                 logits.begin() + last_token_offset + vocab_size,
                 -std::numeric_limits<float>::infinity());
                 
        // Set BOS token probability
        logits[last_token_offset + bos_token_id_] = 0;
    }
}

// ForcedEOSTokenLogitsProcessor implementation
ForcedEOSTokenLogitsProcessor::ForcedEOSTokenLogitsProcessor(int max_length, int eos_token_id)
    : max_length_(max_length)
    , eos_token_id_(eos_token_id) {
}

void ForcedEOSTokenLogitsProcessor::process(
    int batch_id,
    const std::vector<int64_t>& input_ids,
    std::vector<float>& logits,
    const std::vector<int64_t>& dimensions) 
{
    if (input_ids.size() == max_length_ - 1) {
        size_t vocab_size = dimensions.back();
        size_t seq_length = dimensions[1];
        size_t last_token_offset = batch_id * seq_length * vocab_size + (seq_length - 1) * vocab_size;
        
        // Fill with negative infinity
        std::fill(logits.begin() + last_token_offset,
                 logits.begin() + last_token_offset + vocab_size,
                 -std::numeric_limits<float>::infinity());
                 
        // Set EOS token probability
        logits[last_token_offset + eos_token_id_] = 0;
    }
}

} // namespace florence2