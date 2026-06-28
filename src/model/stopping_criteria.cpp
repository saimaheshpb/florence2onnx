#include "model/stopping_criteria.h"
#include <algorithm>

namespace florence2 {

// MaxLengthCriteria implementation
MaxLengthCriteria::MaxLengthCriteria(int max_length) 
    : max_length_(max_length) {
}

std::vector<bool> MaxLengthCriteria::call(
    const std::vector<std::vector<int64_t>>& input_ids, 
    const std::vector<double>& scores) 
{
    std::vector<bool> results;
    results.reserve(input_ids.size());
    
    // Check each sequence length against max_length
    for (const auto& ids : input_ids) {
        results.push_back(ids.size() >= max_length_);
    }
    
    return results;
}

// EosTokenCriteria implementation
EosTokenCriteria::EosTokenCriteria(int64_t eos_token_id) 
    : eos_token_id_(eos_token_id) {
}

std::vector<bool> EosTokenCriteria::call(
    const std::vector<std::vector<int64_t>>& input_ids, 
    const std::vector<double>& scores) 
{
    std::vector<bool> results;
    results.reserve(input_ids.size());
    
    // Check if last token of each sequence is EOS token
    for (const auto& ids : input_ids) {
        results.push_back(!ids.empty() && ids.back() == eos_token_id_);
    }
    
    return results;
}

} // namespace florence2