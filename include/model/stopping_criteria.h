#pragma once

#include <vector>

namespace florence2 {

class StoppingCriteria {
public:
    virtual ~StoppingCriteria() = default;
    virtual std::vector<bool> call(
        const std::vector<std::vector<int64_t>>& input_ids, 
        const std::vector<double>& scores) = 0;
};

class MaxLengthCriteria : public StoppingCriteria {
public:
    explicit MaxLengthCriteria(int max_length);
    
    std::vector<bool> call(
        const std::vector<std::vector<int64_t>>& input_ids, 
        const std::vector<double>& scores) override;

private:
    int max_length_;
};

class EosTokenCriteria : public StoppingCriteria {
public:
    explicit EosTokenCriteria(int64_t eos_token_id);
    
    std::vector<bool> call(
        const std::vector<std::vector<int64_t>>& input_ids, 
        const std::vector<double>& scores) override;

private:
    int64_t eos_token_id_;
};

} // namespace florence2