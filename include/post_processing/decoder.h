#pragma once

#include <string>
#include <vector>
#include <memory>
#include "tokenizer/florence2_tokenizer.h"

namespace florence2 {

class Decoder {
public:
    virtual ~Decoder() = default;
};

class ByteLevelDecoder : public Decoder {
public:
    explicit ByteLevelDecoder(const std::unordered_map<std::string, AddedToken>& added_tokens);
    
    std::vector<std::string> decode_chain(
        Florence2Tokenizer* tokenizer,
        const std::vector<std::string>& tokens);

private:
    std::string convert_tokens_to_string(
        Florence2Tokenizer* tokenizer,
        const std::vector<std::string>& tokens);

    std::unordered_map<std::string, AddedToken> added_tokens_;
};

} // namespace florence2