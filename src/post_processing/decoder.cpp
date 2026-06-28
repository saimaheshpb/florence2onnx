#include "post_processing/decoder.h"
#include <algorithm>

namespace florence2 {

ByteLevelDecoder::ByteLevelDecoder(
    const std::unordered_map<std::string, AddedToken>& added_tokens)
    : added_tokens_(added_tokens) {
}

std::string ByteLevelDecoder::convert_tokens_to_string(
    Florence2Tokenizer* tokenizer,
    const std::vector<std::string>& tokens)
{
    std::string text;
    for (const auto& token : tokens) {
        text += token;
    }

    // Convert using unicode_to_bytes mapping
    std::vector<uint8_t> byte_array;
    byte_array.reserve(text.length());

    for (char c : text) {
        auto it = tokenizer->unicode_to_bytes.find(static_cast<uint16_t>(c));
        if (it != tokenizer->unicode_to_bytes.end()) {
            byte_array.push_back(it->second);
        }
    }

    // Convert bytes to UTF-8 string
    return std::string(byte_array.begin(), byte_array.end());
}

std::vector<std::string> ByteLevelDecoder::decode_chain(
    Florence2Tokenizer* tokenizer,
    const std::vector<std::string>& tokens)
{
    std::vector<std::string> sub_texts;
    std::vector<std::string> current_sub_text;

    for (const auto& token : tokens) {
        bool is_special = false;
        for (const auto& [_, added_token] : added_tokens_) {
            if (added_token.content == token) {
                is_special = true;
                break;
            }
        }

        if (is_special) {
            if (!current_sub_text.empty()) {
                sub_texts.push_back(convert_tokens_to_string(tokenizer, current_sub_text));
                current_sub_text.clear();
            }
            sub_texts.push_back(token);
        } else {
            current_sub_text.push_back(token);
        }
    }

    if (!current_sub_text.empty()) {
        sub_texts.push_back(convert_tokens_to_string(tokenizer, current_sub_text));
    }

    return sub_texts;
}

} // namespace florence2