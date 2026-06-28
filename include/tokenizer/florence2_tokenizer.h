#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <memory>
#include <regex>
#include <nlohmann/json.hpp>
#include "helper/tokens.h"
#include "helper/resource_loader.h"

namespace florence2
{

    struct AddedToken
    {
        std::string content;
        bool lstrip = false;
        bool normalized = false;
        bool rstrip = false;
        bool single_word = false;
        bool special = false;
    };

    struct TokenizerConfig
    {
        bool add_prefix_space = false;
        std::unordered_map<std::string, AddedToken> added_tokens_decoder;
        std::vector<std::string> additional_special_tokens;
        std::string bos_token;
        bool clean_up_tokenization_spaces = true;
        std::string cls_token;
        std::string eos_token;
        std::string errors;
        std::string mask_token;
        int64_t model_max_length = 0;
        std::string pad_token;
        std::string processor_class;
        std::string sep_token;
        std::string tokenizer_class;
        bool trim_offsets = false;
        std::string unk_token;
    };

    class Florence2TokenizerTest;

    class Florence2Tokenizer
    {
        friend class Florence2TokenizerTest;
        friend class ByteLevelDecoder;

    protected:
        static constexpr size_t VOCAB_SIZE = 51289;
        static constexpr size_t MAX_TOKENS = 512;

        std::array<std::string, VOCAB_SIZE> vocabulary_;
        std::unordered_map<std::string, int> vocabulary_dict_;
        std::shared_ptr<Tokens> tokens_;
        std::regex regex_{R"('s|'t|'re|'ve|'m|'ll|'d|\s*[a-zA-Z]+|\s*[0-9]+|\s*[^\s\w]+|\s+)"};
        // Remove 'unicode_to_bytes' from front
        static const std::unordered_map<uint16_t, uint8_t> unicode_to_bytes;  // just this
        std::unordered_map<uint8_t, uint16_t> byte_to_unicode;

    public:
        // Constructor and static factory method
        Florence2Tokenizer(
            const std::unordered_map<std::string, int> &vocabulary,
            const std::unordered_map<std::string, AddedToken> &added_tokens,
            std::shared_ptr<Tokens> tokens);

        static std::shared_ptr<Florence2Tokenizer> init();

        // Public member variables (matching C#)
        std::unordered_set<std::string> special_tokens;
        std::unordered_map<std::string, AddedToken> added_tokens;

        // Core token operations
        std::string id_to_token(int id) const;
        int token_to_id(const std::string &token) const;

        std::vector<std::string> convert_ids_to_tokens(
            const std::vector<int> &ids,
            bool skip_special_tokens = false);

        std::vector<std::string> untokenize(std::vector<std::string> tokens);

        void apply_bpe(std::vector<std::string> &tokens);

        // Tokenization methods
        std::vector<std::vector<std::pair<std::string, int>>>
        tokenize(const std::vector<std::string> &texts);

        std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
        encode(const std::vector<std::string> &texts);

        // Getter for testing
        std::shared_ptr<Tokens> get_tokens() const { return tokens_; }

        static std::vector<std::string> split_and_keep(
            const std::string &input_string,
            const std::vector<char> &delimiters);
            
        std::vector<std::string> tokenize_sentence(const std::string &text);

    private:
        static TokenizerConfig parse_tokenizer_config(const nlohmann::json &j);
    };

    // JSON conversion functions
    void from_json(const nlohmann::json &j, AddedToken &token);
    void from_json(const nlohmann::json &j, TokenizerConfig &config);

} // namespace florence2