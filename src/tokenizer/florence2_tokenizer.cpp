#include "tokenizer/florence2_tokenizer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stdexcept>
#include "helper/resource_loader.h"
#include <iostream>

namespace florence2 {
    // Using hex values for clarity
    const std::unordered_map<uint16_t, uint8_t> Florence2Tokenizer::unicode_to_bytes = {
        // ASCII numbers
        {0x30, 48},
        {0x31, 49},
        {0x32, 50},
        {0x33, 51},
        {0x34, 52}, // 0-4
        {0x35, 53},
        {0x36, 54},
        {0x37, 55},
        {0x38, 56},
        {0x39, 57}, // 5-9

        // Special Unicode characters (using their actual code points)
        {0x0100, 0},
        {0x0101, 1},
        {0x0102, 2},
        {0x0103, 3},
        {0x0104, 4},
        {0x0105, 5},
        {0x0106, 6},
        {0x0107, 7},
        {0x0108, 8},
        {0x0109, 9},
        {0x010A, 10},
        {0x010B, 11},
        {0x010C, 12},
        {0x010D, 13},
        {0x010E, 14},
        {0x010F, 15},
        {0x0110, 16},
        {0x0111, 17},
        {0x0112, 18},
        {0x0113, 19},
        {0x0114, 20},
        {0x0115, 21},
        {0x0116, 22},
        {0x0117, 23},
        {0x0118, 24},
        {0x0119, 25},
        {0x011A, 26},
        {0x011B, 27},
        {0x011C, 28},
        {0x011D, 29},
        {0x011E, 30},
        {0x011F, 31},
        {0x0120, 32},

        // ASCII symbols
        {0x21, 33},
        {0x22, 34},
        {0x23, 35},
        {0x24, 36},
        {0x25, 37},
        {0x26, 38},
        {0x27, 39},
        {0x28, 40},
        {0x29, 41},
        {0x2A, 42},
        {0x2B, 43},
        {0x2C, 44},
        {0x2D, 45},
        {0x2E, 46},
        {0x2F, 47},
        {0x3A, 58},
        {0x3B, 59},
        {0x3C, 60},
        {0x3D, 61},
        {0x3E, 62},
        {0x3F, 63},
        {0x40, 64},
        {0x41, 65},
        {0x42, 66},
        {0x43, 67},
        {0x44, 68},
        {0x45, 69},
        {0x46, 70},
        {0x47, 71},
        {0x48, 72},
        {0x49, 73},
        {0x4A, 74},
        {0x4B, 75},
        {0x4C, 76},
        {0x4D, 77},
        {0x4E, 78},
        {0x4F, 79},
        {0x50, 80},
        {0x51, 81},
        {0x52, 82},
        {0x53, 83},
        {0x54, 84},
        {0x55, 85},
        {0x56, 86},
        {0x57, 87},
        {0x58, 88},
        {0x59, 89},
        {0x5A, 90},
        {0x5B, 91},
        {0x5C, 92},
        {0x5D, 93},
        {0x5E, 94},
        {0x5F, 95},
        {0x60, 96},
        {0x61, 97},
        {0x62, 98},
        {0x63, 99},
        {0x64, 100},
        {0x65, 101},
        {0x66, 102},
        {0x67, 103},
        {0x68, 104},
        {0x69, 105},
        {0x6A, 106},
        {0x6B, 107},
        {0x6C, 108},
        {0x6D, 109},
        {0x6E, 110},
        {0x6F, 111},
        {0x70, 112},
        {0x71, 113},
        {0x72, 114},
        {0x73, 115},
        {0x74, 116},
        {0x75, 117},
        {0x76, 118},
        {0x77, 119},
        {0x78, 120},
        {0x79, 121},
        {0x7A, 122},
        {0x7B, 123},
        {0x7C, 124},
        {0x7D, 125},
        {0x7E, 126},

        // Extended Latin characters
        {0x0121, 127},
        {0x0122, 128},
        {0x0123, 129},
        {0x0124, 130},
        {0x0125, 131},
        {0x0126, 132},
        {0x0127, 133},
        {0x0128, 134},
        {0x0129, 135},
        {0x012A, 136},
        {0x012B, 137},
        {0x012C, 138},
        {0x012D, 139},
        {0x012E, 140},
        {0x012F, 141},
        {0x0130, 142},
        {0x0131, 143},
        {0x0132, 144},
        {0x0133, 145},
        {0x0134, 146},
        {0x0135, 147},
        {0x0136, 148},
        {0x0137, 149},
        {0x0138, 150},
        {0x0139, 151},
        {0x013A, 152},
        {0x013B, 153},
        {0x013C, 154},
        {0x013D, 155},
        {0x013E, 156},
        {0x013F, 157},
        {0x0140, 158},
        {0x0141, 159},
        {0x0142, 160},

        // Special characters
        {0x00A1, 161},
        {0x00A2, 162},
        {0x00A3, 163},
        {0x00A4, 164},
        {0x00A5, 165},
        {0x00A6, 166},
        {0x00A7, 167},
        {0x00A8, 168},
        {0x00A9, 169},
        {0x00AA, 170},
        {0x00AB, 171},
        {0x00AC, 172},
        {0x0143, 173},
        {0x00AE, 174},
        {0x00AF, 175},
        {0x00B0, 176},
        {0x00B1, 177},
        {0x00B2, 178},
        {0x00B3, 179},
        {0x00B4, 180},
        {0x00B5, 181},
        {0x00B6, 182},
        {0x00B7, 183},
        {0x00B8, 184},
        {0x00B9, 185},
        {0x00BA, 186},
        {0x00BB, 187},
        {0x00BC, 188},
        {0x00BD, 189},
        {0x00BE, 190},
        {0x00BF, 191},
        {0x00C0, 192},
        {0x00C1, 193},
        {0x00C2, 194},
        {0x00C3, 195},
        {0x00C4, 196},
        {0x00C5, 197},
        {0x00C6, 198},
        {0x00C7, 199},
        {0x00C8, 200},
        {0x00C9, 201},
        {0x00CA, 202},
        {0x00CB, 203},
        {0x00CC, 204},
        {0x00CD, 205},
        {0x00CE, 206},
        {0x00CF, 207},
        {0x00D0, 208},
        {0x00D1, 209},
        {0x00D2, 210},
        {0x00D3, 211},
        {0x00D4, 212},
        {0x00D5, 213},
        {0x00D6, 214},
        {0x00D7, 215},
        {0x00D8, 216},
        {0x00D9, 217},
        {0x00DA, 218},
        {0x00DB, 219},
        {0x00DC, 220},
        {0x00DD, 221},
        {0x00DE, 222},
        {0x00DF, 223},
        {0x00E0, 224},
        {0x00E1, 225},
        {0x00E2, 226},
        {0x00E3, 227},
        {0x00E4, 228},
        {0x00E5, 229},
        {0x00E6, 230},
        {0x00E7, 231},
        {0x00E8, 232},
        {0x00E9, 233},
        {0x00EA, 234},
        {0x00EB, 235},
        {0x00EC, 236},
        {0x00ED, 237},
        {0x00EE, 238},
        {0x00EF, 239},
        {0x00F0, 240},
        {0x00F1, 241},
        {0x00F2, 242},
        {0x00F3, 243},
        {0x00F4, 244},
        {0x00F5, 245},
        {0x00F6, 246},
        {0x00F7, 247},
        {0x00F8, 248},
        {0x00F9, 249},
        {0x00FA, 250},
        {0x00FB, 251},
        {0x00FC, 252},
        {0x00FD, 253},
        {0x00FE, 254},
        {0x00FF, 255}
    };

    Florence2Tokenizer::Florence2Tokenizer(
        const std::unordered_map<std::string, int> &vocabulary,
        const std::unordered_map<std::string, AddedToken> &added_tokens,
        std::shared_ptr<Tokens> tokens)
        : tokens_(tokens),
          regex_(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)") {
        // Build byte_to_unicode from unicode_to_bytes
        for (const auto &[unicode, byte]: unicode_to_bytes) {
            byte_to_unicode[byte] = unicode;
        }
        // Initialize vocabulary array
        for (const auto &[token, index]: vocabulary) {
            if (!vocabulary_[index].empty()) {
                throw std::invalid_argument("Duplicate vocabulary index");
            }
            vocabulary_[index] = token;
        }

        // Process added tokens
        for (const auto &[id_str, token]: added_tokens) {
            int id = std::stoi(id_str);
            if (!vocabulary_[id].empty() && vocabulary_[id] != token.content) {
                throw std::invalid_argument("Token conflict at index " + std::to_string(id));
            }
            vocabulary_[id] = token.content;
        }

        // Verify vocabulary is not empty
        if (std::all_of(vocabulary_.begin(), vocabulary_.end(),
                        [](const std::string &s) { return s.empty(); })) {
            throw std::runtime_error("Vocabulary is empty");
        }

        // Build vocabulary dictionary
        for (size_t i = 0; i < vocabulary_.size(); i++) {
            if (!vocabulary_[i].empty()) {
                vocabulary_dict_[vocabulary_[i]] = i;
            }
        }

        // Initialize special tokens
        special_tokens = std::unordered_set<std::string>();
        for (const auto &[id_str, token]: added_tokens) {
            if (token.special) {
                special_tokens.insert(token.content);
            }
        }
        this->added_tokens = added_tokens;
    }

    // Helper function to parse TokenizerConfig from JSON
    TokenizerConfig Florence2Tokenizer::parse_tokenizer_config(const nlohmann::json &j) {
        TokenizerConfig config;

        // Parse required fields
        j.at("add_prefix_space").get_to(config.add_prefix_space);
        j.at("clean_up_tokenization_spaces").get_to(config.clean_up_tokenization_spaces);
        j.at("cls_token").get_to(config.cls_token);
        j.at("mask_token").get_to(config.mask_token);
        j.at("model_max_length").get_to(config.model_max_length);
        j.at("pad_token").get_to(config.pad_token);
        j.at("sep_token").get_to(config.sep_token);
        j.at("unk_token").get_to(config.unk_token);

        // Parse optional fields
        if (j.contains("bos_token")) {
            j.at("bos_token").get_to(config.bos_token);
        }
        if (j.contains("eos_token")) {
            j.at("eos_token").get_to(config.eos_token);
        }

        // Parse added_tokens_decoder
        if (j.contains("added_tokens_decoder")) {
            auto &added_tokens = j.at("added_tokens_decoder");
            for (const auto &[key, value]: added_tokens.items()) {
                AddedToken token;
                value.at("content").get_to(token.content);
                value.at("lstrip").get_to(token.lstrip);
                value.at("normalized").get_to(token.normalized);
                value.at("rstrip").get_to(token.rstrip);
                value.at("single_word").get_to(token.single_word);
                value.at("special").get_to(token.special);
                config.added_tokens_decoder[key] = token;
            }
        }

        return config;
    }

    std::shared_ptr<Florence2Tokenizer> Florence2Tokenizer::init() {
        // Load vocab.json
        auto vocab_stream = ResourceLoader::open_resource("vocab.json");
        if (!vocab_stream || !vocab_stream->good()) {
            throw std::runtime_error("Failed to load vocab.json");
        }
        nlohmann::json vocab_json;
        *vocab_stream >> vocab_json;

        // Load tokenizer_config.json
        auto config_stream = ResourceLoader::open_resource("tokenizer_config.json");
        if (!config_stream || !config_stream->good()) {
            throw std::runtime_error("Failed to load tokenizer_config.json");
        }
        nlohmann::json config_json;
        *config_stream >> config_json;

        // Parse config
        auto tokenizer_config = parse_tokenizer_config(config_json);

        // Create vocabulary map
        std::unordered_map<std::string, int> vocab;
        for (const auto &[token, index]: vocab_json.items()) {
            vocab[token] = index.get<int>();
        }

        // Create Tokens instance
        auto tokens = std::make_shared<Tokens>(
            tokenizer_config.pad_token,
            tokenizer_config.unk_token,
            tokenizer_config.cls_token,
            tokenizer_config.sep_token,
            tokenizer_config.mask_token,
            tokenizer_config.eos_token.empty() ? std::nullopt : std::make_optional(tokenizer_config.eos_token),
            tokenizer_config.bos_token.empty() ? std::nullopt : std::make_optional(tokenizer_config.bos_token));

        return std::make_shared<Florence2Tokenizer>(
            vocab,
            tokenizer_config.added_tokens_decoder,
            tokens);
    }

    std::string Florence2Tokenizer::id_to_token(int id) const {
        return vocabulary_[id];
    }

    int Florence2Tokenizer::token_to_id(const std::string &token) const {
        // C# uses Array.IndexOf, which returns -1 if not found
        // For C++, we'll search our vocabulary array
        for (size_t i = 0; i < vocabulary_.size(); i++) {
            if (vocabulary_[i] == token) {
                return i;
            }
        }
        return -1;
    }

    std::vector<std::string> Florence2Tokenizer::convert_ids_to_tokens(
        const std::vector<int> &ids,
        bool skip_special_tokens) {
        std::vector<std::string> result;
        result.reserve(ids.size());

        // Convert IDs to tokens
        for (const auto &id: ids) {
            result.push_back(id_to_token(id));
        }

        // Filter special tokens if requested
        if (skip_special_tokens) {
            std::vector<std::string> filtered;
            std::copy_if(result.begin(), result.end(),
                         std::back_inserter(filtered),
                         [this](const std::string &token) {
                             return special_tokens.find(token) == special_tokens.end();
                         });
            result = std::move(filtered);
        }

        return result;
    }

    std::vector<std::string> Florence2Tokenizer::untokenize(std::vector<std::string> tokens) {
        std::vector<std::string> result;

        // Don't need to reverse tokens like in original code
        for (const auto &token: tokens) {
            // Skip if it's a special token
            if (special_tokens.find(token) != special_tokens.end()) {
                continue;
            }

            // Handle byte-level decoding
            std::string decoded;

            // Check if token starts with Ġ (0xC4 0xA0)
            if (token.length() >= 2 &&
                static_cast<unsigned char>(token[0]) == 0xC4 &&
                static_cast<unsigned char>(token[1]) == 0xA0) {
                // Add space before the actual content
                decoded = " " + token.substr(2);
            } else {
                decoded = token;
            }

            result.push_back(decoded);
        }

        return result;
    }

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t> > >
    Florence2Tokenizer::encode(const std::vector<std::string> &texts) {
        auto tokenized = tokenize(texts);

        if (tokenized.empty()) {
            return {};
        }

        // Find max sequence length among all tokenized texts
        size_t sequence_length = 0;
        for (const auto &tokens: tokenized) {
            sequence_length = std::max(sequence_length,
                                       std::min(static_cast<size_t>(MAX_TOKENS),
                                                tokens.size()));
        }

        std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t> > > result;
        result.reserve(tokenized.size());

        for (const auto &tokens: tokenized) {
            // Calculate actual token length (not exceeding MAX_TOKENS)
            size_t token_length = std::min(static_cast<size_t>(MAX_TOKENS), tokens.size());
            size_t padding_size = sequence_length - token_length;

            // Create token indexes and attention mask
            std::vector<int64_t> token_indexes;
            std::vector<int64_t> attention_mask;
            token_indexes.reserve(sequence_length);
            attention_mask.reserve(sequence_length);

            // Add tokens up to MAX_TOKENS
            for (size_t i = 0; i < token_length; ++i) {
                token_indexes.push_back(tokens[i].second);
                attention_mask.push_back(1);
            }

            // Add padding
            token_indexes.insert(token_indexes.end(), padding_size, 0L);
            attention_mask.insert(attention_mask.end(), padding_size, 0L);

            result.emplace_back(std::move(token_indexes), std::move(attention_mask));
        }

        return result;
    }

    std::vector<std::string> Florence2Tokenizer::tokenize_sentence(const std::string &text) {
        std::vector<std::string> tokens;
        std::string current_word;
        bool is_first = true;

        // First split into words and punctuation
        for (size_t i = 0; i < text.length(); i++) {
            char c = text[i];

            if (std::isspace(c) || std::ispunct(c)) {
                // Process any accumulated word before the space/punctuation
                if (!current_word.empty()) {
                    // Add Ġ prefix for non-first words
                    if (!is_first) {
                        current_word = std::string{static_cast<char>(0xC4), static_cast<char>(0xA0)} + current_word;
                    }

                    // Try to find subwords
                    size_t start = 0;
                    while (start < current_word.length()) {
                        size_t longest_match = 0;
                        std::string matched_token;

                        // Try different lengths of subwords
                        for (size_t len = 1; len <= current_word.length() - start; len++) {
                            std::string potential = current_word.substr(start, len);
                            if (vocabulary_dict_.find(potential) != vocabulary_dict_.end()) {
                                longest_match = len;
                                matched_token = potential;
                            }
                        }

                        if (longest_match > 0) {
                            tokens.push_back(matched_token);
                            start += longest_match;
                        } else {
                            // If no match found, take one character and continue
                            tokens.push_back(current_word.substr(start, 1));
                            start++;
                        }
                    }

                    current_word.clear();
                    is_first = false;
                }

                // Handle punctuation as separate token
                if (std::ispunct(c)) {
                    std::string punct(1, c);
                    if (vocabulary_dict_.find(punct) != vocabulary_dict_.end()) {
                        tokens.push_back(punct);
                    }
                }
            } else {
                current_word += c;
            }
        }

        // Handle last word
        if (!current_word.empty()) {
            if (!is_first) {
                current_word = std::string{static_cast<char>(0xC4), static_cast<char>(0xA0)} + current_word;
            }

            size_t start = 0;
            while (start < current_word.length()) {
                size_t longest_match = 0;
                std::string matched_token;

                for (size_t len = 1; len <= current_word.length() - start; len++) {
                    std::string potential = current_word.substr(start, len);
                    if (vocabulary_dict_.find(potential) != vocabulary_dict_.end()) {
                        longest_match = len;
                        matched_token = potential;
                    }
                }

                if (longest_match > 0) {
                    tokens.push_back(matched_token);
                    start += longest_match;
                } else {
                    tokens.push_back(current_word.substr(start, 1));
                    start++;
                }
            }
        }

        return tokens;
    }

    std::vector<std::vector<std::pair<std::string, int> > >
    Florence2Tokenizer::tokenize(const std::vector<std::string> &texts) {
        int unk_token_id = vocabulary_dict_[tokens_->unknown()];

        std::vector<std::vector<std::pair<std::string, int> > > result;
        result.reserve(texts.size());

        for (const auto &text: texts) {
            std::vector<std::pair<std::string, int> > tokens;

            // Add classification token (<s>)
            std::string cls_token = tokens_->classification();
            tokens.emplace_back(cls_token, vocabulary_dict_[cls_token]);

            // Get byte-level tokens
            auto sentence_tokens = tokenize_sentence(text);

            // Convert tokens to IDs
            for (const auto &token: sentence_tokens) {
                auto it = vocabulary_dict_.find(token);
                if (it != vocabulary_dict_.end()) {
                    tokens.emplace_back(token, it->second);
                } else {
                    tokens.emplace_back(tokens_->unknown(), unk_token_id);
                }
            }

            // Add separation token (</s>)
            std::string sep_token = tokens_->separation();
            tokens.emplace_back(sep_token, vocabulary_dict_[sep_token]);

            result.push_back(std::move(tokens));
        }

        return result;
    }

    std::vector<std::string> Florence2Tokenizer::split_and_keep(
        const std::string &input_string,
        const std::vector<char> &delimiters) {
        std::vector<std::string> result;
        size_t start = 0;

        // Convert delimiters vector to string for find_first_of
        std::string delimiter_str(delimiters.begin(), delimiters.end());

        size_t index;
        while ((index = input_string.find_first_of(delimiter_str, start)) != std::string::npos) {
            if (index - start > 0) {
                result.push_back(input_string.substr(start, index - start));
            }

            result.push_back(input_string.substr(index, 1));

            start = index + 1;
        }

        if (start < input_string.length()) {
            result.push_back(input_string.substr(start));
        }

        return result;
    }
} // namespace florence2
