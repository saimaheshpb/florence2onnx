#include "helper/tokens.h"
#include <cassert>
#include <iostream>

class TokensTest {
private:
    void test_sentence_transformer_tokens() {
        std::cout << "\nTesting SentenceTransformerTokens..." << std::endl;

        florence2::SentenceTransformerTokens tokens;

        // Test all token values
        assert(tokens.padding() == "");
        assert(tokens.unknown() == "[UNK]");
        assert(tokens.classification() == "[CLS]");
        assert(tokens.separation() == "[SEP]");
        assert(tokens.mask() == "[MASK]");
        assert(!tokens.end_of_sequence().has_value());
        assert(!tokens.beginning_of_sequence().has_value());

        std::cout << "All token values verified: OK" << std::endl;
    }

    void test_tokens_inheritance() {
        std::cout << "\nTesting Tokens inheritance..." << std::endl;

        // Test that SentenceTransformerTokens is-a Tokens
        florence2::SentenceTransformerTokens sentence_tokens;
        florence2::Tokens* base_tokens = &sentence_tokens;

        // Verify values through base class pointer
        assert(base_tokens->padding() == "");
        assert(base_tokens->unknown() == "[UNK]");
        assert(base_tokens->classification() == "[CLS]");
        assert(base_tokens->separation() == "[SEP]");
        assert(base_tokens->mask() == "[MASK]");
        assert(!base_tokens->end_of_sequence().has_value());
        assert(!base_tokens->beginning_of_sequence().has_value());

        std::cout << "Inheritance hierarchy verified: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting Tokens tests..." << std::endl;

        try {
            test_sentence_transformer_tokens();
            test_tokens_inheritance();

            std::cout << "\nAll token tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        TokensTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}