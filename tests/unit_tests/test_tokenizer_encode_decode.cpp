#include "tokenizer/florence2_tokenizer.h"
#include "helper/resource_loader.h"
#include <iostream>
#include <cassert>
#include <cmath>

class Florence2TokenizerTest
{
private:
    void test_initialization()
    {
        std::cout << "\nTesting initialization..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();
        assert(tokenizer != nullptr);
        std::cout << "Tokenizer initialized successfully" << std::endl;
    }

    void test_tokenize()
    {
        std::cout << "\nTesting tokenization..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test single text tokenization
        std::vector<std::string> texts = {"Hello world!"};
        auto result = tokenizer->tokenize(texts);

        // Verify structure
        assert(result.size() == 1);
        auto &tokens = result[0];

        // Should start with [CLS] and end with [SEP]
        assert(tokens.front().first == tokenizer->get_tokens()->classification());
        assert(tokens.back().first == tokenizer->get_tokens()->separation());

        // Print tokens for verification
        std::cout << "Tokenized 'Hello world!':" << std::endl;
        for (const auto &[token, id] : tokens)
        {
            std::cout << "Token: '" << token << "', ID: " << id << std::endl;
        }

        // Test multiple texts
        texts = {"Hello world!", "Testing tokenization"};
        result = tokenizer->tokenize(texts);
        assert(result.size() == 2);

        // Test empty input
        texts = {};
        result = tokenizer->tokenize(texts);
        assert(result.empty());

        // Test unknown tokens
        texts = {"@#$%"};
        result = tokenizer->tokenize(texts);
        assert(!result.empty());

        std::cout << "Tokenization tests passed" << std::endl;
    }

    void test_encode()
    {
        std::cout << "\nTesting encoding..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test single text encoding
        std::vector<std::string> texts = {"Hello world!"};
        auto result = tokenizer->encode(texts);

        // Verify structure
        assert(result.size() == 1);
        auto &[input_ids, attention_mask] = result[0];

        // Verify attention mask
        assert(input_ids.size() == attention_mask.size());
        assert(std::all_of(attention_mask.begin(), attention_mask.end(),
                           [](int64_t x)
                           { return x == 0 || x == 1; }));

        // Print encoded result
        std::cout << "Encoded 'Hello world!':" << std::endl;
        std::cout << "Input IDs: ";
        for (int64_t id : input_ids)
        {
            std::cout << id << " ";
        }
        std::cout << "\nAttention mask: ";
        for (int64_t mask : attention_mask)
        {
            std::cout << mask << " ";
        }
        std::cout << std::endl;

        // Test multiple texts of different lengths
        texts = {"Hello world!", "This is a longer text to test padding"};
        result = tokenizer->encode(texts);
        assert(result.size() == 2);

        // Verify all sequences have same length (padding)
        assert(result[0].first.size() == result[1].first.size());
        assert(result[0].second.size() == result[1].second.size());

        // Test empty input
        texts = {};
        result = tokenizer->encode(texts);
        assert(result.empty());

        // Test MAX_TOKENS limit
        std::string long_text(1000, 'a'); // Text longer than MAX_TOKENS
        texts = {long_text};
        result = tokenizer->encode(texts);
        assert(result[0].first.size() <= 512); // MAX_TOKENS is 512

        std::cout << "Encoding tests passed" << std::endl;
    }

    void test_tokenize_sentence()
    {
        std::cout << "\nTesting sentence tokenization..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test contractions
        std::vector<std::string> texts = {"I'll don't can't won't"};
        auto result = tokenizer->tokenize(texts);

        std::cout << "Tokenized contractions:" << std::endl;
        for (const auto &[token, id] : result[0])
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        // Test special characters
        texts = {"Hello, world! How are you?"};
        result = tokenizer->tokenize(texts);

        std::cout << "\nTokenized with punctuation:" << std::endl;
        for (const auto &[token, id] : result[0])
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        // Test numbers
        texts = {"Testing 123 456.789"};
        result = tokenizer->tokenize(texts);

        std::cout << "\nTokenized with numbers:" << std::endl;
        for (const auto &[token, id] : result[0])
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        std::cout << "Sentence tokenization tests passed" << std::endl;
    }

    void test_split_and_keep()
    {
        std::cout << "\nTesting split_and_keep..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test basic splitting with spaces
        std::vector<char> delimiters = {' '};
        auto test_string = "Hello World";
        auto result = tokenizer->split_and_keep(test_string, delimiters);

        std::cout << "Split result for 'Hello World':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 3); // "Hello", " ", "World"

        // Test multiple delimiters
        delimiters = {' ', ',', '.'};
        test_string = "Hello, World. Test";
        result = tokenizer->split_and_keep(test_string, delimiters);

        std::cout << "\nSplit result for 'Hello, World. Test':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 7); // "Hello", ",", " ", "World", ".", " ", "Test"

        // Test consecutive delimiters
        test_string = "Hello,,World";
        result = tokenizer->split_and_keep(test_string, delimiters);

        std::cout << "\nSplit result for 'Hello,,World':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 4); // "Hello", ",", ",", "World"

        std::cout << "Split and keep tests passed" << std::endl;
    }

public:
    // Add to run_all_tests()
    void run_all_tests()
    {
        std::cout << "Starting Florence2Tokenizer tests..." << std::endl;

        try
        {
            florence2::ResourceLoader::set_resource_path("./src/resources");

            test_initialization();
            test_tokenize();
            test_encode();
            test_tokenize_sentence();
            test_split_and_keep(); // Added this line

            std::cout << "\nAll tests passed!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

int main()
{
    try
    {
        Florence2TokenizerTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}