#include "tokenizer/florence2_tokenizer.h"
#include "helper/resource_loader.h"
#include <iostream>
#include <cassert>

class TokenizerSpecificTest
{
private:
    void test_split_and_keep()
    {
        std::cout << "\nTesting split_and_keep..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test case 1: Basic splitting with space
        std::vector<char> delimiters = {' '};
        auto result = tokenizer->split_and_keep("Hello World", delimiters);
        std::cout << "Test 1 - 'Hello World' with space delimiter:" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 3); // "Hello", " ", "World"

        // Test case 2: Multiple delimiters
        delimiters = {' ', ',', '.'};
        result = tokenizer->split_and_keep("Hello, World. Test", delimiters);
        std::cout << "\nTest 2 - 'Hello, World. Test' with space, comma, period:" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 7); // "Hello", ",", " ", "World", ".", " ", "Test"

        // Test case 3: Empty string
        result = tokenizer->split_and_keep("", delimiters);
        assert(result.empty());
        std::cout << "\nTest 3 - Empty string: OK" << std::endl;

        // Test case 4: String with only delimiters
        result = tokenizer->split_and_keep(".,.", delimiters);
        std::cout << "\nTest 4 - String with only delimiters '.,.':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "'" << token << "'" << std::endl;
        }
        assert(result.size() == 3); // ".", ",", "."

        std::cout << "Split and keep tests passed" << std::endl;
    }

    void test_tokenize_sentence()
    {
        std::cout << "\nTesting tokenize_sentence..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test case 1: Basic space handling
        std::string text = "Hello world!";
        auto result = tokenizer->tokenize_sentence(text);
        std::cout << "Test 1 - Basic text 'Hello world!':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        // Test case 2: Multiple spaces
        text = "Hello   world";
        result = tokenizer->tokenize_sentence(text);
        std::cout << "\nTest 2 - Multiple spaces 'Hello   world':" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        // Test case 3: Contractions and special characters
        text = "I'll don't won't";
        result = tokenizer->tokenize_sentence(text);
        std::cout << "\nTest 3 - Contractions:" << std::endl;
        for (const auto &token : result)
        {
            std::cout << "Token: '" << token << "'" << std::endl;
        }

        // Test case 4: Full encoding test
        text = "Hello world!";
        auto tokenized = tokenizer->tokenize({text});
        std::cout << "\nTest 4 - Full encoding of 'Hello world!':" << std::endl;
        for (const auto &token_array : tokenized)
        {
            for (const auto &[token, id] : token_array)
            {
                std::cout << "'" << token << "' -> " << id;
                if (tokenizer->special_tokens.find(token) != tokenizer->special_tokens.end())
                {
                    std::cout << " (special token)";
                }
                std::cout << std::endl;
            }
        }
    }

    void test_token_encoding()
    {
        std::cout << "\nTesting token encoding..." << std::endl;
        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Test with a simple sentence
        std::string test_text = "Hello world!";
        std::cout << "Input text: '" << test_text << "'" << std::endl;

        // First look at intermediate steps
        auto tokenized = tokenizer->tokenize({test_text});
        std::cout << "\nTokenized form (token -> ID mapping):" << std::endl;
        for (const auto &token_array : tokenized)
        {
            for (const auto &[token, id] : token_array)
            {
                std::cout << "'" << token << "' -> " << id;
                // Print if it's a special token
                if (tokenizer->special_tokens.find(token) != tokenizer->special_tokens.end())
                {
                    std::cout << " (special token)";
                }
                std::cout << std::endl;
            }
        }

        // Then check final encoded form
        auto encoded = tokenizer->encode({test_text});
        std::cout << "\nFinal encoded form:" << std::endl;
        for (const auto &[input_ids, attention_mask] : encoded)
        {
            std::cout << "Input IDs: ";
            for (const auto &id : input_ids)
            {
                // Print the ID and corresponding token
                std::cout << id << "(" << tokenizer->id_to_token(id) << ") ";
            }
            std::cout << "\nAttention mask: ";
            for (const auto &mask : attention_mask)
            {
                std::cout << mask << " ";
            }
            std::cout << std::endl;
        }
    }

public:
    void run_all_tests()
    {
        std::cout << "Starting Tokenizer Specific Tests..." << std::endl;

        try
        {
            florence2::ResourceLoader::set_resource_path("./src/resources");

            test_split_and_keep();
            // test_tokenize_sentence();
            test_token_encoding();

            std::cout << "\nAll specific tests passed!" << std::endl;
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
        TokenizerSpecificTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}