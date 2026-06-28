#include "tokenizer/florence2_tokenizer.h"
#include "helper/resource_loader.h"
#include <iostream>
#include <cassert>
#include <cmath>

class Florence2TokenizerTest
{
private:
    void test_regex()
    {
        std::cout << "\nTesting regex pattern..." << std::endl;

        try
        {
            // Test with simplified pattern
            std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d|\s*[a-zA-Z]+|\s*[0-9]+|\s*[^\s\w]+|\s+)");

            // Test some sample matches
            std::string test_text = "Hello world! I'll try 123.";
            std::string result;
            std::smatch matches;
            std::string::const_iterator searchStart(test_text.cbegin());

            while (std::regex_search(searchStart, test_text.cend(), matches, pattern))
            {
                std::cout << "Found: " << matches[0] << std::endl;
                searchStart = matches.suffix().first;
            }

            std::cout << "Regex pattern compilation and testing: OK" << std::endl;
        }
        catch (const std::regex_error &e)
        {
            std::cerr << "Regex error: " << e.what() << std::endl;
            throw;
        }
    }
    void test_init()
    {
        std::cout << "\nTesting Florence2Tokenizer initialization..." << std::endl;

        try
        {
            florence2::ResourceLoader::set_resource_path("./src/resources/");

            auto tokenizer = florence2::Florence2Tokenizer::init();
            assert(tokenizer != nullptr);

            // Test tokens existence
            assert(tokenizer->get_tokens() != nullptr);

            std::cout << "Basic initialization: OK" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Tokenizer initialization failed: " << e.what() << std::endl;
            throw;
        }
    }

    void test_token_operations()
    {
        std::cout << "\nTesting token operations..." << std::endl;

        auto tokenizer = florence2::Florence2Tokenizer::init();

        // Debug: Print first few vocabulary entries
        std::cout << "First few vocabulary entries:" << std::endl;
        for (size_t i = 0; i < 10; i++)
        {
            auto token = tokenizer->id_to_token(i);
            if (!token.empty())
            {
                std::cout << "vocab[" << i << "]: " << token << std::endl;
            }
        }

        // // Debug: Print special tokens
        // std::cout << "\nSpecial tokens in vocabulary:" << std::endl;
        // for (const auto& special_token : tokenizer->special_tokens) {
        //     std::cout << special_token << std::endl;
        // }

        // Test with known token "happy"
        const std::string test_token = "happy";
        std::cout << "\nTesting with token: " << test_token << std::endl;
        int token_id = tokenizer->token_to_id(test_token);
        std::cout << "Token ID: " << token_id << std::endl;

        if (token_id != -1)
        {
            std::string recovered_token = tokenizer->id_to_token(token_id);
            std::cout << "Recovered token: " << recovered_token << std::endl;
            assert(recovered_token == test_token);
            std::cout << "Token conversion roundtrip: OK" << std::endl;
        }
        else
        {
            std::cout << "Token not found in vocabulary" << std::endl;
        }

        // Test converting multiple IDs with known valid IDs
        std::vector<int> test_ids = {token_id}; // Using the ID we just found
        auto tokens = tokenizer->convert_ids_to_tokens(test_ids, false);
        assert(tokens.size() == test_ids.size());
        std::cout << "\nConverted tokens:" << std::endl;
        for (const auto &token : tokens)
        {
            std::cout << token << std::endl;
        }
        std::cout << "Converting IDs: OK" << std::endl;

        // Test untokenize with actual tokens from vocabulary
        std::vector<std::string> test_tokens = {"happi", "##ness"}; // Using actual tokens
        auto untokenized = tokenizer->untokenize(test_tokens);
        assert(!untokenized.empty());
        std::cout << "\nUntokenized result: " << untokenized[0] << std::endl;
        assert(untokenized[0] == "happiness");
        std::cout << "Untokenization: OK" << std::endl;
    }

public:
    void run_all_tests()
    {
        std::cout << "Starting Florence2Tokenizer tests..." << std::endl;

        try
        {
            test_regex();
            test_init();
            test_token_operations();
            std::cout << "\nAll Florence2Tokenizer initialization tests passed!" << std::endl;
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