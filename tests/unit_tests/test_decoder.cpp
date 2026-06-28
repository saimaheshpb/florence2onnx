#include "post_processing/decoder.h"
#include <iostream>
#include <cassert>

namespace florence2 {
namespace test {

class ByteDecoderTest {
public:
    void run_all_tests() {
        std::cout << "Starting ByteLevelDecoder tests..." << std::endl;
        ResourceLoader::set_resource_path("./src/resources/");
        try {
            test_initialization();
            test_convert_tokens();
            test_decode_chain();
            test_decode_with_special_tokens();

            std::cout << "\nAll ByteLevelDecoder tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_initialization() {
        std::cout << "\nTesting initialization..." << std::endl;
        
        std::unordered_map<std::string, AddedToken> added_tokens = {
            {"test", AddedToken{"test", false, true, false, false, true}}
        };
        ByteLevelDecoder decoder(added_tokens);
        
        std::cout << "Initialization: OK" << std::endl;
    }

    void test_convert_tokens() {
        std::cout << "\nTesting token conversion..." << std::endl;
        
        // Create tokenizer
        auto tokenizer = Florence2Tokenizer::init();
        
        // Create decoder with empty added tokens
        std::unordered_map<std::string, AddedToken> added_tokens;
        ByteLevelDecoder decoder(added_tokens);
        
        // Test basic token decoding
        std::vector<std::string> tokens = {"hello", "world"};
        auto result = decoder.decode_chain(tokenizer.get(), tokens);
        
        assert(result.size() == 1);
        assert(!result[0].empty());
        
        std::cout << "Basic token conversion: OK" << std::endl;
    }

    void test_decode_chain() {
        std::cout << "\nTesting decode chain..." << std::endl;
        
        auto tokenizer = Florence2Tokenizer::init();
        
        // Create decoder with no special tokens
        std::unordered_map<std::string, AddedToken> added_tokens;
        ByteLevelDecoder decoder(added_tokens);
        
        // Test sequence of normal tokens
        std::vector<std::string> tokens = {"this", "is", "a", "test"};
        auto result = decoder.decode_chain(tokenizer.get(), tokens);
        
        assert(result.size() == 1); // Should be combined into one string
        
        std::cout << "Normal token chain: OK" << std::endl;
    }

    void test_decode_with_special_tokens() {
        std::cout << "\nTesting decode with special tokens..." << std::endl;
        
        auto tokenizer = Florence2Tokenizer::init();
        
        // Create decoder with a special token
        std::unordered_map<std::string, AddedToken> added_tokens = {
            {"<sep>", AddedToken{"<sep>", false, true, false, false, true}}
        };
        ByteLevelDecoder decoder(added_tokens);
        
        // Test sequence with special token
        std::vector<std::string> tokens = {"first", "part", "<sep>", "second", "part"};
        auto result = decoder.decode_chain(tokenizer.get(), tokens);
        
        assert(result.size() == 3); // Should be split by special token
        assert(result[1] == "<sep>");
        
        std::cout << "Special token handling: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::ByteDecoderTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}