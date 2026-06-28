#include "post_processing/decoder_config.h"
#include <iostream>
#include <cassert>

namespace florence2 {
namespace test {

class DecoderConfigTest {
public:
    void run_all_tests() {
        std::cout << "Starting Decoder Config tests..." << std::endl;

        try {
            test_generation_config();
            test_normalized_config();

            std::cout << "\nAll Decoder Config tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_generation_config() {
        std::cout << "\nTesting GenerationConfig..." << std::endl;
        
        // Test static values
        assert(GenerationConfig::get_no_repeat_ngram_size() == 3);
        assert(GenerationConfig::get_num_beams() == 3);
        assert(GenerationConfig::get_max_length() == 1025);
        assert(GenerationConfig::get_top_k() == 50);
        
        std::cout << "GenerationConfig default values: OK" << std::endl;
    }

    void test_normalized_config() {
        std::cout << "\nTesting NormalizedConfig..." << std::endl;
        
        NormalizedConfig config;
        
        // Test decoder values
        assert(config.get_num_decoder_layers() == 6);
        assert(config.get_num_decoder_heads() == 12);
        assert(config.get_decoder_hidden_size() == 768);
        
        std::cout << "Decoder values: OK" << std::endl;
        
        // Test encoder values
        assert(config.get_num_encoder_layers() == 6);
        assert(config.get_num_encoder_heads() == 12);
        assert(config.get_encoder_hidden_size() == 768);
        
        std::cout << "Encoder values: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::DecoderConfigTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}