#include "model/logits_processor.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace florence2 {
namespace test {

class LogitsProcessorTest {
public:
    void run_all_tests() {
        std::cout << "Starting LogitsProcessor tests..." << std::endl;

        try {
            test_no_repeat_ngram_processor();
            test_forced_bos_processor();
            test_forced_eos_processor();
            test_batch_slice();

            std::cout << "\nAll LogitsProcessor tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_batch_slice() {
        std::cout << "\nTesting batch slice..." << std::endl;
        
        // Test 2D tensor
        std::vector<float> tensor_2d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> dims_2d = {2, 3}; // 2 batches, 3 values each
        
        auto slice = LogitsProcessor::get_batch_slice(1, tensor_2d, dims_2d);
        assert(slice.size() == 3);
        assert(slice[0] == 4.0f && slice[1] == 5.0f && slice[2] == 6.0f);
        
        // Test 3D tensor with dimension[1] = 1
        std::vector<float> tensor_3d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> dims_3d = {2, 1, 3}; // 2 batches, 1 middle dim, 3 values
        
        auto slice_3d = LogitsProcessor::get_batch_slice(1, tensor_3d, dims_3d);
        assert(slice_3d.size() == 3);
        assert(slice_3d[0] == 4.0f && slice_3d[1] == 5.0f && slice_3d[2] == 6.0f);
        
        std::cout << "Batch slice: OK" << std::endl;
    }

    void test_no_repeat_ngram_processor() {
        std::cout << "\nTesting NoRepeatNGramLogitsProcessor..." << std::endl;
        
        NoRepeatNGramLogitsProcessor processor(3); // ngram size of 3
        
        // Test sequence: [1, 2, 3, 1, 2]
        // Should ban token 3 as next token to prevent [1, 2, 3] from repeating
        std::vector<int64_t> input_ids = {1, 2, 3, 1, 2};
        std::vector<float> logits(10, 0.0f); // 10 possible tokens
        std::vector<int64_t> dimensions = {1, 10}; // 1 batch, 10 tokens
        
        processor.process(0, input_ids, logits, dimensions);
        
        // Check that token 3 is banned (set to negative infinity)
        assert(std::isinf(logits[3]) && logits[3] < 0);
        assert(!std::isinf(logits[4])); // Other tokens should be unchanged
        
        std::cout << "NoRepeatNGramLogitsProcessor: OK" << std::endl;
    }

    void test_forced_bos_processor() {
        std::cout << "\nTesting ForcedBOSTokenLogitsProcessor..." << std::endl;
        
        const int bos_token_id = 1;
        ForcedBOSTokenLogitsProcessor processor(bos_token_id);
        
        // Test at first position
        std::vector<int64_t> input_ids = {0}; // Single token
        std::vector<float> logits(10, 0.0f);
        std::vector<int64_t> dimensions = {1, 10};
        
        processor.process(0, input_ids, logits, dimensions);
        
        // Check that only BOS token is allowed
        for (int i = 0; i < 10; i++) {
            if (i == bos_token_id) {
                assert(logits[i] == 0.0f);
            } else {
                assert(std::isinf(logits[i]) && logits[i] < 0);
            }
        }
        
        std::cout << "ForcedBOSTokenLogitsProcessor: OK" << std::endl;
    }

    void test_forced_eos_processor() {
        std::cout << "\nTesting ForcedEOSTokenLogitsProcessor..." << std::endl;
        
        const int max_length = 5;
        const int eos_token_id = 2;
        ForcedEOSTokenLogitsProcessor processor(max_length, eos_token_id);
        
        // Test at max_length - 1
        std::vector<int64_t> input_ids(max_length - 1, 0);
        std::vector<float> logits(10, 0.0f);
        std::vector<int64_t> dimensions = {1, 10};
        
        processor.process(0, input_ids, logits, dimensions);
        
        // Check that only EOS token is allowed
        for (int i = 0; i < 10; i++) {
            if (i == eos_token_id) {
                assert(logits[i] == 0.0f);
            } else {
                assert(std::isinf(logits[i]) && logits[i] < 0);
            }
        }
        
        // Test before max_length - 1
        std::vector<int64_t> short_input_ids(max_length - 2, 0);
        std::vector<float> normal_logits(10, 1.0f);
        
        processor.process(0, short_input_ids, normal_logits, dimensions);
        
        // Check that no tokens are forced
        for (float logit : normal_logits) {
            assert(logit == 1.0f);
        }
        
        std::cout << "ForcedEOSTokenLogitsProcessor: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::LogitsProcessorTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}