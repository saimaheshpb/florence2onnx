#include "model/stopping_criteria.h"
#include <iostream>
#include <cassert>

namespace florence2 {
namespace test {

class StoppingCriteriaTest {
public:
    void run_all_tests() {
        std::cout << "Starting StoppingCriteria tests..." << std::endl;

        try {
            test_max_length_criteria();
            test_eos_token_criteria();

            std::cout << "\nAll StoppingCriteria tests passed!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void test_max_length_criteria() {
        std::cout << "\nTesting MaxLengthCriteria..." << std::endl;
        
        MaxLengthCriteria criteria(3);  // max length of 3
        
        // Test cases
        std::vector<std::vector<int64_t>> input_ids = {
            {1, 2, 3},      // length = 3, should stop
            {1, 2},         // length = 2, should continue
            {1, 2, 3, 4},   // length = 4, should stop
            {}              // length = 0, should continue
        };
        std::vector<double> scores = {1.0, 1.0, 1.0, 1.0};
        
        auto results = criteria.call(input_ids, scores);
        
        assert(results.size() == 4);
        assert(results[0] == true);   // length 3 = max_length
        assert(results[1] == false);  // length 2 < max_length
        assert(results[2] == true);   // length 4 > max_length
        assert(results[3] == false);  // length 0 < max_length
        
        std::cout << "MaxLengthCriteria: OK" << std::endl;
    }

    void test_eos_token_criteria() {
        std::cout << "\nTesting EosTokenCriteria..." << std::endl;
        
        const int64_t eos_token = 2;
        EosTokenCriteria criteria(eos_token);
        
        // Test cases
        std::vector<std::vector<int64_t>> input_ids = {
            {1, 2},     // ends with EOS
            {1, 3},     // doesn't end with EOS
            {2},        // single EOS token
            {}          // empty sequence
        };
        std::vector<double> scores = {1.0, 1.0, 1.0, 1.0};
        
        auto results = criteria.call(input_ids, scores);
        
        assert(results.size() == 4);
        assert(results[0] == true);   // ends with EOS
        assert(results[1] == false);  // doesn't end with EOS
        assert(results[2] == true);   // is EOS
        assert(results[3] == false);  // empty
        
        std::cout << "EosTokenCriteria: OK" << std::endl;
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::StoppingCriteriaTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}