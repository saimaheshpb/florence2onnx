#include "post_processing/post_processor.h"
#include <iostream>
#include <cassert>

namespace florence2 {
namespace test {

class Florence2PostProcessorTest {
private:
    void test_task_type_mapping() {
        std::cout << "\nTesting task type mapping..." << std::endl;

        try {
            // Test each task type mapping
            assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OCR) == 
                   PostProcessingTypes::pure_text);
            assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OCR_WITH_REGION) == 
                   PostProcessingTypes::ocr_with_region);
            assert(Florence2PostProcessor::GetPostProcessingType(TaskType::CAPTION) == 
                   PostProcessingTypes::pure_text);
            assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OD) == 
                   PostProcessingTypes::description_with_bboxes);

            std::cout << "Task type mapping: OK" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Task type mapping test failed: " << e.what() << std::endl;
            throw;
        }
    }

    void test_initialization() {
        std::cout << "\nTesting Florence2PostProcessor initialization..." << std::endl;

        try {
            auto processor = std::make_unique<Florence2PostProcessor>();
            assert(processor != nullptr);
            std::cout << "Basic initialization: OK" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Initialization test failed: " << e.what() << std::endl;
            throw;
        }
    }

    void test_config_initialization() {
        std::cout << "\nTesting configuration initialization..." << std::endl;

        try {
            Florence2PostProcessorConfig config;
            
            // Test default values
            assert(config.numBboxHeightBins == 1000);
            assert(config.numBboxWidthBins == 1000);
            assert(config.boxQuantizationMode == QuantizerMode::Floor);
            assert(config.coordinatesHeightBins == 1000);
            assert(config.coordinatesWidthBins == 1000);
            assert(!config.parseTasks.empty());

            // Test parse tasks initialization
            bool hasOdTask = false;
            bool hasPhraseGrounding = false;
            for (const auto& task : config.parseTasks) {
                if (task.taskName == PostProcessingTypes::od) {
                    hasOdTask = true;
                }
                if (task.taskName == PostProcessingTypes::phrase_grounding) {
                    hasPhraseGrounding = true;
                    assert(task.filterByBlackList); // This task should have filterByBlackList set to true
                }
            }
            assert(hasOdTask);
            assert(hasPhraseGrounding);

            std::cout << "Configuration initialization: OK" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Configuration initialization test failed: " << e.what() << std::endl;
            throw;
        }
    }

public:
    void run_all_tests() {
        std::cout << "Starting Florence2PostProcessor tests..." << std::endl;

        try {
            test_initialization();
            test_task_type_mapping();
            test_config_initialization();

            std::cout << "\nAll Florence2PostProcessor tests passed successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
    }
};

} // namespace test
} // namespace florence2

int main() {
    try {
        florence2::test::Florence2PostProcessorTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}