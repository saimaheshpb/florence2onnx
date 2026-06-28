#include <iostream>
#include <cassert>
#include "downloader/status.h"
#include "model/model_source.h"

namespace florence2 {
namespace test {

class StatusModelTest {
private:
    void test_download_status() {
        std::cout << "\nTesting DownloadStatus..." << std::endl;
        
        DownloadStatus status;
        
        // Test initial values
        assert(status.get_error().empty());
        assert(status.get_message().empty());
        assert(status.get_progress() == 0.0f);
        std::cout << "Initial values check: OK" << std::endl;

        // Test setters and getters
        status.set_error("Test error");
        status.set_message("Test message");
        status.set_progress(0.5f);

        assert(status.get_error() == "Test error");
        assert(status.get_message() == "Test message");
        assert(status.get_progress() == 0.5f);
        std::cout << "Setters/Getters check: OK" << std::endl;

        // Test progress bounds
        status.set_progress(1.0f);
        assert(status.get_progress() == 1.0f);
        std::cout << "Progress bounds check: OK" << std::endl;
    }

    void test_model_enum() {
        std::cout << "\nTesting Model enum..." << std::endl;

        // Test all enum values are distinct
        std::vector<IModelSource::Model> models = {
            IModelSource::Model::DecoderModelMerged,
            IModelSource::Model::EmbedTokens,
            IModelSource::Model::EncoderModel,
            IModelSource::Model::VisionEncoder,
            IModelSource::Model::DecoderModel
        };

        for(size_t i = 0; i < models.size(); ++i) {
            for(size_t j = i + 1; j < models.size(); ++j) {
                assert(models[i] != models[j]);
            }
        }
        std::cout << "Model enum values check: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting Status and Model tests..." << std::endl;

        try {
            test_download_status();
            test_model_enum();

            std::cout << "\nAll Status and Model tests passed!" << std::endl;
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
        florence2::test::StatusModelTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}