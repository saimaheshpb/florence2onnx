#include "downloader/florence_model_downloader.h"
#include <iostream>
#include <cassert>
#include <filesystem>

namespace florence2
{
    namespace test
    {

        class FlorenceModelDownloaderTest
        {
        public:
            void run_all_tests()
            {
                std::cout << "Starting FlorenceModelDownloader tests..." << std::endl;

                try
                {
                    test_initialization();
                    test_model_paths();
                    test_status_generation();
                    // test_error_handling();
                    test_download_functionality();

                    std::cout << "\nAll FlorenceModelDownloader tests passed!" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: " << e.what() << std::endl;
                    throw;
                }
            }

        private:
            void test_initialization()
            {
                std::cout << "\nTesting FlorenceModelDownloader initialization..." << std::endl;

                std::string test_path = "./tests/test_resources";
                FlorenceModelDownloader downloader(test_path);

                // Test initial state
                assert(!downloader.is_ready());
                std::cout << "Initial state check: OK" << std::endl;
            }

            void test_model_paths()
            {
                std::cout << "\nTesting model paths..." << std::endl;

                std::string test_path = "./tests/test_resources";
                FlorenceModelDownloader downloader(test_path);

                // Test try_get_model_path with non-existent model
                std::string model_path;
                try
                {
                    downloader.try_get_model_path(IModelSource::Model::DecoderModelMerged, model_path);
                    assert(false && "Expected exception for non-existent model");
                }
                catch (const std::runtime_error &)
                {
                    std::cout << "Non-existent model path handling: OK" << std::endl;
                }
            }

            void test_status_generation()
            {
                std::cout << "\nTesting status generation..." << std::endl;

                // Test status with zero elapsed time
                auto status = FlorenceModelDownloader::get_status(1000, 500, 0, 0.0, "Testing");
                assert(status.get_progress() == 0.5f);
                assert(!status.get_message().empty());
                std::cout << "Status (no elapsed time): " << status.get_message() << std::endl;

                // Test status with elapsed time and progress
                status = FlorenceModelDownloader::get_status(1000, 500, 250, 1.0, "Testing");
                assert(status.get_progress() == 0.5f);
                assert(!status.get_message().empty());
                assert(status.get_message().find("Testing") != std::string::npos);
                std::cout << "Status (with elapsed time): " << status.get_message() << std::endl;

                // Test edge cases
                status = FlorenceModelDownloader::get_status(0, 0, 0, 0.0, "Zero case");
                assert(status.get_progress() == 0.0f);
                std::cout << "Status edge cases: OK" << std::endl;
            }

            void test_error_handling()
            {
                std::cout << "\nTesting error handling..." << std::endl;

                std::string test_path = "./tests/test_resources";
                FlorenceModelDownloader downloader(test_path);

                // Test getting bytes of non-existent model
                try
                {
                    downloader.get_model_bytes(IModelSource::Model::DecoderModelMerged);
                    assert(false && "Expected exception for non-existent model bytes");
                }
                catch (const std::runtime_error &)
                {
                    std::cout << "Non-existent model bytes handling: OK" << std::endl;
                }

                // Test invalid model type
                try
                {
                    std::string model_path;
                    downloader.try_get_model_path(static_cast<IModelSource::Model>(999), model_path);
                    assert(false && "Expected exception for invalid model type");
                }
                catch (const std::runtime_error &)
                {
                    std::cout << "Invalid model type handling: OK" << std::endl;
                }
            }

            void test_download_functionality()
            {
                std::cout << "\nTesting download functionality..." << std::endl;

                std::string test_path = "./models";
                FlorenceModelDownloader downloader(test_path);

                // Status callback for download progress
                auto status_callback = [](const IStatus &status)
                {
                    if (!status.get_error().empty())
                    {
                        std::cout << "\nDownload error: " << status.get_error() << std::endl;
                    }
                    else
                    {
                        std::cout << "\rDownload progress: " << (status.get_progress() * 100.0f)
                                  << "% - " << status.get_message() << std::flush;
                    }
                };

                // Test single model download
                bool success = downloader.download_model(
                    IModelSource::Model::DecoderModelMerged,
                    status_callback);

                if (!success)
                {
                    throw std::runtime_error("Single model download failed");
                }
                std::cout << "\nSingle model download: OK" << std::endl;

                // Test downloading all models
                success = downloader.download_models(status_callback);
                if (!success)
                {
                    throw std::runtime_error("All models download failed");
                }
                std::cout << "\nAll models download: OK" << std::endl;
            }
        };

    } // namespace test
} // namespace florence2

int main()
{
    try
    {
        florence2::test::FlorenceModelDownloaderTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}