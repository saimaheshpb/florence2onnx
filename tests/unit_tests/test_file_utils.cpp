#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <filesystem>
#include "utils/file_utils.h"

namespace florence2 {
namespace test {

class FileUtilsTest {
private:
    const std::string test_base_path = "test_resources/test_output";

    void test_lock_file_operations() {
        std::cout << "\nTesting lock file operations..." << std::endl;

        std::string test_path = test_base_path + "/locks/test_file";
        
        // Clean up any existing files
        std::filesystem::remove_all(test_base_path + "/locks");

        // Test creating lock file for the first time
        bool lock_created = file_utils::create_lock_file(test_path);
        assert(lock_created && "First lock creation should succeed");
        assert(std::filesystem::exists(test_path + ".lock"));
        std::cout << "Initial lock file creation: OK" << std::endl;

        // Test attempting to create lock file when it already exists
        bool second_lock = file_utils::create_lock_file(test_path);
        assert(!second_lock && "Second lock creation should fail");
        std::cout << "Duplicate lock prevention: OK" << std::endl;

        // Create the actual file
        std::filesystem::create_directories(std::filesystem::path(test_path).parent_path());
        {
            std::ofstream test_file(test_path);
            test_file << "test content";
        }

        // Verify file isn't complete while lock exists
        assert(!file_utils::is_file_complete(test_path));
        std::cout << "Incomplete file check with lock: OK" << std::endl;

        // Remove lock
        bool lock_removed = file_utils::remove_lock_file(test_path);
        assert(lock_removed);
        assert(!std::filesystem::exists(test_path + ".lock"));
        std::cout << "Lock file removal: OK" << std::endl;

        // Verify file is now complete
        assert(file_utils::is_file_complete(test_path));
        std::cout << "Complete file check: OK" << std::endl;

        // Test lock file in non-existent directory
        std::string deep_path = test_base_path + "/locks/deep/nested/test_file";
        bool deep_lock_created = file_utils::create_lock_file(deep_path);
        assert(deep_lock_created && "Lock creation should create parent directories");
        assert(std::filesystem::exists(deep_path + ".lock"));
        std::cout << "Deep path lock creation: OK" << std::endl;

        // Clean up
        std::filesystem::remove_all(test_base_path + "/locks");
    }

    void test_directory_operations() {
        std::cout << "\nTesting directory operations..." << std::endl;

        std::string test_dir = test_base_path + "/nested/path";
        std::string test_file = test_dir + "/test.txt";

        // Clean up any existing directory
        std::filesystem::remove_all(test_base_path + "/nested");

        // Test directory creation
        file_utils::ensure_directory_exists(test_file);
        assert(std::filesystem::exists(test_dir));
        std::cout << "Directory creation: OK" << std::endl;

        // Try creating it again (should succeed)
        file_utils::ensure_directory_exists(test_file);
        std::cout << "Repeat directory creation: OK" << std::endl;

        // Clean up
        std::filesystem::remove_all(test_base_path + "/nested");
    }

public:
    void run_all_tests() {
        std::cout << "Starting File Utils tests..." << std::endl;

        try {
            // Ensure base test directory exists
            std::filesystem::create_directories(test_base_path);

            test_lock_file_operations();
            test_directory_operations();

            // Clean up
            std::filesystem::remove_all(test_base_path);

            std::cout << "\nAll File Utils tests passed!" << std::endl;
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
        florence2::test::FileUtilsTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}