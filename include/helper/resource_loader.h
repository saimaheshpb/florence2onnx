#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <memory>

namespace florence2 {

/*
 * C++ Port Changes from C#:
 * 1. Changed from static class to class with static methods
 * 2. Replaced Assembly.GetManifestResourceStream with filesystem operations
 * 3. Using std::vector<uint8_t> instead of byte[]
 * 4. Added resource directory path configuration
 * 5. Using RAII principles with file handling
 * 6. Added error handling with exceptions
 */
class ResourceLoader {
public:
    static void set_resource_path(const std::string& path) {
        resource_path_ = path;
    }

    // Equivalent to OpenResource - returns unique_ptr to ifstream
    static std::unique_ptr<std::ifstream> open_resource(const std::string& resource_file);

    // Equivalent to GetResource - returns vector of bytes
    static std::vector<uint8_t> get_resource(const std::string& resource_file);

private:
    static std::filesystem::path get_resource_path(const std::string& resource_file);
    inline static std::string resource_path_ = "Resources";  // Default path
};

} // namespace florence2