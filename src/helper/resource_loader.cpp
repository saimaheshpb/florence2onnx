#include "helper/resource_loader.h"
#include <stdexcept>
#include <sstream>

namespace florence2 {

std::filesystem::path ResourceLoader::get_resource_path(const std::string& resource_file) {
    std::filesystem::path full_path = std::filesystem::path(resource_path_) / resource_file;
    if (!std::filesystem::exists(full_path)) {
        std::ostringstream error_msg;
        error_msg << "Resource file not found: " << full_path.string();
        throw std::runtime_error(error_msg.str());
    }
    return full_path;
}

std::unique_ptr<std::ifstream> ResourceLoader::open_resource(const std::string& resource_file) {
    auto path = get_resource_path(resource_file);
    auto stream = std::make_unique<std::ifstream>(path, std::ios::binary);
    if (!stream->is_open()) {
        std::ostringstream error_msg;
        error_msg << "Failed to open resource: " << path.string();
        throw std::runtime_error(error_msg.str());
    }
    return stream;
}

std::vector<uint8_t> ResourceLoader::get_resource(const std::string& resource_file) {
    auto path = get_resource_path(resource_file);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::ostringstream error_msg;
        error_msg << "Failed to open resource: " << path.string();
        throw std::runtime_error(error_msg.str());
    }

    // Get file size and create vector
    auto size = file.tellg();
    std::vector<uint8_t> buffer(static_cast<size_t>(size));

    // Read file content
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}

} // namespace florence2