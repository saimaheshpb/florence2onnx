#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <thread>  // Add this for std::this_thread
#include <chrono>  // Add this for chrono literals
#include <iostream>  // Using std::cout for logging instead of spdlog (TODO)
#include "model/model_source.h"
#include "downloader/status.h"
#include "utils/file_utils.h"
#include "../../third_party/curl-android/include/curl/system.h"

namespace florence2 {

class FlorenceModelDownloader : public IModelSource {
private:
    std::string model_folder_base_path_;
    std::unordered_map<Model, std::string> model_paths_;

    // Helper functions
    static std::string get_model_filename(Model model);
    void pre_initialize_repository_from_disk();

public:
    explicit FlorenceModelDownloader(const std::string& model_folder_base_path);

    bool is_ready() const;

    // IModelSource interface implementation
    bool try_get_model_path(Model model, std::string& model_path) override;
    std::vector<uint8_t> get_model_bytes(Model model) override;

    // Download functionality
    bool download_models(std::function<void(const IStatus&)> on_status_update);

    // Single model download
    bool download_model(Model model,
                       std::function<void(const IStatus&)> on_status_update);

    static DownloadStatus get_status(int64_t total_download_size, 
                                   int64_t total_bytes_read,
                                   int64_t previous_bytes_read,
                                   double elapsed_seconds,
                                   const std::string& message = "");

private:
    // CURL callback helpers
    static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata);
    static int progress_callback(void* clientp, 
                               curl_off_t dltotal, 
                               curl_off_t dlnow,
                               curl_off_t ultotal, 
                               curl_off_t ulnow);

    struct DownloadContext {
        std::ofstream file;
        std::function<void(const IStatus&)> status_callback;
        int64_t total_size;
        int64_t downloaded;
        int64_t last_downloaded;
        std::chrono::steady_clock::time_point last_update;
        std::string message;
    };
};

} // namespace florence2