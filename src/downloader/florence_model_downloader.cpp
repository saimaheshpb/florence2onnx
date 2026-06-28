#include "downloader/florence_model_downloader.h"
#include <fstream>
#include <chrono>
#include <filesystem>
#include "helper/byte_formatter.h"
#include<vector>
#include "../../third_party/curl-android/include/curl/curl.h"

namespace florence2
{

    std::string FlorenceModelDownloader::get_model_filename(Model model)
    {
        switch (model)
        {
        case Model::DecoderModelMerged:
            return "decoder_model_merged_q4.onnx";
        case Model::EmbedTokens:
            return "embed_tokens_int8.onnx";
        case Model::EncoderModel:
            return "encoder_model_q4f16.onnx";
        case Model::VisionEncoder:
            return "vision_encoder_q4f16.onnx";
        case Model::DecoderModel:
            return "decoder_model_q4f16.onnx";
        default:
            throw std::invalid_argument("Invalid model type");
        }
    }

    FlorenceModelDownloader::FlorenceModelDownloader(const std::string &model_folder_base_path)
        : model_folder_base_path_(model_folder_base_path)
    {
        file_utils::ensure_directory_exists(model_folder_base_path);
        pre_initialize_repository_from_disk();
    }

    void FlorenceModelDownloader::pre_initialize_repository_from_disk()
    {
        for (int i = static_cast<int>(Model::DecoderModelMerged);
             i <= static_cast<int>(Model::DecoderModel); ++i)
        {
            auto model = static_cast<Model>(i);
            if (model_paths_.find(model) == model_paths_.end())
            {
                std::string model_filename = get_model_filename(model);
                std::string file_path = model_folder_base_path_ + "/" + model_filename;

                if (file_utils::is_file_complete(file_path))
                {
                    model_paths_[model] = file_path;
                }
            }
        }
    }

    bool FlorenceModelDownloader::is_ready() const
    {
        for (int i = static_cast<int>(Model::DecoderModelMerged);
             i <= static_cast<int>(Model::DecoderModel); ++i)
        {
            if (model_paths_.find(static_cast<Model>(i)) == model_paths_.end())
            {
                return false;
            }
        }
        return true;
    }

    bool FlorenceModelDownloader::try_get_model_path(Model model, std::string &model_path)
    {
        auto it = model_paths_.find(model);
        if (it != model_paths_.end())
        {
            model_path = it->second;
            return true;
        }
        throw std::runtime_error("Model not initialized. Call download_models first.");
    }

    std::vector<uint8_t> FlorenceModelDownloader::get_model_bytes(Model model)
    {
        std::string model_path;
        if (try_get_model_path(model, model_path))
        {
            std::ifstream file(model_path, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error("Failed to open model file: " + model_path);
            }

            // Get file size
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            // Read file content
            std::vector<uint8_t> buffer(size);
            if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
            {
                throw std::runtime_error("Failed to read model file: " + model_path);
            }

            return buffer;
        }
        throw std::runtime_error("Model not initialized. Call download_models first.");
    }

    DownloadStatus FlorenceModelDownloader::get_status(
        int64_t total_download_size,
        int64_t total_bytes_read,
        int64_t previous_bytes_read,
        double elapsed_seconds,
        const std::string &message)
    {
        DownloadStatus status;

        if (elapsed_seconds > 0)
        {
            double speed = static_cast<double>(total_bytes_read - previous_bytes_read) / elapsed_seconds;
            status.set_message(
                ByteFormatter::format_bytes(total_bytes_read) + " / " +
                ByteFormatter::format_bytes(total_download_size) + " (" +
                ByteFormatter::format_bytes(speed) + "/s) " + message + "\n");
        }
        else
        {
            status.set_message(
                ByteFormatter::format_bytes(total_bytes_read) + " / " +
                ByteFormatter::format_bytes(total_download_size) + " " + message + "\n");
        }

        status.set_progress(total_download_size > 0 ? static_cast<float>(total_bytes_read) / total_download_size : 0.0f);

        return status;
    }

    bool FlorenceModelDownloader::download_models(std::function<void(const IStatus &)> on_status_update)
    {
        if (is_ready())
        {
            return true;
        }

        bool success = true;

        // Download each missing model
        for (int i = static_cast<int>(Model::DecoderModelMerged);
             i <= static_cast<int>(Model::DecoderModel); ++i)
        {
            Model model = static_cast<Model>(i);
            if (model_paths_.find(model) == model_paths_.end())
            {
                if (!download_model(model, on_status_update))
                {
                    success = false;
                    break;
                }
            }
        }

        return success;
    }

    size_t FlorenceModelDownloader::write_callback(void *ptr, size_t size, size_t nmemb, void *userdata)
    {
        auto *context = static_cast<DownloadContext *>(userdata);
        size_t total_size = size * nmemb;

        if (!context->file.write(static_cast<char *>(ptr), total_size))
        {
            return 0; // Indicate error
        }

        context->downloaded += total_size;

        // Update status periodically (every second)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - context->last_update)
                           .count();

        if (elapsed >= 1)
        {
            context->status_callback(get_status(
                context->total_size,
                context->downloaded,
                context->last_downloaded,
                elapsed,
                context->message));
            context->last_update = now;
            context->last_downloaded = context->downloaded;
        }

        return total_size;
    }

    int FlorenceModelDownloader::progress_callback(void *clientp,
                                                   curl_off_t dltotal,
                                                   curl_off_t dlnow,
                                                   curl_off_t ultotal,
                                                   curl_off_t ulnow)
    {
        auto *context = static_cast<DownloadContext *>(clientp);
        if (dltotal > 0)
        {
            context->total_size = dltotal;
        }
        return 0;
    }

    // In the download_model function, add debug logging:
    bool FlorenceModelDownloader::download_model(
        Model model,
        std::function<void(const IStatus &)> on_status_update)
    {
        const std::string model_filename = get_model_filename(model);
        const std::string file_path = model_folder_base_path_ + "/" + model_filename;

        std::cout << "Attempting download for model: " << model_filename << std::endl;
        std::cout << "File path: " << file_path << std::endl;

        // Check if file already exists and is complete
        if (file_utils::is_file_complete(file_path))
        {
            std::cout << "File already exists and is complete" << std::endl;
            model_paths_[model] = file_path;
            return true;
        }

        // Try to create lock file
        std::cout << "Attempting to create lock file for: " << file_path << std::endl;
        if (!file_utils::create_lock_file(file_path))
        {
            std::cout << "Failed to create lock file, this shouldn't happen on first attempt" << std::endl;
            // Rest of the waiting code...
            std::cout << "Successfully created lock file" << std::endl;
            // Another process is downloading, wait and check periodically
            const int max_wait_seconds = 300; // 5 minutes
            const int check_interval = 5;     // 5 seconds
            int waited = 0;

            while (!file_utils::is_file_complete(file_path) && waited < max_wait_seconds)
            {
                std::this_thread::sleep_for(std::chrono::seconds(check_interval));
                waited += check_interval;
            }

            if (file_utils::is_file_complete(file_path))
            {
                model_paths_[model] = file_path;
                return true;
            }

            DownloadStatus error_status;
            error_status.set_error("Timeout waiting for another process to download model");
            error_status.set_progress(0.0f);
            on_status_update(error_status);
            return false;
        }

        // Initialize status
        DownloadStatus status;
        status.set_message("Downloading Florence2 Model " + std::to_string(static_cast<int>(model))+"\n");
        status.set_progress(0.0f);
        on_status_update(status);

        try
        {
            file_utils::ensure_directory_exists(file_path);

            // Setup download context
            DownloadContext context{
                std::ofstream(file_path, std::ios::binary),
                on_status_update,
                0,                                // total_size
                0,                                // downloaded
                0,                                // last_downloaded
                std::chrono::steady_clock::now(), // last_update
                status.get_message()};

            if (!context.file.is_open())
            {
                file_utils::remove_lock_file(file_path);
                throw std::runtime_error("Failed to create output file: " + file_path);
            }

            // Initialize CURL
            std::unique_ptr<CURL, void (*)(CURL *)> curl(curl_easy_init(), curl_easy_cleanup);
            if (!curl)
            {
                file_utils::remove_lock_file(file_path);
                throw std::runtime_error("Failed to initialize CURL");
            }

            // Setup CURL options
            std::string url = "https://huggingface.co/onnx-community/Florence-2-base/resolve/main/onnx/"+model_filename ;
            curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &context);
            curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, progress_callback);
            curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &context);
            curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);
            curl_easy_setopt(curl.get(), CURLOPT_BUFFERSIZE, 1024L * 1024L); // 1MB buffer

            // Add these options to help with large file transfers
            curl_easy_setopt(curl.get(), CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 0L);         // No timeout
            curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 30L); // 30 seconds connect timeout

            // Perform download with retry logic
            const int max_retries = 3;
            int retry_count = 0;
            bool download_success = false;
            CURLcode last_error = CURLE_OK;

            while (!download_success && retry_count < max_retries)
            {
                CURLcode res = curl_easy_perform(curl.get());
                last_error = res;

                if (res == CURLE_OK)
                {
                    // Verify downloaded size matches expected size
                    curl_off_t content_length;
                    res = curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
                    if (res == CURLE_OK && context.downloaded == content_length)
                    {
                        download_success = true;
                    }
                }

                if (!download_success)
                {
                    retry_count++;
                    if (retry_count < max_retries)
                    {
                        std::cout << "Download failed with error: "
                                  << curl_easy_strerror(last_error)
                                  << ", retrying in 5 seconds..." << std::endl;
                        std::this_thread::sleep_for(std::chrono::seconds(5));

                        // Setup resume from partial download
                        if (context.downloaded > 0)
                        {
                            context.file.close();
                            context.file.open(file_path, std::ios::binary | std::ios::app);
                            curl_easy_setopt(curl.get(), CURLOPT_RESUME_FROM_LARGE, context.downloaded);
                        }
                    }
                }
            }

            context.file.close();

            if (!download_success)
            {
                std::string error_msg = "Failed to download model: ";
                error_msg += curl_easy_strerror(last_error);
                std::filesystem::remove(file_path);
                file_utils::remove_lock_file(file_path);
                throw std::runtime_error(error_msg);
            }

            // Check file size one more time
            auto file_size = std::filesystem::file_size(file_path);
            if (file_size != context.total_size)
            {
                std::filesystem::remove(file_path);
                file_utils::remove_lock_file(file_path);
                throw std::runtime_error("Final file size mismatch");
            }

            // Success - remove lock file and add to model paths
            file_utils::remove_lock_file(file_path);
            model_paths_[model] = file_path;
            return true;
        }
        catch (const std::exception &e)
        {
            DownloadStatus error_status;
            error_status.set_error("Failed to download model: " + std::string(e.what()));
            error_status.set_progress(0.0f);
            on_status_update(error_status);

            if (std::filesystem::exists(file_path))
            {
                std::filesystem::remove(file_path);
            }
            file_utils::remove_lock_file(file_path);
            return false;
        }
    }

} // namespace florence2