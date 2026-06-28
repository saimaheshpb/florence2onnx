#pragma once

#include <string>
#include <filesystem>
#include <fstream>

namespace florence2
{
    namespace file_utils
    {

        inline bool create_lock_file(const std::string &path)
        {
            try
            {
                std::string lock_path = path + ".lock";
                std::cout << "Creating lock file at: " << lock_path << std::endl;

                // First check if lock file exists
                if (std::filesystem::exists(lock_path))
                {
                    std::cout << "Lock file already exists" << std::endl;
                    return false;
                }

                // Ensure directory exists
                std::filesystem::create_directories(
                    std::filesystem::path(lock_path).parent_path());

                std::ofstream lock_file(lock_path);
                bool success = lock_file.good();
                std::cout << "Lock file creation " << (success ? "succeeded" : "failed") << std::endl;
                return success;
            }
            catch (const std::exception &e)
            {
                std::cout << "Exception creating lock file: " << e.what() << std::endl;
                return false;
            }
        }
        inline bool remove_lock_file(const std::string &path)
        {
            try
            {
                return std::filesystem::remove(path + ".lock");
            }
            catch (...)
            {
                return false;
            }
        }

        inline bool is_file_complete(const std::string &path)
        {
            try
            {
                return std::filesystem::exists(path) &&
                       std::filesystem::file_size(path) > 0 &&
                       !std::filesystem::exists(path + ".lock");
            }
            catch (...)
            {
                return false;
            }
        }

        inline void ensure_directory_exists(const std::string &path)
        {
            std::filesystem::create_directories(
                std::filesystem::path(path).parent_path());
        }

    } // namespace file_utils
} // namespace florence2