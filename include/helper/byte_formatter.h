#pragma once
#include <string>
#include <cmath>
#include <array>
#include <sstream>
#include <iomanip>

namespace florence2 {

/*
 * C++ Port Changes from C#:
 * 1. Changed from static class to class with static method
 * 2. Renamed from 'FormatBytes' to 'format_bytes' following C++ naming convention
 * 3. Used std::array instead of C# array for units
 * 4. Used std::ostringstream for string formatting instead of C# string formatting
 * 5. Used std::fixed and std::setprecision for decimal precision control
 * 6. Changed string concatenation to stream operations
 */
class ByteFormatter {
public:
    static std::string format_bytes(double bytes, int digits = 1) {
        // Static array instead of C#'s new[] initialization
        static const std::array<std::string, 9> units = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
        
        if (bytes == 0) return "0 B";
        
        bool is_negative = bytes < 0;
        if (is_negative) bytes = -bytes;
        
        const int c = 1024;
        // Combined Math.Floor and Math.Log operations with std::floor and std::log
        int power = std::min(
            static_cast<int>(std::floor(std::log(bytes) / std::log(c))), 
            static_cast<int>(units.size() - 1)
        );
        
        // Using stringstream instead of C#'s string formatting and concatenation
        std::ostringstream stream;
        if (is_negative) stream << "-";
        
        // std::fixed and std::setprecision replace C#'s "0." + new string('#', digits)
        stream << std::fixed << std::setprecision(digits) 
               << (bytes / std::pow(c, power)) << " " << units[power];
        
        return stream.str();
    }
};

} // namespace florence2