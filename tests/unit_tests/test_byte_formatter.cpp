/*
 * C++ Port Changes from C# Testing:
 * 1. Using C++ assert instead of C# Assert
 * 2. Added more structured test organization with class
 * 3. Added explicit exception handling with try-catch
 * 4. Using std::cout for test progress output instead of C# Console
 * 5. Split tests into logical units with descriptive names
 * 6. Added comprehensive test cases not present in original C# code
 */
#include "helper/byte_formatter.h"
#include <iostream>
#include <cassert>
#include <string>

class ByteFormatterTest {
private:
    void test_zero() {
        std::cout << "Testing zero bytes..." << std::endl;
        auto result = florence2::ByteFormatter::format_bytes(0);
        assert(result == "0 B");
        std::cout << "Zero bytes: OK" << std::endl;
    }

    void test_simple_bytes() {
        std::cout << "\nTesting simple byte values..." << std::endl;
        // Testing basic conversions
        assert(florence2::ByteFormatter::format_bytes(500) == "500.0 B");
        assert(florence2::ByteFormatter::format_bytes(1024) == "1.0 KB");
        assert(florence2::ByteFormatter::format_bytes(1536) == "1.5 KB");
        std::cout << "Simple byte values: OK" << std::endl;
    }

    void test_negative_values() {
        std::cout << "\nTesting negative values..." << std::endl;
        // Added negative value testing
        assert(florence2::ByteFormatter::format_bytes(-1024) == "-1.0 KB");
        assert(florence2::ByteFormatter::format_bytes(-2048) == "-2.0 KB");
        std::cout << "Negative values: OK" << std::endl;
    }

    void test_large_values() {
        std::cout << "\nTesting large values..." << std::endl;
        // Testing larger units
        auto gb = 1024.0 * 1024.0 * 1024.0;
        assert(florence2::ByteFormatter::format_bytes(gb) == "1.0 GB");
        assert(florence2::ByteFormatter::format_bytes(gb * 1.5) == "1.5 GB");
        std::cout << "Large values: OK" << std::endl;
    }

    void test_precision() {
        std::cout << "\nTesting different precisions..." << std::endl;
        // Testing the digits parameter
        auto kb = 1024.0 * 1.5;
        assert(florence2::ByteFormatter::format_bytes(kb, 1) == "1.5 KB");
        assert(florence2::ByteFormatter::format_bytes(kb, 2) == "1.50 KB");
        assert(florence2::ByteFormatter::format_bytes(kb, 3) == "1.500 KB");
        std::cout << "Precision tests: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting ByteFormatter tests...\n" << std::endl;
        
        try {
            test_zero();
            test_simple_bytes();
            test_negative_values();
            test_large_values();
            test_precision();
            
            std::cout << "\nAll ByteFormatter tests passed successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Test failed with exception: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        ByteFormatterTest test_suite;
        test_suite.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}