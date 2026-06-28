#include "helper/resource_loader.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>

class ResourceLoaderTest {
private:
    // Helper to create test resource file
    void create_test_resource(const std::string& filename, const std::string& content) {
        std::filesystem::create_directories("test_resources");
        std::ofstream file("test_resources/" + filename);
        file << content;
    }

    void cleanup_test_resources() {
        std::filesystem::remove_all("test_resources");
    }

    void test_open_resource() {
        std::cout << "Testing resource opening..." << std::endl;

        const std::string test_content = "Test Resource Content";
        create_test_resource("test.txt", test_content);
        
        florence2::ResourceLoader::set_resource_path("test_resources");
        auto stream = florence2::ResourceLoader::open_resource("test.txt");
        
        std::string content;
        std::getline(*stream, content);
        
        assert(content == test_content);
        std::cout << "Resource opening: OK" << std::endl;
    }

    void test_get_resource() {
        std::cout << "\nTesting resource content retrieval..." << std::endl;

        const std::string test_content = "Binary Content Test";
        create_test_resource("binary.dat", test_content);
        
        auto bytes = florence2::ResourceLoader::get_resource("binary.dat");
        std::string retrieved_content(bytes.begin(), bytes.end());
        
        assert(retrieved_content == test_content);
        std::cout << "Resource content retrieval: OK" << std::endl;
    }

    void test_resource_not_found() {
        std::cout << "\nTesting resource not found handling..." << std::endl;

        bool exception_caught = false;
        try {
            florence2::ResourceLoader::get_resource("nonexistent.file");
        }
        catch (const std::runtime_error& e) {
            exception_caught = true;
        }
        
        assert(exception_caught);
        std::cout << "Resource not found handling: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting ResourceLoader tests...\n" << std::endl;
        
        try {
            test_open_resource();
            test_get_resource();
            test_resource_not_found();
            cleanup_test_resources();
            
            std::cout << "\nAll ResourceLoader tests passed successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Test failed with exception: " << e.what() << std::endl;
            cleanup_test_resources();
            throw;
        }
    }
};

int main() {
    try {
        ResourceLoaderTest test_suite;
        test_suite.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}