# Collect all test files
file(GLOB_RECURSE TEST_SOURCES "unit_tests/*.cpp")

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Main project target
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the filename without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Add include directories - make sure the paths are correct
    target_include_directories(${TEST_NAME} 
        PUBLIC 
            ${CMAKE_SOURCE_DIR}/include      # Main project includes
            ${ONNXRUNTIME_INCLUDE_DIR}       # ONNX Runtime includes
            /opt/homebrew/include/onnxruntime  # Fallback for Mac
    )
    
    # Link against main library and dependencies
    target_link_libraries(${TEST_NAME} 
        PRIVATE 
            ${PROJECT_NAME}
            ${ONNXRUNTIME_LIB}
            CURL::libcurl
            spdlog::spdlog
            nlohmann_json::nlohmann_json
            ${OpenCV_LIBS}
    )
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Print some debug info
    message(STATUS "Test ${TEST_NAME}:")
    message(STATUS "  Source: ${TEST_SOURCE}")
    message(STATUS "  Include dirs: ${CMAKE_SOURCE_DIR}/include")
endforeach()