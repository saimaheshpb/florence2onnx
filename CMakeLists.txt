# Add at the top of your CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(florence2-cpp VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Using FetchContent for JSON (common for both platforms)
include(FetchContent)
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Check if we're building for Android
if(ANDROID)
    # Android-specific configuration
    add_definitions(-DANDROID)

    # Android log library
    find_library(log-lib log)

    # Android-specific curl setup
    set(CURL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl-android/include)
    set(CURL_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl-android/lib/${ANDROID_ABI}/libcurl.so)

    # Android-specific OpenSSL setup
    set(OPENSSL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl-android/include)
    set(OPENSSL_LIBRARIES
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl-android/lib/${ANDROID_ABI}/libssl.a
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl-android/lib/${ANDROID_ABI}/libcrypto.a
    )

    # Set paths for ONNX Runtime
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android")
    endif()

    # Set OpenCV path for Android
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv-android/sdk/native/jni")
    find_package(OpenCV REQUIRED)

    # Set ONNX Runtime paths for Android
    set(ONNXRUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android/lib/${ANDROID_ABI}/libonnxruntime.so)
    set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android/include)
    set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android/include/onnxruntime/core/session)
else()
    # Desktop build configuration
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

    # Optional: Allow user to specify custom paths
    if(DEFINED CURL_ROOT)
        list(APPEND CMAKE_PREFIX_PATH ${CURL_ROOT})
    endif()

    if(DEFINED SPDLOG_ROOT)
        list(APPEND CMAKE_PREFIX_PATH ${SPDLOG_ROOT})
    endif()

    if(DEFINED ONNXRUNTIME_ROOT)
        list(APPEND CMAKE_PREFIX_PATH ${ONNXRUNTIME_ROOT})
    endif()

    # Find required packages for desktop
    find_package(CURL REQUIRED)
    find_package(spdlog REQUIRED)
    find_package(OpenCV REQUIRED)

    # Find ONNX Runtime for desktop
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        if(APPLE)
            set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")
        endif()
    endif()

    # Find ONNX Runtime libraries for desktop
    find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS ${ONNXRUNTIME_ROOT}/lib
            REQUIRED
    )

    find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATHS ${ONNXRUNTIME_ROOT}/include/onnxruntime
            REQUIRED
    )
endif()

# Platform-specific settings
if(WIN32)
    add_definitions(-DWINDOWS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
elseif(APPLE)
    add_definitions(-DMACOS)
elseif(UNIX AND NOT APPLE)
    add_definitions(-DLINUX)
endif()

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create library
if(ANDROID)
    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
else()
    add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
endif()

# Include directories
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${json_SOURCE_DIR}/include
        $<$<BOOL:${ANDROID}>:${CURL_INCLUDE_DIR}>
        $<$<BOOL:${ANDROID}>:${OPENSSL_INCLUDE_DIR}>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
if(ANDROID)
    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            ${ONNXRUNTIME_LIB}
            ${OpenCV_LIBS}
            ${log-lib}
            ${CURL_LIBRARY}
            ${OPENSSL_LIBRARIES}
            PRIVATE
            nlohmann_json::nlohmann_json
    )
else()
    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            CURL::libcurl
            spdlog::spdlog
            ${ONNXRUNTIME_LIB}
            ${OpenCV_LIBS}
            PRIVATE
            nlohmann_json::nlohmann_json
    )
endif()

# Enable testing only for desktop builds
if(NOT ANDROID)
    enable_testing()
    add_subdirectory(tests)
endif()