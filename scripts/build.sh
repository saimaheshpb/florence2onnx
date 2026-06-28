#!/bin/bash

# Create build directory
mkdir -p build && cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# Run tests (when we add them)
# ctest -C Release