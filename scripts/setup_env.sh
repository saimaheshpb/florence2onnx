#!/bin/bash

# Detect OS
case "$(uname -s)" in
    Darwin)  # macOS
        if [ -d "/opt/homebrew/opt" ]; then
            # M1 Mac
            export CMAKE_PREFIX_PATH="/opt/homebrew/opt/curl:/opt/homebrew/opt/spdlog"
        else
            # Intel Mac
            export CMAKE_PREFIX_PATH="/usr/local/opt/curl:/usr/local/opt/spdlog"
        fi
        echo "Set up environment for macOS"
        ;;
    Linux)
        # Most Linux distributions install to standard locations
        # Usually no need to set CMAKE_PREFIX_PATH
        echo "Linux detected - using standard paths"
        ;;
    MINGW*|MSYS*|CYGWIN*)  # Windows
        echo "Please set CMAKE_PREFIX_PATH manually to your dependency locations"
        echo "Example: set CMAKE_PREFIX_PATH=C:\path\to\curl;C:\path\to\spdlog"
        ;;
    *)
        echo "Unknown operating system"
        ;;
esac

# Print current settings
echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"