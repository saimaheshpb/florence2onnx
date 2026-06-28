@echo off

REM Check if vcpkg is being used
if defined VCPKG_ROOT (
    echo Using vcpkg from %VCPKG_ROOT%
    set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
) else (
    echo Please set CMAKE_PREFIX_PATH to your dependency locations
    echo Example: set CMAKE_PREFIX_PATH=C:\path\to\curl;C:\path\to\spdlog
)

echo Current settings:
echo CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%
echo CMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%