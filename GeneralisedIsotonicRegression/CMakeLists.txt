cmake_minimum_required(VERSION 3.16)

project(
    IsotonicRegression
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    TARGET_COMPILE_OPTIONS
    # -march=native
    # -O3
    # -Wall
    # -Wextra
    # -Wpedantic
    # -fsanitize=address
)

include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

# FetchContent_Declare(
#     fmt
#     GIT_REPOSITORY "https://github.com/fmtlib/fmt"
#     GIT_TAG 10.0.0
#     GIT_SHALLOW TRUE
# )

FetchContent_MakeAvailable(
    Eigen3
    # fmt
)

# Just Clang and GNU
# add_compile_options("-fno-omit-frame-pointer")
# add_link_options("-fno-omit-frame-pointer")
#
# add_compile_options("-fsanitize=address")
# add_link_options("-fsanitize=address")
#
# add_compile_options("-fsanitize=undefined")
# add_link_options("-fsanitize=undefined")

add_subdirectory(isotonic_regression)

option(BUILD_GIR_CLI_TOOL "Build CLI Tool" OFF)
option(BUILD_GIR_EXAMPLES "Build Examples" OFF)
option(BUILD_GIR_TESTS    "Build Tests" OFF)

message(STATUS "Build GIR CLI Tool: ${BUILD_GIR_CLI_TOOL}")
message(STATUS "Build GIR Examples: ${BUILD_GIR_EXAMPLES}")
message(STATUS "Build GIR Tests: ${BUILD_GIR_TESTS}")

if (BUILD_GIR_CLI_TOOL)
    message(STATUS "Building CLI Tool")
    add_subdirectory(cli)
endif (BUILD_GIR_CLI_TOOL)

if (BUILD_GIR_EXAMPLES)
    message(STATUS "Building Examples")
    add_subdirectory(example)
endif (BUILD_GIR_EXAMPLES)

if (BUILD_GIR_TESTS)
    message(STATUS "Building Tests")
    add_subdirectory(tests)
endif (BUILD_GIR_TESTS)
