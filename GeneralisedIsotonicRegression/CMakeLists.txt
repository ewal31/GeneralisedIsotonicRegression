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
    CLANG_COMPILER
    -Wall
    -Wextra
    -Wpedantic
)

set(
    TARGET_COMPILE_OPTIONS
    -march=native
)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY "https://github.com/catchorg/Catch2"
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY "https://github.com/fmtlib/fmt"
    GIT_TAG 10.0.0
    GIT_SHALLOW TRUE
)

set(FAST_BUILD ON CACHE INTERNAL "Fast Build")
set(BUILD_EXAMPLES OFF CACHE INTERNAL "Build examples")

FetchContent_Declare(
    HIGHS
    GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS"
    GIT_TAG v1.5.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    Catch2
    Eigen3
    fmt
    HIGHS
)

add_subdirectory(isotonic_regression)

if (BUILD_GIR_EXAMPLES)
    add_subdirectory(example)
endif (BUILD_GIR_EXAMPLES)

if (BUILD_GIR_TESTS)
    add_subdirectory(tests)
endif (BUILD_GIR_TESTS)

# Just Clang and GNU
add_compile_options("-fno-omit-frame-pointer")
add_link_options("-fno-omit-frame-pointer")

add_compile_options("-fsanitize=address")
add_link_options("-fsanitize=address")

add_compile_options("-fsanitize=undefined")
add_link_options("-fsanitize=undefined")
