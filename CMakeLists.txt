cmake_minimum_required(VERSION 3.16)

project(
    IsotonicRegression
    VERSION 0.3.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    TARGET_COMPILE_OPTIONS
    # -march=native
    # -Wall
    # -Wpedantic
)

add_compile_options(
    # -fsanitize=address
    # -fsanitize=undefined
    # -fno-omit-frame-pointer
    # -fsanitize=leak
)

add_link_options(
    # -fsanitize=address
    # -fsanitize=undefined
    # -fno-omit-frame-pointer
    # -fsanitize=leak
)

include(FetchContent)

function(FetchContent_MakeAvailable2 pkgs) # until cmake 3.28
    foreach(pkg ${ARGV})
        string(TOLOWER ${pkg} pkgLower)
        FetchContent_GetProperties(${pkg})

        if(NOT "${${pkgLower}_POPULATED}")
            FetchContent_Populate(${pkg})
            add_subdirectory("${${pkgLower}_SOURCE_DIR}" "${${pkgLower}_BINARY_DIR}" EXCLUDE_FROM_ALL)
        endif()
    endforeach()
endfunction()

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
    # 3.4.0
    GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c
    # hash: sha256-1/4xMetKMDOgZgzz3WMxfHUEpmdAm52RqZvz6i0mLEw=
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(
    Eigen3
)

if (EMSCRIPTEN)
    message(STATUS "This is an Emscripten build.")
    message(STATUS "Other possible targets will be ignored.")

    set(
        TARGET_COMPILE_OPTIONS
        ${TARGET_COMPILE_OPTIONS}
        -Os
    )

    add_subdirectory(isotonic_regression)
    add_subdirectory(webassembly)

else ()

    option(BUILD_GIR_CLI_TOOL "Build CLI Tool" OFF)
    option(BUILD_GIR_EXAMPLES "Build Examples" OFF)
    option(BUILD_GIR_TESTS    "Build Tests" OFF)
    option(BUILD_GIR_PYTHON   "Build Python Interface" OFF)

    message(STATUS "Build GIR CLI Tool: ${BUILD_GIR_CLI_TOOL}")
    message(STATUS "Build GIR Examples: ${BUILD_GIR_EXAMPLES}")
    message(STATUS "Build GIR Tests: ${BUILD_GIR_TESTS}")
    message(STATUS "Build GIR Benchmarks: ${BUILD_GIR_BENCHMARKS}")
    message(STATUS "Build GIR Python Interface: ${BUILD_GIR_PYTHON}")

    set(
        TARGET_COMPILE_OPTIONS
        ${TARGET_COMPILE_OPTIONS}
        -O3
    )

    add_subdirectory(isotonic_regression)

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

    if (BUILD_GIR_BENCHMARKS)
        message(STATUS "Building Benchmarks")
        add_subdirectory(benchmarks)
    endif (BUILD_GIR_BENCHMARKS)

    if (BUILD_GIR_PYTHON)
        message(STATUS "Building Python Interface")
        add_subdirectory(python)
    endif (BUILD_GIR_PYTHON)

endif ()
