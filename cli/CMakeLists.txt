set(EXECUTABLE_NAME gir)

FetchContent_Declare(
    csv-parser
    # Repository has broken writing of floats that was never merged
    # GIT_REPOSITORY "https://github.com/vincentlaucsb/csv-parser"
    GIT_REPOSITORY "https://github.com/ahoarau/csv-parser"
    GIT_TAG 9a83b884b5a152e6151091a3bac5c4c11873be10
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY "https://github.com/jarro2783/cxxopts"
    GIT_TAG v3.1.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    csv-parser
    cxxopts
)

add_executable(
    ${EXECUTABLE_NAME}
    gir.cpp
)

target_include_directories(
    ${EXECUTABLE_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# csv library definition in CMakeLists.txt doesn't include header
target_include_directories(
    ${EXECUTABLE_NAME}
    SYSTEM
    PRIVATE
    ${csv-parser_SOURCE_DIR}/include
)

target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    isotonic_regression
    csv
    cxxopts::cxxopts
)

target_compile_options(
    ${EXECUTABLE_NAME}
    PRIVATE
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
