set(EXECUTABLE_NAME gir)

FetchContent_Declare(
    csv-parser
    GIT_REPOSITORY "https://github.com/vincentlaucsb/csv-parser"
    GIT_TAG 2.1.3
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
