set(EXECUTABLE_NAME gir_benchmarks)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Don't use Google Test")

FetchContent_Declare(
    GoogleBenchmark
    GIT_REPOSITORY "https://github.com/google/benchmark"
    # 1.8.3
    GIT_TAG 344117638c8ff7e239044fd0fa7085839fc03021
    # hash: sha256-DqGGfNjKPW9HFJrX9arFHyNYjB61uoL6NabZatTWrr0=
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    GoogleBenchmark
)

add_executable(
    ${EXECUTABLE_NAME}
    benchmark.cpp
)

target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    benchmark::benchmark
    isotonic_regression
)

target_compile_options(
    ${EXECUTABLE_NAME}
    PRIVATE
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

install(TARGETS ${EXECUTABLE_NAME})
