if (EMSCRIPTEN)
    # 64bit might work if we add -sWASM_BIGINT
    set(HIGHSINT64 OFF CACHE INTERNAL "Use 32 bit integers indexing")

else ()
    set(HIGHSINT64 ON CACHE INTERNAL "Use 64 bit integers indexing")

endif ()


set(FAST_BUILD ON CACHE INTERNAL "Fast Build")
set(BUILD_EXAMPLES OFF CACHE INTERNAL "Build examples")

FetchContent_Declare(
    HIGHS
    GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS"
    GIT_TAG v1.5.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    HIGHS
)

add_library(isotonic_regression STATIC)

target_sources(
    isotonic_regression
    PRIVATE
    src/generalized_isotonic_regression.cpp
    src/loss.cpp
    src/utility.cpp
)

target_include_directories(
    isotonic_regression
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
    isotonic_regression
    PUBLIC
    Eigen3::Eigen
)

target_link_libraries(
    isotonic_regression
    PRIVATE
    highs::highs
)

target_compile_options(
    isotonic_regression
    PRIVATE
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET isotonic_regression PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
