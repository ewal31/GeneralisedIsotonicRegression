add_library(isotonic_regression STATIC)

target_sources(
    isotonic_regression
    PRIVATE
    src/generalized_isotonic_regression.cpp
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
    fmt::fmt
    highs::highs
)

target_compile_options(
    isotonic_regression
    PRIVATE
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET isotonic_regression PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
