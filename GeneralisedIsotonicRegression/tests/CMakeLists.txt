add_executable(
    test
    test_generalized_isotonic_regression.cpp
    test_utility.cpp
)

target_link_libraries(
    test
    PRIVATE
    Catch2::Catch2WithMain
    isotonic_regression
)

target_compile_options(
    test
    PRIVATE
    -march=native
)

set_property(TARGET test PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
