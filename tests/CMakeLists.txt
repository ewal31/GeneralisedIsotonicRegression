FetchContent_Declare(
    Catch2
    GIT_REPOSITORY "https://github.com/catchorg/Catch2"
    # 3.4.0
    GIT_TAG 6e79e682b726f524310d55dec8ddac4e9c52fb5f
    # hash: sha256-DqGGfNjKPW9HFJrX9arFHyNYjB61uoL6NabZatTWrr0=
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    Catch2
)

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
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET test PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

install(TARGETS test)
