set(LIBRARY_NAME multivariate_isotonic_regression)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "https://github.com/pybind/pybind11"
    # v1.6.0
    GIT_TAG 8a099e44b3d5f85b20f05828d919d2332a8de841
    # hash: sha256-sO/Fa+QrAKyq2EYyYMcjPrYI+bdJIrDoj6L3JHoDo3E=
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    pybind11
)

pybind11_add_module(
    ${LIBRARY_NAME}
    lib.cpp
)

target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC
    isotonic_regression
)

target_compile_definitions(
    ${LIBRARY_NAME}
    PRIVATE
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
