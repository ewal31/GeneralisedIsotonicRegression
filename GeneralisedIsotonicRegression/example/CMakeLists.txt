set(EXECUTABLE_NAME main)

add_executable(
    ${EXECUTABLE_NAME}
    main.cpp
)

target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    isotonic_regression
)

target_compile_options(
    ${EXECUTABLE_NAME}
    PRIVATE
    -march=native
)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
