set(EXECUTABLE_NAME example)

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
    ${TARGET_COMPILE_OPTIONS}
)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

install(TARGETS ${EXECUTABLE_NAME})
