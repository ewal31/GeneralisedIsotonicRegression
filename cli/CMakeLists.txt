set(EXECUTABLE_NAME gir)

FetchContent_Declare(
    csv-parser
    # Repository has broken writing of floats that was never merged
    # GIT_REPOSITORY "https://github.com/vincentlaucsb/csv-parser"
    GIT_REPOSITORY "https://github.com/ahoarau/csv-parser"
    GIT_TAG 9a83b884b5a152e6151091a3bac5c4c11873be10
    # hash: sha256-QyY9zVzXq4a0Mgn/erVim67aIjdGf1Dy9DRySjrjojk=
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(
    csv-parser
)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY "https://github.com/jarro2783/cxxopts"
    # v3.1.1
    GIT_TAG eb787304d67ec22f7c3a184ee8b4c481d04357fd
    # hash: sha256-lJPMaXBfrCeUhhXha5f7zmOGtyEDzU3oPTMirPTFZzQ=
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(
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

install(TARGETS ${EXECUTABLE_NAME})
