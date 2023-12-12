set(EXECUTABLE_NAME girweb)

FetchContent_Declare(
    csv-parser
    # Repository has broken writing of floats that was never merged
    # GIT_REPOSITORY "https://github.com/vincentlaucsb/csv-parser"
    GIT_REPOSITORY "https://github.com/ahoarau/csv-parser"
    GIT_TAG 9a83b884b5a152e6151091a3bac5c4c11873be10
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    csv-parser
)

add_executable(
    ${EXECUTABLE_NAME}
    gir.cpp
)

target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    isotonic_regression
    csv
)

# csv library definition in CMakeLists.txt doesn't include header
target_include_directories(
    ${EXECUTABLE_NAME}
    SYSTEM
    PRIVATE
    ${csv-parser_SOURCE_DIR}/include
)

set(CMAKE_EXECUTABLE_SUFFIX ".js")

target_compile_options(
    ${EXECUTABLE_NAME}
    PRIVATE
    ${TARGET_COMPILE_OPTIONS}
    -sMAIN_MODULE=2
)

# -sMAIN_MODULE=2 TODO need to add more exports manually to use this but reduces size
set_target_properties(${EXECUTABLE_NAME} PROPERTIES LINK_FLAGS "-Os -sWASM=1 -sFILESYSTEM=0 -sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=2 -sEXPORTED_RUNTIME_METHODS=['UTF8ToString'] --bind --no-entry")
