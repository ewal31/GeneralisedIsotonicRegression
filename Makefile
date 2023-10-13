CLI_SRCS := $(wildcard cli/*)
EXAMPLE_SRCS := $(wildcard example/*)
ISO_SRCS1 := $(wildcard isotonic_regression/*)
ISO_SRCS2 := $(wildcard isotonic_regression/*/*)
TEST_SRCS := $(wildcard tests/*)

BUILD_TYPE := Release # RelWithDebInfo

build/cli/gir: CMakeLists.txt $(CLI_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_GIR_CLI_TOOL=ON -DBUILD_GIR_EXAMPLES=OFF -DBUILD_GIR_TESTS=OFF
	make -C build

cli: build/cli/gir

build/example/main: CMakeLists.txt $(EXAMPLE_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_GIR_CLI_TOOL=OFF -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=OFF
	make -C build

run: build/example/main
	./build/example/main

build/tests/test: CMakeLists.txt $(TEST_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_GIR_CLI_TOOL=OFF -DBUILD_GIR_EXAMPLES=OFF -DBUILD_GIR_TESTS=ON
	make -C build

test: build/tests/test
	./build/tests/test

lspsymbols:
	cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DBUILD_GIR_CLI_TOOL=ON -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON

all:
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_GIR_CLI_TOOL=ON -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	make -C build

clean:
	rm -rf build
	rm -rf debug
