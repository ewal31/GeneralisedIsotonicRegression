APP_SRCS := $(wildcard example/*)
ISO_SRCS := $(wildcard isotonic_regression/*/*)
TEST_SRCS := $(wildcard tests/*)

build/example/main: CMakeLists.txt $(APP_SRCS) $(ISO_SRCS)
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release
	make -C build

run: build/example/main
	./build/example/main

build/tests/test: CMakeLists.txt $(ISO_SRCS) $(TEST_SRCS)
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release
	make -C build

test: build/tests/test
	./build/tests/test

lspsymbols:
	cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES

clean:
	rm -rf build
	rm -rf debug
