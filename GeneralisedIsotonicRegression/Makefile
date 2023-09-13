APP_SRCS := $(wildcard example/*)
ISO_SRCS1 := $(wildcard isotonic_regression/*)
ISO_SRCS2 := $(wildcard isotonic_regression/*/*)
TEST_SRCS := $(wildcard tests/*)

build/example/main: CMakeLists.txt $(APP_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	#cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	make -C build

run: build/example/main
	./build/example/main

build/tests/test: CMakeLists.txt $(TEST_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	#cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	make -C build

test: build/tests/test
	./build/tests/test

lspsymbols:
	cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES

clean:
	rm -rf build
	rm -rf debug
