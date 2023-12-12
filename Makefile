CLI_SRCS := $(wildcard cli/*)
EXAMPLE_SRCS := $(wildcard example/*)
ISO_SRCS1 := $(wildcard isotonic_regression/*)
ISO_SRCS2 := $(wildcard isotonic_regression/*/*)
TEST_SRCS := $(wildcard tests/*)
WEBASSEMBLY_SRCS := $(wildcard webassembly/*)

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

webbuild/webassembly/girweb.wasm: CMakeLists.txt $(WEBASSEMBLY_SRCS) $(ISO_SRCS1) $(ISO_SRCS2)
	$(if $(shell command -v emcmake 2> /dev/null), $(info Found `emcmake`),$(error Please obtain a copy of emsdk and source `emsdk_env.sh`))
	emcmake cmake . -Bwebbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	emmake make -C webbuild

wasm: webbuild/webassembly/girweb.wasm
	$(if $(shell command -v emcmake 2> /dev/null), $(info Found `emrun`),$(error Please obtain a copy of emsdk and source `emsdk_env.sh`))
	echo "$$INDEX_HTML" > webbuild/webassembly/index.html
	emrun webbuild/webassembly/index.html

lspsymbols:
	cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DBUILD_GIR_CLI_TOOL=ON -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON

lspsymbolsweb:
	$(if $(shell command -v emcmake 2> /dev/null), $(info Found `emrun`),$(error Please obtain a copy of emsdk and source `emsdk_env.sh`))
	emcmake cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES

all:
	cmake . -Bbuild -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_GIR_CLI_TOOL=ON -DBUILD_GIR_EXAMPLES=ON -DBUILD_GIR_TESTS=ON
	make -C build

clean:
	rm -rf build
	rm -rf webbuild
	rm -rf debug

export INDEX_HTML
define INDEX_HTML
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>GIR TEST</title>
  </head>
  <body>
    <h1>Generalised Isotonic Regression</h1>
    <form id="regression-form">

        <label for="iterations">Max Iterations:</label>
        <input type="text" id="iterations" name="iterations" value="0"><br>

        <label for="loss-function">Choose a Loss Function:</label>
        <select name="loss-function" id="loss-function">
          <option value="L2">L2 Weighted</option>
          <option value="L1">L1</option>
          <option value="HUBER">Huber</option>
          <option value="NotImplemented">NotImplemented</option>
        </select><br>

        <textarea id="input" name="input" rows="20" style="width: 49%;">X_1,   y
  0,   1
  1, 2.2
  3, 1.1
  4, 1.1
 10,   5</textarea>
        <textarea id="output" name="output" rows="20" style="width: 49%;" readonly></textarea>
        <br>
        <button>Run Regression</button>
    </form>
    <textarea id="console" name="console" rows="20" style="width: 49%;" readonly></textarea>
    <script src="girweb.js" id="girweb-js"></script>
    <script type="text/javascript">
        var result
        function validate(e){
            result && result.delete()
            Module.set_console_element("console")
            var input = document.getElementById('input')
            var loss = document.getElementById('loss-function')
            var iterations = document.getElementById('iterations')
            var output = document.getElementById('output')
            result = Module.run_iso_regression(loss.value, input.value, "0.1", iterations.value)
            output.value = result.get_formatted(Math.max(result.iterations - 1, 0))
            return false;
        }

        function init(){
            document.getElementById('regression-form').onsubmit = validate;
        }

        window.onload = init;
    </script>
  </body>
</html>
endef
