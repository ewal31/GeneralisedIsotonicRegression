#pragma once

#include <emscripten/bind.h>


std::string run_iso_regression(const std::string& input);

EMSCRIPTEN_BINDINGS(EmbindVectorDouble) {
    emscripten::function("run_iso_regression", &run_iso_regression);

    // emscripten::class_<std::stringstream>("StringStream")
    //     .constructor<std::string>()
    //     .property(
    //         "str",
    //         emscripten::select_overload<std::string(void)const>(&std::stringstream::str),
    //         emscripten::select_overload<void(const std::string&)>(&std::stringstream::str)
    //     );

    /*
    * For example:
    *
    * const list = new Module.VectorDouble();
    * list.push_back(8);
    * list.push_back(2);
    * Module.run_iso_regression(list);
    *
    */
    // emscripten::register_vector<double>("VectorDouble");
}
