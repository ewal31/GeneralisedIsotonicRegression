#pragma once

#include <sstream>

#include <emscripten/bind.h>
#include <Eigen/Core>
#include <Eigen/SparseCore> // # TODO should be able to delete


uint32_t read_total_lines(const std::string& input_file);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
read_input_data(std::string input_file);

void run_iso_regression(std::stringstream& input);
int runtest(); // # TODO delete

EMSCRIPTEN_BINDINGS(EmbindVectorDouble) {
    emscripten::function("run_iso_regression", &run_iso_regression);
    emscripten::function("run_test", &runtest);

    emscripten::class_<std::stringstream>("StringStream")
        .constructor<std::string>()
        .property(
            "str",
            emscripten::select_overload<std::string(void)const>(&std::stringstream::str),
            emscripten::select_overload<void(const std::string&)>(&std::stringstream::str)
        );

    /*
    * For example:
    *
    * const list = new Module.VectorDouble();
    * list.push_back(8);
    * list.push_back(2);
    * Module.run_iso_regression(list);
    *
    */
    emscripten::register_vector<double>("VectorDouble");
}
