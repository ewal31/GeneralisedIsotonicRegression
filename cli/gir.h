#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <utility>

#include <generalized_isotonic_regression.h>

uint64_t read_total_lines(const std::string& input_file);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
read_input_data(
    const std::string& input_file,
    const std::string& monotonicity_modifier
);

void write_result(
    const std::string& output_file,
    const Eigen::VectorXd& y_fit,
    const gir::VectorXu& group
);

template <typename LossType>
void run(
    const std::string& input_file,
    const std::string& output_file,
    const std::string& monotonicity_modifier,
    const gir::LossFunction<LossType>& loss_function
) {
    std::cout << "Reading Input" << std::endl;
    const auto [X, y, weight] = read_input_data(input_file, monotonicity_modifier);

    std::cout << "Building Adjacency Matrix" << std::endl;
    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    std::cout << "Running Isotonic Regression" << std::endl;

    auto [groups, y_fit] = generalised_isotonic_regression(
        adjacency_matrix,
        y(idx_new),
        weight(idx_new),
        loss_function);

    double total_loss = loss_function.loss(
        y(idx_new),
        y_fit,
        weight);
    std::cout << "Finished with total loss: " << total_loss << std::endl;

    std::cout << "Writing Result" << std::endl;

    // TODO test this is the right order
    write_result(output_file, y_fit(idx_original), groups(idx_original));

    std::cout << "Finished." << std::endl;
}
