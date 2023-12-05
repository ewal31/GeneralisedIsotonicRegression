#include "gir.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <emscripten/emscripten.h>
#include <Eigen/Core>
#include <Eigen/SparseCore> // # TODO should be able to delete
#include <generalized_isotonic_regression.h>

/*
 * Haven't used the csv parser here, as it seems to make the webassembly
 * binary quite large and doesn't seem to really want to work.
 */

uint32_t count_lines(const std::string& input) {
    return std::count_if(
            input.begin(), input.end(),
            [](const char c) { return c == '\n'; }
        ) + (input[input.size() - 1] == '\n' ? 0 : 1);
}

long parse_row(
    const std::string& input,
    std::vector<std::string_view>& buffer,
    long start_idx
) {
    long idx = start_idx;
    long stop_idx = idx - 1;

    while (idx < input.size() && input[idx] != '\n') {
        if (input[idx] == ' ' || input[idx] == '\t' || input[idx] == '\r') {
            if (idx  == start_idx) {
                start_idx = ++idx;
            } else { // TODO not handling spaces in between characters
                stop_idx = stop_idx > start_idx ? stop_idx : idx - 1;
            }
        }

        if (input[idx] == ',') {
            stop_idx = stop_idx > start_idx ? stop_idx : idx - 1;
            buffer.push_back(std::basic_string_view(&input[start_idx], stop_idx - start_idx + 1));
            start_idx = ++idx;
        } else {
            ++idx;
        }
    }

    stop_idx = stop_idx > start_idx ? stop_idx : idx - 1;
    buffer.push_back(std::basic_string_view(&input[start_idx], stop_idx - start_idx + 1));
    return idx + 1;
}

inline double parse_double(const std::string_view& str) {
    // TODO not yet implemented in stdlib :(
    // double parsed_double{};
    // std::from_chars(str.data(), str.data() + str.size(), parsed_double); // TODO no error checking
    // return parsed_double;
    return std::stod(std::string(str));
}

/*
 * e.g.
 * var input = "X_1, y\n0, 1\n1,2.2\n3,1.1";
 */

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
parse_input_data(const std::string& input) {

    const auto total_lines = count_lines(input);
    const auto total_values = total_lines - 1;

    long index = 0;
    std::vector<std::string_view> buffer;

    // Get column headings
    index = parse_row(input, buffer, index);

    uint32_t weight_column = std::numeric_limits<uint32_t>::max();
    uint32_t y_column = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> x_columns;

    for (size_t i = 0; i < buffer.size(); ++i) {
        if (buffer[i].rfind("X_", 0) == 0) {
            x_columns.push_back(i);
        } else if (buffer[i] == "y") {
            y_column = i;
        } else {
            weight_column = i;
        }
    }

    bool has_weights = weight_column < std::numeric_limits<uint32_t>::max();

    Eigen::MatrixXd X(total_values, x_columns.size());
    Eigen::VectorXd y(total_values);
    Eigen::VectorXd weight = Eigen::VectorXd::Ones(total_values);

    uint32_t row = 0;
    while (index < input.size()) {
        buffer.clear();
        index = parse_row(input, buffer, index);

        for (size_t col = 0; col < x_columns.size(); ++col) {
            X(row, col) = parse_double(buffer[x_columns[col]]);
        }

        y(row) = parse_double(buffer[y_column]);

        if (has_weights)
            weight(row) = parse_double(buffer[weight_column]);

        ++row;
    }

    return std::make_tuple(std::move(X), std::move(y), std::move(weight));
}

uint32_t count_digits(uint32_t number) {
    int length = 1;
    while ( number /= 10 )
        length++;
    return length;
}

std::string
format_output_data(gir::VectorXu groups, Eigen::VectorXd y_fit) {
    const uint32_t min_column_width = 5;
    const uint32_t precision = 6;

    double integral_part;
    std::modf(groups.maxCoeff(), &integral_part);
    const uint32_t group_width = std::max(
        min_column_width,
        count_digits(integral_part));

    std::modf(y_fit.maxCoeff(), &integral_part);
    const uint32_t y_width = std::max(
        min_column_width,
        precision + count_digits(integral_part));

    std::stringstream ss;
    ss << std::setw(group_width + 1)
       << "group,"
       << std::setw(y_width)
       << "y_fit"
       << std::fixed
       << std::setprecision(precision);

    for (size_t row = 0; row < groups.size(); ++row) {
        ss << '\n'
           << std::setw(group_width)
           << groups(row)
           << ","
           << std::setw(y_width)
           << y_fit(row);
    }

    return ss.str();
}

std::string
run_iso_regression(
    const std::string& input
) {
    const auto loss_function = gir::L2();

    const auto [X, y, weight] = parse_input_data(input);

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

    return format_output_data(groups(idx_original), y_fit(idx_original));
}
