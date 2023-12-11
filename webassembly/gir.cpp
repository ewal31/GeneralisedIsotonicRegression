#include "gir.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <emscripten/emscripten.h>

HTMLElementBuffer::int_type HTMLElementBuffer::overflow(int_type c) {
    *pptr() = c;
    pbump(1);
    write_to_element();
    init();
    return c;
}

int HTMLElementBuffer::sync() {
    if (pptr() > &data[0]) {
        write_to_element();
    }
    init();
    return 0;
}

void HTMLElementBuffer::write_to_element() {
    *pptr() = 0; // treat as c string
    if (this->element_id == "") {
        EM_ASM_({
            console.log(Module.UTF8ToString($0));
        }, &data[0]);
    } else {
        EM_ASM_({
            var element = document.getElementById('console');
            element.value += Module.UTF8ToString($0);
            element.scrollTop = element.scrollHeight;
        }, &data[0]);
    }
}

void HTMLElementBuffer::init() {
    // leave extra space to add the overflow character
    // and null terminator as c string is required
    setp(&data[0], &data[0] + data.size() - 2);
}

HTMLElementBuffer::HTMLElementBuffer() {
    init();
}

void HTMLElementBuffer::set_element(const std::string& element_id) {
    this->element_id = element_id;
}

void update_element_buffer_element_id(const std::string& element_id) {
    element_buffer.set_element(element_id);
}

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
            if (idx == start_idx) {
                start_idx = ++idx;
            } else { // TODO not handling spaces in between characters
                stop_idx = stop_idx > start_idx ? stop_idx : idx - 1;
            }
        } else if (input[idx] == ',') {
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
format_input_data(
    Eigen::MatrixXd X,
    Eigen::VectorXd y
) {
    const uint32_t min_column_width = 5;
    const uint32_t precision = 2;
    std::vector<uint32_t> column_widths(X.cols());

    double integral_part;
    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        std::modf(X.col(col).maxCoeff(), &integral_part);
        column_widths[col] = std::max(
            min_column_width,
            1 + precision + count_digits(integral_part));
    }

    std::modf(y.maxCoeff(), &integral_part);
    const uint32_t y_width = std::max(
        min_column_width,
        // extra 1 for decimal point
        1 + precision + count_digits(integral_part));

    // std::modf(weights.maxCoeff(), &integral_part);
    // const uint32_t weights_width = std::max(
    //     7,
    //     // extra 1 for decimal point
    //     1 + precision + count_digits(integral_part));

    std::stringstream ss;
    for (Eigen::Index col = 0; col < X.cols(); ++col) {
        ss << std::setw(column_widths[col] - 1)
           << "X_"
           << col + 1
           << ", ";
    }

    ss << std::setw(y_width)
       << "y"
       // << std::setw(weights_width)
       // << "weights"
       << std::fixed
       << std::setprecision(precision);

    auto sorted_idxs = gir::argsort(X);
    Eigen::MatrixXd sorted_X = X(sorted_idxs, Eigen::all);
    Eigen::VectorXd sorted_y = y(sorted_idxs);

    for (size_t row = 0; row < X.rows(); ++row) {
        ss << '\n';

        for (Eigen::Index col = 0; col < X.cols(); ++col) {
            ss << std::setw(column_widths[col])
               << sorted_X(row, col)
               << ", ";
        }

        ss << std::setw(y_width)
           << sorted_y(row);
           // << ", "
           // << std::setw(weights_width)
           // << weights(row);
    }

    return ss.str();
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
        // extra 1 for decimal point
        1 + precision + count_digits(integral_part));

    std::stringstream ss;
    ss << std::setw(group_width + 2)
       << "group, "
       << std::setw(y_width)
       << "y_fit"
       << std::fixed
       << std::setprecision(precision);

    for (size_t row = 0; row < groups.size(); ++row) {
        ss << '\n'
           << std::setw(group_width)
           << groups(row)
           << ", "
           << std::setw(y_width)
           << y_fit(row);
    }

    return ss.str();
}

std::string generate_input_data(
    uint32_t total,
    uint32_t dimensions
) {
    const auto [X, y] = gir::generate_monotonic_points(
            total, 0.1, dimensions);
    return format_input_data(X, y);
}

gir_result
run_iso_regression(
    const std::string& loss_function,
    const std::string& input,
    const std::string& loss_parameter,
    const std::string& max_iterations
) {
    // This version will return 0 on failure, which is the same as run
    // until solution can't be improved anymore
    uint32_t parsed_max_iterations = (uint32_t) std::atoi(max_iterations.c_str());

    if (loss_function == "L2") {
        element_console << "Running with L2 Loss (Weighted)" << std::endl;
        return run_iso_regression_with_loss(
            gir::L2_WEIGHTED(),
            input,
            parsed_max_iterations);
    } else if (loss_function == "L1") {
        element_console << "Running with L1 Loss (Not-Weighted)" << std::endl;
        return run_iso_regression_with_loss(
            gir::L1(),
            input,
            parsed_max_iterations);
    } else if (loss_function == "HUBER") {
        const auto delta = std::stod(std::string(loss_parameter));
        element_console << "Running with Huber Loss delta=" << delta << " (Not-Weighted)" << std::endl;
        return run_iso_regression_with_loss(
            gir::HUBER(delta),
            input,
            parsed_max_iterations);
    } else {
        return gir_result("ERROR: Invalid Loss Function '" + loss_function + '\'');
    }

}
