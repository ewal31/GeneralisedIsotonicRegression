#include "gir.h"

#include <iostream>
#include <vector>

#include <csv.hpp>
#include <emscripten/emscripten.h>
#include <generalized_isotonic_regression.h>


uint32_t read_total_lines(std::stringstream& input) {
    uint32_t count = 0;
    while (input.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
        ++count;
    }
    input.clear(std::ios_base::eofbit);
    count = count - (input.unget().peek() == '\n');
    input.seekp(0, std::ios_base::end);
    input.seekg(0, std::ios_base::beg);
    return count;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
read_input_data(std::stringstream& input) {

    std::cout << "before" << std::endl;
    std::cout << "tellg " << input.tellg() << "\n";
    std::cout << "tellp " << input.tellp() << "\n";
    std::cout << "good " << input.good() << "\n";
    std::cout << "eof " << input.eof() << "\n";
    std::cout << "fail " << input.fail() << "\n";
    std::cout << "bad " << input.bad() << "\n";
    std::cout << "rdstate " << input.rdstate() << "\n";
    std::cout << "contents " << input.str() << "\n" << std::endl;

    const auto total_lines = read_total_lines(input);
    const auto total_values = total_lines - 1;

    std::cout << "after" << std::endl;
    std::cout << "tellg " << input.tellg() << "\n";
    std::cout << "tellp " << input.tellp() << "\n";
    std::cout << "good " << input.good() << "\n";
    std::cout << "eof " << input.eof() << "\n";
    std::cout << "fail " << input.fail() << "\n";
    std::cout << "bad " << input.bad() << "\n";
    std::cout << "rdstate " << input.rdstate() << "\n";
    std::cout << "contents " << input.str() << "\n" << std::endl;

    csv::CSVFormat format;
    format
        .delimiter(',')
        .header_row(0)
        .trim({' '});

    csv::CSVReader reader = csv::parse(std::string(input.str()), format);

    const auto& columns = reader.get_col_names();

    std::cout << "found columns: ";
    for (const auto& v : columns) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    const bool has_weights =
        std::find(columns.begin(), columns.end(), "weight") != columns.end();
    const bool has_y =
        std::find(columns.begin(), columns.end(), "y") != columns.end();

    std::vector<std::string> X_columns;
    std::copy_if(
        columns.begin(),
        columns.end(),
        std::back_inserter(X_columns),
        [](const auto& column){
            return column.rfind("X_", 0) == 0;
        });

    if (X_columns.size() == 0 || !has_y || total_lines == 1) {
        std::cout << "Invalid or Empty CSV File. ";
        std::cout << "Expecting the following columns\n";
        std::cout << "1. X_1, X_2, ... X_n\n";
        std::cout << "2. y\n";
        std::cout << "3. (Optional) weight\n";
        std::cout << "and a least one value.";
        exit(1);
    } else {
        std::cout << ">> Found '" << X_columns.size() << "' X columns, with '";
        std::cout << total_lines - 1 << "' values";
        if (has_weights)
            std::cout << " and weights." << std::endl;
        else
            std::cout << ", but no weights." << std::endl;
    }

    Eigen::MatrixXd X(total_values, X_columns.size());
    Eigen::VectorXd y(total_values);
    Eigen::VectorXd weight = Eigen::VectorXd::Ones(total_values);

    uint64_t row = 0;
    for (csv::CSVRow& file_row : reader) {
        for (size_t col = 0; col < X_columns.size(); ++col) {
            X(row, col) = file_row[X_columns[col]].get<double>();
        }
        y(row) = file_row["y"].get<double>();
        if (has_weights)
            weight(row) = file_row["weight"].get<double>();
        ++row;
    }

    return std::make_tuple(std::move(X), std::move(y), std::move(weight));
}

void run_iso_regression(
    std::stringstream& input
) {
    const auto loss = gir::L2();

    std::cout << "Found a total of: " << read_total_lines(input) << " lines." << std::endl;

    const auto [X, y, weight] = read_input_data(input);

    std::cout << "X:\n" << X << std::endl;
    std::cout << "y:\n" << X << std::endl;
    std::cout << "weights:\n" << X << std::endl;
}

int runtest() {
    auto [X, y] = gir::generate_monotonic_points(10, 1e-2, 2);
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

    std::cout << "point positions:\n" << X << '\n' << std::endl;
    std::cout << "monotonic axis:\n" << y << '\n' << std::endl;

    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    std::cout << "adjacency matrix:\n" << adjacency_matrix << '\n' << std::endl;
    std::cout << "adjacency matrix ordering:\n" << idx_new << '\n' << std::endl;
    std::cout << "points reordered\n";
    std::cout << X(idx_new, Eigen::all) << "\n" << std::endl;
    std::cout << y(idx_new) << "\n" << std::endl;

    gir::L2 loss_function;

    auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weights,
            loss_function);

    std::cout << "fit y values:\n" << y_fit << "\n" << std::endl;
    std::cout << "into groups:\n" << groups << "\n" << std::endl;

    double total_loss = loss_function.loss(
        y(idx_new),
        y_fit,
        weights);

    std::cout << "with total loss: " << total_loss << "\n" << std::endl;

    return 0;
}
