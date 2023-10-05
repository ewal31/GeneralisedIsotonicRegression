#include <filesystem>
#include <iostream>
#include <vector>
#include <utility>

#include <csv.hpp>
#include <cxxopts.hpp>
#include <generalized_isotonic_regression.h>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
read_input_data(std::string input_file) {
    csv::CSVFormat format;
    format
        .delimiter({'\t', ','})
        .trim({' '});

    csv::CSVReader reader(input_file, format);

    const auto& columns = reader.get_col_names();
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

    if (X_columns.size() == 0 || !has_y) {
        // TODO could be more precise with checking.
        std::cout << "Invalid CSV format. Expecting the following columns\n";
        std::cout << "1. X_1, X_2, ... X_n\n";
        std::cout << "2. y\n";
        std::cout << "3. (Optional) weight\n";
        exit(1);
    }

    // TODO probably best just to countlines then read directly into Eigen
    std::vector<std::vector<double>> X(X_columns.size());
    std::vector<double> y;
    std::vector<double> weight;

    for (csv::CSVRow& row : reader) {
        for (size_t i = 0; i < X_columns.size(); ++i) {
            X[i].push_back(row[X_columns[i]].get<double>());
        }
        y.push_back(row["y"].get<double>());
        weight.push_back(has_weights ? row["weight"].get<double>() : 1.);
    }

    Eigen::MatrixXd X_mat(X[0].size(), X.size());
    for (size_t i = 0; i < X_columns.size(); ++i) {
        X_mat(Eigen::all, i) =
            Eigen::Map<Eigen::VectorXd>(&X[i][0], X[i].size());
    }

    return std::make_tuple(
        X_mat,
        Eigen::Map<Eigen::VectorXd>(&y[0], y.size()),
        Eigen::Map<Eigen::VectorXd>(&weight[0], weight.size()));
}

void write_result(const std::string& output_file, const Eigen::VectorXd& y_fit) {
    std::ofstream ofstream(output_file, std::ofstream::out);
    // csv::DelimWriter<std::ofstream, ',', '"', false> writer(ofstream);
    // csv::set_decimal_places(12); // TODO configurable
    //for (double val : y_fit) ofstream <<  y_fit << '\n';
    // writer.flush();
    ofstream << y_fit;
    ofstream.flush();
    ofstream.close();
}

void
run(
    const std::string& input_file,
    const std::string& output_file,
    const gir::LossFunction loss)
{
    const auto [X, y, weight] = read_input_data(input_file);

    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight(idx_new),
            loss);

    write_result(output_file, y_fit(idx_original)); // TODO test this is the right order
}

int main(int argc, char** argv) {
    std::unordered_map<std::string, gir::LossFunction> loss_functions {
        {"L2", gir::LossFunction::L2},
        {"L2_WEIGHTED", gir::LossFunction::L2_WEIGHTED},
    };

    std::string loss_functions_string = "Loss Function [";
    std::for_each(loss_functions.begin(), loss_functions.end(),
        [&loss_functions_string, is_first = true](const auto& entry) mutable {
            if (is_first) {
                loss_functions_string += entry.first;
                is_first = false;
            } else {
                loss_functions_string += '|' + entry.first;
            }
        });
    loss_functions_string += "]";

    cxxopts::Options options(
        "gir",
        "This program runs isotonic regression on the provided input file with the specified loss function.");
    options
        .allow_unrecognised_options()
        .add_options()
        ("i,input", "Input File", cxxopts::value<std::string>())
        ("o,output", "Output File", cxxopts::value<std::string>())
        ("l,loss", loss_functions_string, cxxopts::value<std::string>()->default_value("L2"))
        ("h,help", "Print usage");

    std::vector<std::string> required_options {
        "input",
        "output"
    };

    auto parsed_options = options.parse(argc, argv);

    if (parsed_options.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (parsed_options.unmatched().size()) {
        std::cout << options.help() << std::endl;
        std::cout << "Unexpected options:";
        for (const auto& opt : parsed_options.unmatched())
            std::cout << " " << opt;
        std::cout << std::endl;
        exit(1);
    }

    required_options.erase(
        std::remove_if(required_options.begin(), required_options.end(),
            [&parsed_options](const std::string& option){
                return parsed_options.count(option);
            }),
        required_options.end());

    if (required_options.size()) {
        std::cout << options.help() << std::endl;
        std::cout << "Missing required options:";
        for (const auto& opt : required_options)
            std::cout << " " << opt;
        std::cout << std::endl;
        exit(1);
    }

    const auto& input_file = parsed_options["input"].as<std::string>();
    const auto& output_file = parsed_options["output"].as<std::string>();

    if (!std::filesystem::exists(std::filesystem::path(input_file))) {
        std::cout << "Input file '" << input_file;
        std::cout << "' not found." << std::endl;
        exit(1);
    }

    const auto& parsed_loss = parsed_options["loss"].as<std::string>();
    if (!loss_functions.count(parsed_loss)) {
        std::cout << options.help() << std::endl;
        std::cout << "Unrecognised loss function " << parsed_loss << std::endl;
        exit(1);
    }
    gir::LossFunction loss = loss_functions[parsed_loss];

    std::cout << "Running " << options.program();
    std::cout << " with loss function '" << parsed_loss << "' on\n";
    std::cout << ">> '" << input_file << "'\n";
    std::cout << "and writing result to \n";
    std::cout << ">> '" << output_file << "'" << std::endl;

    run(input_file, output_file, loss);

    return 0;
}
