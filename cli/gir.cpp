#include "gir.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <vector>

#include <csv.hpp>
#include <cxxopts.hpp>

uint64_t read_total_lines(const std::string& input_file) {
    uint64_t count = 0;
    std::ifstream f_stream(input_file, std::ios::binary);
    while (f_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n')) {
        ++count;
    }
    f_stream.clear(std::ios_base::eofbit);
    return count - (f_stream.unget().peek() == '\n');
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
read_input_data(
    const std::string& input_file,
    const std::string& monotonicity_modifier
) {
    const auto total_lines = read_total_lines(input_file);
    const auto total_values = total_lines - 1;

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

    std::vector<int> monotonicity_mult(X_columns.size(), 1);

    if (monotonicity_modifier.size() > 0) {
        size_t pos = 0;
        for (int i = 0; i < X_columns.size(); ++i) {
            try {
                if (monotonicity_modifier[pos] == ',') {
                    ++pos;
                }

                if (pos >= monotonicity_modifier.size()) {
                    std::cout << "Invalid Monotonicity Direction Argument: -m '";
                    std::cout << monotonicity_modifier << "'\n";
                    std::cout << "Not enough arguments for columns in :";
                    std::cout << input_file << std::endl;
                    exit(1);
                }

                size_t pos_adj;
                int parsed = std::stoi(monotonicity_modifier.substr(pos), &pos_adj);
                pos += pos_adj;

                if (parsed == 1 || parsed == -1) {
                    monotonicity_mult[i] = parsed;
                } else {
                    std::cout << "Invalid Monotonicity Direction Argument: -m '";
                    std::cout << monotonicity_modifier << "'\n";
                    std::cout << "Expecting arguments of the form '-m -1,1,1,-1'\n";
                    std::cout << "Found '" << parsed << '\'' << std::endl;
                    exit(1);
                }
            }
            catch (const std::exception& e) {
                std::cout << "Invalid Monotonicity Direction Argument: -m '";
                std::cout << monotonicity_modifier << "'\n";
                std::cout << "Expecting arguments of the form '-m -1,1,1,-1'" << std::endl;
                exit(1);
            }
        }

        if (pos < monotonicity_modifier.size()) {
            std::cout << "Invalid Monotonicity Direction Argument: -m '";
            std::cout << monotonicity_modifier << "'\n";
            std::cout << ": contains more arguments than columns in ";
            std::cout << input_file << std::endl;
            exit(1);
        }
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

    for (int col = 0; col < X.cols(); ++col) {
        X(Eigen::all, col) *= monotonicity_mult[col];
    }

    return std::make_tuple(std::move(X), std::move(y), std::move(weight));
}

void write_result(
    const std::string& output_file,
    const Eigen::VectorXd& y_fit,
    const gir::VectorXu& group
) {
    std::ofstream ofstream(output_file, std::ofstream::out);
    csv::DelimWriter<std::ofstream, ',', '"', false> writer(ofstream);
    csv::set_decimal_places(12); // TODO configurable

    writer << std::make_tuple("y_fit", "group");
    for (Eigen::Index i = 0; i < y_fit.rows(); ++i)
        writer << std::make_tuple(y_fit(i), group(i));

    writer.flush();
    ofstream.flush();
    ofstream.close();
}

int main(int argc, char** argv) {
    std::vector<std::string> loss_functions {
        "L1", "L2", "L2_WEIGHTED", "HUBER"
    };

    std::string loss_functions_string = "Loss Function [";
    std::for_each(loss_functions.begin(), loss_functions.end(),
        [&loss_functions_string, is_first = true](const auto& entry) mutable {
            if (is_first) {
                loss_functions_string += entry;
                is_first = false;
            } else {
                loss_functions_string += '|' + entry;
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
        ("delta", "Huber Loss Delta", cxxopts::value<double>()->default_value("1.0"))
        ("m,monotonicity", "Comma separated monotonicity direction for each column of X: '1' for ascending, '-1' for descending. (default: ascending)", cxxopts::value<std::string>())
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

    auto parsed_loss = parsed_options["loss"].as<std::string>();
    std::transform(parsed_loss.begin(), parsed_loss.end(), parsed_loss.begin(),
        [](unsigned char c){ return std::toupper(c); });

    if (std::find(loss_functions.begin(), loss_functions.end(), parsed_loss) == loss_functions.end()) {
        std::cout << options.help() << std::endl;
        std::cout << "Unrecognised loss function " << parsed_loss << std::endl;
        exit(1);
    }

    std::string monotonicity_modifier;
    if (parsed_options.count("monotonicity")) {
        monotonicity_modifier = parsed_options["monotonicity"].as<std::string>();
    }

    std::cout << "Running " << options.program();
    std::cout << " with loss function '" << parsed_loss << "' on\n";
    std::cout << ">> '" << input_file << "'\n";
    std::cout << "and writing result to \n";
    std::cout << ">> '" << output_file << "'" << std::endl;

    if (parsed_loss == "L1")
        run(input_file, output_file, monotonicity_modifier, gir::L1());
    else if (parsed_loss == "L2")
        run(input_file, output_file, monotonicity_modifier, gir::L2());
    else if (parsed_loss == "L2_WEIGHTED")
        run(input_file, output_file, monotonicity_modifier, gir::L2_WEIGHTED());
    else if (parsed_loss == "HUBER")
        run(input_file, output_file, monotonicity_modifier, gir::HUBER(parsed_options["delta"].as<double>()));

    return 0;
}
