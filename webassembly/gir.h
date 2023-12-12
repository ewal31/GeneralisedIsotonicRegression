#pragma once

#include <generalized_isotonic_regression.h>

#include <emscripten/bind.h>
#include <Eigen/Core>

// A really minimial buffer implementation for writing
// data into a html element or the browser console if
// not set
//
// This should only be used for an output stream
class HTMLElementBuffer : public std::streambuf {
    std::array<char, 1024> data;
    std::string element_id;

    int_type overflow(int_type c = traits_type::eof()) override;
    int sync() override;
    void write_to_element();
    void init();

  public:
    HTMLElementBuffer();
    void set_element(const std::string& element_id);
};

HTMLElementBuffer element_buffer;
std::ostream element_console(&element_buffer);
void update_element_buffer_element_id(const std::string& element_id);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
parse_input_data(const std::string& input);

std::string
format_output_data(gir::VectorXu groups, Eigen::VectorXd y_fit);

class gir_result {
    uint32_t total_iterations;
    const uint32_t total_rows;
    const uint32_t total_columns;
    const Eigen::MatrixXd x;
    const Eigen::VectorXd y;
    const Eigen::VectorXd weights;
    std::vector<gir::VectorXu> groups;
    std::vector<Eigen::VectorXd> y_fit;
    std::vector<double> loss;
    std::string status;

  public:
    gir_result(
        const uint32_t total_rows,
        const uint32_t total_columns,
        const Eigen::MatrixXd x,
        const Eigen::VectorXd y,
        const Eigen::VectorXd weights
    ) : total_iterations(0),
        total_rows(total_rows),
        total_columns(total_columns),
        x(x),
        y(y),
        weights(weights)
    {}

    gir_result(
        std::string status
    ) : total_iterations(0),
        total_rows(0),
        total_columns(0),
        status(status)
    {}

    uint32_t iterations() const {
        return total_iterations;
    }

    uint32_t rows() const {
        return total_rows;
    }

    uint32_t cols() const {
        return total_columns;
    }

    double get_x(uint32_t row, uint32_t col) const {
        return x(row, col);
    }

    double get_y(uint32_t row) const {
        return y(row);
    }

    double get_group(uint32_t iteration, uint32_t row) const {
        return groups[iteration](row);
    }

    double get_y_fit(uint32_t iteration, uint32_t row) const {
        return y_fit[iteration](row);
    }

    double get_loss(uint32_t iteration) const {
        return loss[iteration];
    }

    std::string get_formatted(uint32_t iteration) const {
        return total_rows == 0 ? status : format_output_data(groups[iteration], y_fit[iteration]);
    }

    void set_status(std::string& status) {
        this->status = status;
    }

    template <typename LossType>
    void add_iteration(
        const LossType& loss_function,
        gir::VectorXu groups,
        Eigen::VectorXd y_fit
    ) {
        ++this->total_iterations;
        this->groups.push_back(std::move(groups));
        this->y_fit.push_back(std::move(y_fit));
        loss.push_back(this->calculate_loss(loss_function));
    }

    template <typename LossType>
    double calculate_loss(const LossType& loss_function) {
        return loss_function.loss(
            y,
            y_fit.back(),
            weights
        );
    }
};

std::string generate_input_data(
    uint32_t total,
    uint32_t dimensions
);

template<typename LossType>
void
run_gir (
    gir_result& result,
    gir::VectorXu& idx_original,
    Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& weights,
    const gir::LossFunction<LossType>& loss_fun,
    uint64_t max_iterations = 0
) {
    const uint64_t total_observations = y.rows();
    uint64_t group_count = 0;

    // objective value of partitions that is used to decide
    // which cut to make at each iteration
    Eigen::VectorXd group_loss = Eigen::VectorXd::Zero(total_observations);

    gir::VectorXu groups = gir::VectorXu::Zero(total_observations);
    Eigen::VectorXd y_fit = Eigen::VectorXd::Constant(
        total_observations,
        loss_fun.estimator(y, weights)
    );

    // Add starting point before first split
    result.add_iteration(
        loss_fun,
        groups(idx_original),
        y_fit(idx_original)
    );

    // These iterations could potentially be done in parallel (except the first)
    for (uint64_t iteration = 0; max_iterations == 0 || iteration < max_iterations; ) {
        const auto status = gir_update(
            adjacency_matrix,
            y,
            weights,
            loss_fun,
            group_count,
            group_loss,
            groups,
            y_fit
        );

        if (status == 1) {
            ++iteration;
            result.add_iteration(
                loss_fun,
                groups(idx_original),
                y_fit(idx_original)
            );
            element_console
                << "Completed iteration: "
                << iteration
                << " with loss: "
                << result.get_loss(result.iterations() - 1)
                << std::endl;
        } else if (status == -1) {
            return;
        }
    }
}

template<typename LossType>
gir_result
run_iso_regression_with_loss(
    const gir::LossFunction<LossType>& loss_function,
    const std::string& input,
    const uint32_t max_iterations
) {
    element_console << "Parsing Input Data" << std::endl;
    const auto [X, y, weights] = parse_input_data(input);

    gir_result result(X.rows(), X.cols(), X, y, weights);

    element_console << "Building Adjacency Matrix" << std::endl;
    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    element_console << "Running Isotonic Regression" << std::endl;

    run_gir(
        result,
        idx_original,
        adjacency_matrix,
        y,
        weights,
        loss_function,
        max_iterations
    );

    double total_loss = result.calculate_loss(loss_function);
    element_console << "Finished with total loss: " << total_loss << std::endl;

    return result;
}

gir_result run_iso_regression(
    const std::string& loss_function,
    const std::string& input,
    const std::string& loss_paramteter,
    const std::string& max_iterations
);

EMSCRIPTEN_BINDINGS(EmbindVectorDouble) {
    emscripten::function("run_iso_regression", &run_iso_regression);
    emscripten::function("set_console_element", &update_element_buffer_element_id);
    emscripten::function("generate_input_data", &generate_input_data);

    emscripten::class_<gir_result>("Result")
        .property("iterations", &gir_result::iterations)
        .property("rows", &gir_result::rows)
        .property("cols", &gir_result::cols)
        .function("get_x", &gir_result::get_x)
        .function("get_y", &gir_result::get_y)
        .function("get_group", &gir_result::get_group)
        .function("get_y_fit", &gir_result::get_y_fit)
        .function("get_loss", &gir_result::get_loss)
        .function("get_formatted", &gir_result::get_formatted);
}
