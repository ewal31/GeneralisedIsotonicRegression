#include "generalized_isotonic_regression.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <Highs.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>


double calculate_loss_estimator(
    const LossFunction loss,
    const Eigen::VectorXd& vals
) {
    switch (loss) {
        case LossFunction::L2: return vals.mean();
    }
}

Eigen::VectorXd calculate_loss_derivative(
    const LossFunction loss,
    const double loss_estimator,
    const Eigen::VectorXd& vals
) {
    switch (loss) {
        case LossFunction::L2: return normalise(
                                   2 * (vals.array() - loss_estimator));
    }
}

Eigen::VectorXd normalise(const Eigen::VectorXd& loss_derivative) {
    if (loss_derivative.cwiseAbs().maxCoeff() > 0)
        // normalise largest coefficient to +/- 1
        return loss_derivative / loss_derivative.cwiseAbs().maxCoeff();
    return loss_derivative;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
generate_monotonic_points(size_t total, double sigma, size_t dimensions) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<> point_distribution(0, 1);
    std::normal_distribution<double> noise_distribution(0., sigma);

    Eigen::MatrixXd points = Eigen::MatrixXd::NullaryExpr(
        total,
        dimensions,
        [&point_distribution, &generator](){
            return point_distribution(generator);
        });

    Eigen::VectorXd regressed_values = points.rowwise().prod() +
        Eigen::MatrixXd::NullaryExpr(
            total,
            1,
            [&noise_distribution, &generator](){
                return noise_distribution(generator);
            });

    return std::make_pair(std::move(points), std::move(regressed_values));
}

Eigen::SparseMatrix<int>
adjacency_to_LP_standard_form(
    const Eigen::SparseMatrix<bool>& adjacency_matrix
) {
    uint16_t total_constraints = adjacency_matrix.nonZeros();
    uint16_t total_observations = adjacency_matrix.rows();
    uint16_t columns = 2 * total_observations + total_constraints;

    VectorXu to_reserve(columns);
    for (int j = 0; j < 2 * total_observations; ++j) {
        to_reserve(j) = 1;
    }
    for (int j = 2 * total_observations; j < columns; ++j) {
        to_reserve(j) = 2;
    }

    Eigen::SparseMatrix<int, Eigen::ColMajor> standard_form(
        total_observations, columns);

    standard_form.reserve(to_reserve);

    int idx = 0;
    for (int j = 0; j < total_observations; ++j) {
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j);
            it;
            ++it
        ) {
            // add edges
            standard_form.insert(it.row(), 2 * total_observations + idx) = 1;
            standard_form.insert(it.col(), 2 * total_observations + idx) = -1;
            ++idx;
        }

        // Add slack/surplus variables (source and sink)
        standard_form.insert(j, j) = 1;
        standard_form.insert(j, j + total_observations) = -1;
    }

    standard_form.makeCompressed();

    return standard_form;

    // // TODO reserve matrices
    // Eigen::SparseMatrix<int> row_coefficients(
    //     total_observations, 2 * total_observations + total_constraints);

    // Eigen::SparseMatrix<int> col_coefficients(
    //     total_observations, 2 * total_observations + total_constraints);

    // uint16_t idx = 0;
    // for (int j = 0; j < adjacency_matrix.outerSize(); ++j) {
    //     for (
    //         Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j);
    //         it;
    //         ++it
    //     ) {
    //         row_coefficients.insert(it.row(), 2 * total_observations + idx) = 1;
    //         col_coefficients.insert(it.col(), 2 * total_observations + idx) = 1;
    //         ++idx;
    //     }

    //     // Add slack variables (source and sink)
    //     row_coefficients.insert(j, j) = 1;
    //     col_coefficients.insert(j, j + total_observations) = 1;
    // }

    // row_coefficients.makeCompressed();
    // col_coefficients.makeCompressed();
    // return row_coefficients - col_coefficients;
}

// TODO take idxs into account
Eigen::VectorX<bool>
minimum_cut(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd loss_gradient, // z in the paper
    const VectorXu idxs
) {
    const uint16_t total_observations = idxs.rows();

    // already optimal
    if (loss_gradient(idxs).isApproxToConstant(0)) {
        return Eigen::VectorX<bool>::Constant(total_observations, true);
    }

    /*
     * min b^T x
     * A x >= c
     * x >= 0
     *
     */

    // TODO I need to think about this a bit more, but I think the subsets should always be consecutive
    // and this is therefore okay
    const auto A = adjacency_to_LP_standard_form(
        adjacency_matrix.block(
            idxs(0), idxs(0), total_observations, total_observations));

    const uint16_t total_constraints = A.cols() - 2 * total_observations;

    std::vector<double> b(A.cols());
    for (size_t i = 0; i < total_observations * 2; ++i) b[i] = 1;
    for (size_t i = 0; i < total_constraints; ++i) b[2 * total_observations + i] = 0;

    const std::vector<double> c(loss_gradient(idxs).begin(), loss_gradient(idxs).end());

    const double infinity = 1.0e30; // Highs treats large numbers as infinity
    HighsModel model;
    model.lp_.num_col_ = A.cols();
    model.lp_.num_row_ = A.rows();
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = 0;
    model.lp_.col_cost_ = b;
    model.lp_.col_lower_ = std::vector<double>(A.cols(), 0);
    model.lp_.col_upper_ = std::vector<double>(A.cols(), infinity);
    model.lp_.row_lower_ = c;
    model.lp_.row_upper_ = c;

    std::vector<int> column_start_positions(A.cols() + 1);
    std::vector<int> nonzero_row_index(A.nonZeros());
    std::vector<double> nonzero_values(A.nonZeros());
    uint16_t idx = 0;
    for (size_t j = 0; j < A.outerSize(); ++j) {
        column_start_positions[j] = idx;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, j); it; ++it) {
            nonzero_row_index[idx] = it.row();
            nonzero_values[idx] = it.value();
            ++idx;
        }
    }
    column_start_positions[column_start_positions.size()-1] = idx;

    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = column_start_positions;
    model.lp_.a_matrix_.index_ = nonzero_row_index;
    model.lp_.a_matrix_.value_ = nonzero_values;

    Highs highs;
    HighsStatus return_status;
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 4); // Primal
    highs.setOptionValue("log_to_console", false);

    return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);

    // Solve the model
    return_status = highs.run();
    assert(return_status == HighsStatus::kOk);

    // Get the model status
    const auto model_status = highs.getModelStatus();
    assert(model_status == HighsModelStatus::kOptimal);

    const HighsInfo& info = highs.getInfo();
    assert(info.dual_solution_status == 2); // feasible

    // Could also get solution from slack and surplus by looking at the
    // weight distributions either side of 0 and finding the middle point
    // Eigen::VectorXd x =
    //     Eigen::VectorXd::Map(&highs.getSolution().col_value[total_observations], total_observations).array() +
    //     Eigen::VectorXd::Map(&highs.getSolution().col_value[0], total_observations).array();
    // std::partial_sum(x.begin(), x.end(), x.begin());
    // const double middle_point = x(x.rows() - 1) / 2;

    // fmt::println("colsol:\n{}", highs.getSolution().col_value);

    // Eigen::VectorX<bool> solution(total_observations);
    // size_t i = 0;
    // for (; x(i) <= middle_point; ++i) { solution(i) = false; } // could be some instability
    // for (; i < total_observations; ++i) { solution(i) = true; }

    const Eigen::VectorX<bool> dual_solution = Eigen::VectorXd::Map(
        &highs.getSolution().row_dual[0],
        highs.getSolution().row_dual.size()).array() > 0; // 0 left = 1 right

    return dual_solution;
}

std::pair<VectorXu, Eigen::VectorXd>
generalised_isotonic_regression(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd& y,
    const LossFunction loss_function,
    const uint16_t max_iterations
) {
    const uint16_t total_observations = y.rows();

    uint16_t group_count = 0;
    double max_cut_values = 0;
    const double sentinal = 1e-30; // TODO handle properly

    VectorXu groups = VectorXu::Zero(total_observations);
    Eigen::VectorXd y_fit = Eigen::VectorXd::Zero(total_observations);

    Eigen::VectorXd group_loss = Eigen::VectorXd::Zero(total_observations); // objective value of partition problems. used to decide which cut to make at each iteration
    VectorXu considered_idxs = VectorXu::Zero(total_observations);

    for (uint16_t iteration = 1; max_iterations == 0 || iteration < max_iterations; ++iteration) {
        auto [max_cut_value, max_cut_idx] = argmax(group_loss);

        if (max_cut_value == sentinal) {
            break;
        }

        considered_idxs = find(groups,
            [&groups, &idx = groups(max_cut_idx)](const int i){
                return idx == groups(i);
            });

        const double estimator = calculate_loss_estimator(loss_function, y(considered_idxs));
        const auto derivative = calculate_loss_derivative(loss_function, estimator, y(considered_idxs));
        const auto solution =  minimum_cut(adjacency_matrix, derivative, considered_idxs);
        auto [left, right] = argpartition(solution);

        if (right.rows() == considered_idxs.rows()) {
            group_loss(considered_idxs).array() = sentinal;
        } else {
            group_loss(considered_idxs).array() =
                calculate_loss_derivative(loss_function, estimator, y(considered_idxs(right))).sum() -
                    calculate_loss_derivative(loss_function, estimator, y(considered_idxs(left))).sum();

            y_fit(considered_idxs(left)).array() = calculate_loss_estimator(loss_function, y(considered_idxs(left)));
            y_fit(considered_idxs(right)).array() = calculate_loss_estimator(loss_function, y(considered_idxs(right)));
            groups(considered_idxs(left)).array() = ++group_count;
            groups(considered_idxs(right)).array() = ++group_count;
        }

        std::cout << "Iteration: " << iteration << std::endl;
        std::cout << "group_loss:\n" << group_loss << std::endl;
        std::cout << "y_fit:\n" << y_fit << std::endl;
        std::cout << "groups:\n" << groups << "\n" << std::endl;
    }

    // Renumber from 1
    group_count = 1;
    for (size_t i = 0; i < groups.rows() - 1; ++i) {
        auto current = groups(i);
        groups(i) = group_count;
        group_count = current == groups(i+1) ? group_count : group_count + 1;
    }
    groups(groups.rows() - 1) = group_count;

    return std::make_pair(std::move(groups), std::move(y_fit));
}

void run() {
    auto [X, y] = generate_monotonic_points(5, 0.001, 2);

    // Eigen::MatrixX<uint8_t> X(5, 1);
    // X << 1,
    //      2,
    //      3,
    //      4,
    //      5;

    // Eigen::VectorXd y(5);
    // // regressed_values << 1, 1, 1.2, 1, 1;
    // y << 1, 1, 3, 5, 5;

    // std::cout << "points:\n" << X << '\n' << std::endl;
    // std::cout << "regressed_values:\n" << y << '\n' << std::endl;

    auto [adjacency_matrix, idx_original, idx_new] =
        points_to_adjacency(X);

    std::cout << "adjacency_matrix:\n" << adjacency_matrix << '\n' << std::endl;
    // std::cout << "idx_original:\n" << idx_original << '\n' << std::endl;
    // std::cout << "idx_new:\n" << idx_new << '\n' << std::endl;
    std::cout << "points reordered to adjacency_matrix\n";
    std::cout << X(idx_new, Eigen::all) << "\n" << std::endl;
    std::cout << y(idx_new) << "\n" << std::endl;

    auto [groups, y_fit] = generalised_isotonic_regression(adjacency_matrix, y, LossFunction::L2);

    std::cout << "fit y values:\n" << y_fit << "\n" << std::endl;
}
