#include "generalized_isotonic_regression.h"

#include <Highs.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace gir {

#ifdef __EMSCRIPTEN__
using HiGHS_Index = int32_t;

#else
using HiGHS_Index = int64_t;

#endif

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
generate_monotonic_points(uint64_t total, double sigma, uint64_t dimensions) {
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

uint64_t constraints_count(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& considered_idxs
) {
    // The function assumes there are no 0's stored in the adjacency_matrix;
    uint64_t previous = 0;
    uint64_t total_constraints = 0;

    for (size_t j : considered_idxs) {
        for (Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j); it; ++it) {
            previous = total_constraints;
            total_constraints += std::binary_search(considered_idxs.begin(), considered_idxs.end(), it.row());
            if (previous > total_constraints) {
                std::cout << "Overflow Error" << std::endl;
            }
        }
    }

    return total_constraints;
}

Eigen::SparseMatrix<int>
adjacency_to_LP_standard_form(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& considered_idxs
) {
    // The function assumes there are no 0's stored in the adjacency_matrix;
    // and that considered_idxs is sorted

    const uint64_t total_constraints = constraints_count(adjacency_matrix, considered_idxs);
    const uint64_t total_observations = considered_idxs.rows();
    const uint64_t columns = 2 * total_observations + total_constraints;

    VectorXu to_reserve(columns);
    for (Eigen::Index j = 0; j < 2 * total_observations; ++j) {
        to_reserve(j) = 1;
    }
    for (Eigen::Index j = 2 * total_observations; j < columns; ++j) {
        to_reserve(j) = 2;
    }

    Eigen::SparseMatrix<int, Eigen::ColMajor> standard_form(
        total_observations, columns);
    standard_form.reserve(to_reserve);

    int idx = 0;
    for (Eigen::Index j = 0; j < total_observations; ++j) {
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, considered_idxs(j));
            it;
            ++it
        ) {
            // add edges
            auto row_idx = std::find(considered_idxs.begin(), considered_idxs.end(), it.row());
            if (row_idx != considered_idxs.end()) {
                standard_form.insert( // row of constraint
                        std::distance(considered_idxs.begin(), row_idx),
                        2 * total_observations + idx) = 1;
                standard_form.insert( // col of constraint
                        j,
                        2 * total_observations + idx) = -1;
                ++idx;
            }
        }

        // Add slack/surplus variables (source and sink)
        standard_form.insert(j, j) = 1;
        standard_form.insert(j, j + total_observations) = -1;
    }

    standard_form.makeCompressed();

    return standard_form;
}

Eigen::VectorX<bool>
minimum_cut(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd loss_gradient, // z in the paper
    const VectorXu considered_idxs
) {
    /*
     * min b^T x
     * A x >= c
     * x >= 0
     *
     */
    const uint64_t total_observations = considered_idxs.rows();
    const auto A = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);
    const uint64_t total_constraints = A.cols() - 2 * total_observations;

    std::vector<double> b(A.cols());
    for (size_t i = 0; i < total_observations * 2; ++i) b[i] = 1;
    for (size_t i = 0; i < total_constraints; ++i) b[2 * total_observations + i] = 0;

    std::vector<double> c(loss_gradient.begin(), loss_gradient.end());

    const double infinity = 1.0e30; // Highs treats large numbers as infinity
    HighsModel model;
    model.lp_.num_col_ = A.cols();
    model.lp_.num_row_ = A.rows();
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = 0;
    model.lp_.col_cost_ = std::move(b);
    model.lp_.col_lower_ = std::move(std::vector<double>(A.cols(), 0));
    model.lp_.col_upper_ = std::move(std::vector<double>(A.cols(), infinity));
    model.lp_.row_lower_ = c;
    model.lp_.row_upper_ = std::move(c);

    // TODO instead of building an Eigen::SparseMatrix and then converting
    // it again here, could just directly build it in this format.
    // Might be harder to understand though.
    std::vector<HiGHS_Index> column_start_positions(A.cols() + 1);
    std::vector<HiGHS_Index> nonzero_row_index(A.nonZeros());
    std::vector<double> nonzero_values(A.nonZeros());
    uint64_t idx = 0;
    for (Eigen::Index j = 0; j < A.outerSize(); ++j) {
        column_start_positions[j] = idx;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, j); it; ++it) {
            nonzero_row_index[idx] = it.row();
            nonzero_values[idx] = it.value();
            ++idx;
        }
    }
    column_start_positions[column_start_positions.size()-1] = idx;

    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = std::move(column_start_positions);
    model.lp_.a_matrix_.index_ = std::move(nonzero_row_index);
    model.lp_.a_matrix_.value_ = std::move(nonzero_values);

    Highs highs;
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 4); // Primal
    highs.setOptionValue("log_to_console", false);

    highs.passModel(model);

    HighsStatus return_status = highs.run();
    assert(return_status == HighsStatus::kOk);

    const auto model_status = highs.getModelStatus();
    assert(model_status == HighsModelStatus::kOptimal);

    const HighsInfo& info = highs.getInfo();
    assert(info.dual_solution_status == 2); // feasible

    // Could also get solution from slack and surplus by looking at the
    // weight distributions either side of 0 and finding the middle point

    return Eigen::VectorXd::Map(
        &highs.getSolution().row_dual[0],
        highs.getSolution().row_dual.size()).array() > 0; // 0 left = 1 right
}

} // namespace gir
