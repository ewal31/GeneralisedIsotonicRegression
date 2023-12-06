#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <tuple>
#include <utility>

#include "loss.h"
#include "utility.h"

namespace gir {

template<typename V>
bool
is_monotonic(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorX<V>& y,
    const double tolerance = 1e-6
) {
    for (Eigen::Index j = 0; j < adjacency_matrix.cols(); ++j) {
        for (Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j); it; ++it) {
            if (it.value()) {
                if ((y(it.row()) - y(it.col())) > tolerance) {
                    return false;
                }
            }
        }
    }

    return true;
}

template<typename V, typename K>
bool
is_monotonic(
    const Eigen::MatrixX<V>& points,
    const Eigen::VectorX<K>& y,
    const double tolerance = 1e-6
) {
    auto sorted_idxs = argsort(points);
    Eigen::MatrixX<V> sorted_points = points(sorted_idxs, Eigen::all);
    Eigen::VectorX<V> sorted_y = y(sorted_idxs);

    for (size_t row1 = 1; row1 < y.rows(); ++row1) {
        for (size_t row2 = 0; row2 < row1; ++row2) {
            bool is_smaller = (sorted_points(row2, Eigen::all).array() <=
                    sorted_points(row1, Eigen::all).array()).all();
            bool within_tolerance = (sorted_y(row2) - sorted_y(row1)) < tolerance;

            if (is_smaller && !within_tolerance) return false;
        }
    }

    return true;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
generate_monotonic_points(
    uint64_t total,
    double sigma = 0.1,
    uint64_t dimensions = 3
);

uint64_t constraints_count(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& considered_idxs
);

/**
 * Build a sparse adjacency matrix from a set of points.
 *
 * The created adjacency matrix doesn't contain links between two nodes
 * if there is another path joining each node.
 *
 * For example, given the points 1, 2, 3, 4 that follow ordering
 * we have the following graph
 *
 *   -------
 *  /       \
 * 1 -> 2    |
 *   \  |  \ |
 *    v v   vv
 *      3 -> 4
 *
 * converted to an adjacency matrix (considering only upper triangle)
 * we end up with
 *
 *   1  2  3  4     we don't include 1-3 as it is implied by 1-2-3
 * 1    x           and the same with the other missing links
 * 2       x
 * 3          x
 * 4
 *
 *
 * @param `points` each row is a multidimensional point
 * @return a tuple containing the following:
 *         - adjacency matrix: the point adjacency(i, j) == true iff points(i) <= points(j)
 *         - ind_original:     adjacency(ind_original(i), ind_original(j)) shows whether there is an edge between points(i) and points(j)
 *         - ind_new:          adjacency(i, j) compares points(ind_new(i)) and points(ind_new(j))
 *
 */
template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency(const Eigen::MatrixX<V>& points) {
    // TODO with lots of points this is a real bottleneck at O(n^2)
    // the memory usage seems fine though.
    // ideas:
    // * sorted idxs along each axis then can just move to the next point
    //   and shouldn't need to do all the n comparisons each time?
    // * maybe something from computational geometry?
    // * not sure how eigen goes multithreaded?
    // * keep tree of linked points so can run from newest backwards and save on comparisons
    // * some sort of topological sort?

    const uint64_t total_points = points.rows();
    const auto& sorted_idxs = argsort(points);
    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    // Not sure what a good estimate would be.
    // Below is roughly half upperbound. maybe something like log(x)
    //adjacency.reserve(Eigen::VectorXi::LinSpaced(total_points, 0, total_points-1).array() / 2 + 1);
    VectorXu degree = VectorXu::Zero(total_points);
    const Eigen::MatrixX<V> sorted_points = points(sorted_idxs, Eigen::all).transpose();
    Eigen::VectorX<bool> is_predecessor = Eigen::VectorX<bool>::Zero(total_points);
    Eigen::VectorX<bool> is_equal = Eigen::VectorX<bool>::Zero(total_points);

    for (uint64_t i = 1; i < total_points; ++i) {
        const auto& previous_points = sorted_points(Eigen::all,
            VectorXu::LinSpaced(i, 0, i-1)).array();
        const auto& current_point = sorted_points(Eigen::all,
            VectorXu::LinSpaced(i, i, i)).array();

        is_predecessor(Eigen::seq(0, i-1)) =
            (previous_points <= current_point).colwise().all();

        // Just for Equal Points :(
        // TODO it is probably also valid to just create a loop
        // i.e. just a constraint from the last repeated to the first repeated
        //      in any group of repeated points
        is_equal.setZero();
        for (Eigen::Index idx = i-1; idx >= 0; --idx) {
            if (!sorted_points(Eigen::all, idx).cwiseEqual(sorted_points(Eigen::all, i)).all())
                break;
            is_equal(idx) = true;
            ++degree(idx);
            adjacency.insert(i, idx) = 1;
        }

        degree(i) = is_predecessor.count();

        /* If there is a chain of points that are all predecessors, we take only
         * the largest. So, we check if the outgoing edge of a predecessor
         * connects to another predecessor (and would take it instead).
         */
        for (Eigen::Index j = 0; j < adjacency.outerSize(); ++j) {
            if (is_predecessor(j)) {
                for (
                    Eigen::SparseMatrix<bool>::InnerIterator it(adjacency, j);
                    it;
                    ++it
                ) {
                    // TODO it might be worth stopping early if is_predecessor is all 0's;
                    if (it.value() && !is_equal(it.row())) {
                        is_predecessor(it.row()) = false;
                    }
                }
            }
        }

        adjacency.col(i) = is_predecessor.sparseView();
    }

    const auto& degree_idxs = argsort(degree);
    VectorXu rev_degree_idxs(degree_idxs.rows());
    rev_degree_idxs(degree_idxs) = VectorXu::LinSpaced(degree_idxs.rows(), 0, degree_idxs.rows() - 1);

    // TODO try build from triplets. They probably have a smart way buildling the matrix.
    // create a copy of adjacency reordered to be the same order as degree_idxs.
    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency_ordered(total_points, total_points);
    // adjacency_ordered.reserve(Eigen::VectorXi::LinSpaced(total_points, 0, total_points-1).array() / 2 + 1);
    // // adjacency_ordered.reserve(
    // //     Eigen::VectorXi::Constant(total_points, degree.maxCoeff()));

    // for (uint64_t j = 0; j < adjacency.outerSize(); ++j) {
    //     for (
    //         Eigen::SparseMatrix<bool>::InnerIterator it(adjacency, j);
    //         it;
    //         ++it
    //     ) {
    //         adjacency_ordered.insert(
    //             rev_degree_idxs(it.row()),
    //             rev_degree_idxs(it.col())) = it.value();
    //     }
    // }
    // adjacency_ordered.makeCompressed();

    std::vector<Eigen::Triplet<bool>> tripletList;
    tripletList.reserve(adjacency.nonZeros());

    for (Eigen::Index j = 0; j < adjacency.outerSize(); ++j) {
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(adjacency, j);
            it;
            ++it
        ) {
            tripletList.emplace_back(rev_degree_idxs(it.row()), rev_degree_idxs(it.col()), it.value());
        }
    }

    adjacency_ordered.setFromTriplets(tripletList.begin(), tripletList.end());

    VectorXu idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);

    VectorXu degreem(total_points);
    degreem(degree_idxs) = idxs;

    VectorXu ordm(total_points);
    ordm(sorted_idxs) = idxs;

    VectorXu ind_original = degreem(ordm);
    auto ind_new = argsort(ind_original);

    return std::make_tuple(
        std::move(adjacency_ordered),
        std::move(ind_original),
        std::move(ind_new));
}

Eigen::SparseMatrix<int>
adjacency_to_LP_standard_form(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const VectorXu& idxs
);

Eigen::VectorX<bool>
minimum_cut(
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd loss_gradient, // z in the paper
    const VectorXu idxs
);

// TODO change return type to enum
template<typename LossType, typename int_type>
int8_t
gir_update (
    Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& weights,
    const LossFunction<LossType>& loss_fun,
    int_type& group_count,
    Eigen::VectorXd& group_loss,
    VectorXu& groups,
    Eigen::VectorXd& y_fit
) {
    const double sentinal = 1e-30; // TODO handle properly

    auto [max_cut_value, max_cut_idx] = argmax(group_loss);

    if (max_cut_value == sentinal) {
        return -1; // optimal
    }

    VectorXu considered_idxs = find(groups,
        [&groups, &idx = groups(max_cut_idx)](const int i){
            return idx == groups(i);
        });

    const double estimator = loss_fun.estimator(y(considered_idxs), weights(considered_idxs));

    const auto& derivative = loss_fun.derivative(estimator, y(considered_idxs), weights(considered_idxs));

    const bool zero_loss = derivative.isApproxToConstant(0);
    const bool no_constraints =
        constraints_count(adjacency_matrix, considered_idxs) == 0;

    if (zero_loss) {
        group_loss(considered_idxs).array() = sentinal;
        y_fit(considered_idxs).array() = estimator;
        return 0; // nothing changed

    } else if (no_constraints) {
        group_loss(considered_idxs).array() = sentinal;

        for (auto idx : considered_idxs) {
            y_fit(idx) = loss_fun.estimator(
                    y(idx, Eigen::all), weights(idx, Eigen::all));
            groups(idx) = ++group_count;
        }
        // y_fit(considered_idxs).array() = estimator;
        // groups(considered_idxs).array() = ++group_count;
        return 1; // something changed

    } else {
        const auto solution =
            minimum_cut(adjacency_matrix, derivative, considered_idxs); // TODO could pass num constrains in as calculating twice
        auto [left, right] = argpartition(solution);

        const bool no_cut = left.rows() == 0 || right.rows() == 0;

        if (no_cut) {
            group_loss(considered_idxs).array() = sentinal;
            y_fit(considered_idxs).array() = estimator;
            return 0; // nothing changed

        } else {
            const auto& loss_right = loss_fun.derivative(
                estimator, y(considered_idxs(right)), weights(considered_idxs(right))).sum();

            const auto& loss_left = loss_fun.derivative(
                estimator, y(considered_idxs(left)), weights(considered_idxs(left))).sum();

            group_loss(considered_idxs).array() = loss_right - loss_left;

            y_fit(considered_idxs(left)).array() = loss_fun.estimator(
                y(considered_idxs(left)), weights(considered_idxs(left)));

            y_fit(considered_idxs(right)).array() = loss_fun.estimator(
                y(considered_idxs(right)), weights(considered_idxs(right)));

            groups(considered_idxs(right)).array() = ++group_count;

            return 1; // something changed
        }
    }
}

// TODO need a version that can take by reference as well
//      but the copy is really convenient for auto changing
//      the type
template<typename LossType>
std::pair<VectorXu, Eigen::VectorXd>
generalised_isotonic_regression (
    Eigen::SparseMatrix<bool>& adjacency_matrix,
    Eigen::VectorXd y,
    Eigen::VectorXd weights,
    LossFunction<LossType> loss_fun,
    uint64_t max_iterations = 0
) {
    const uint64_t total_observations = y.rows();
    uint64_t group_count = 0;

    // objective value of partitions that is used to decide
    // which cut to make at each iteration
    Eigen::VectorXd group_loss = Eigen::VectorXd::Zero(total_observations);

    // returned result
    VectorXu groups = VectorXu::Zero(total_observations);
    Eigen::VectorXd y_fit = Eigen::VectorXd::Zero(total_observations);

    // These iterations could potentially be done in parallel (except the first)
    for (uint64_t iteration = 0; max_iterations == 0 || iteration < max_iterations; ++iteration) {
        if (gir_update(
            adjacency_matrix,
            y,
            weights,
            loss_fun,
            group_count,
            group_loss,
            groups,
            y_fit
        )) break;
    }

    return std::make_pair(std::move(groups), std::move(y_fit));
}

} // namespace gir
