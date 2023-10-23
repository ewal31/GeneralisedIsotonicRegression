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


// ECDF Implementation (lets just try recursive first)
template<typename V>
void _ecdf(const Eigen::Matrix<V>& points) {

}

template<typename V>
void ecdf(const Eigen::Matrix<V>& points) {
    const uint64_t total_points = points.rows();
    const uint64_t dimensions = points.cols();

    // sort by y to speed up comparison checks
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            if (points(i, 1) == points(j, 1))
                return points(i, 0) < points(j, 0);
            return points(i, 1) < points(j, 1);
        });

    // TODO don't actually have to build this. Could instead when looking up a point
    // just jump back through x_sorted then y_sorted
    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);

    // but work on these indices then sorted by x
    VectorXu x_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        x_sorted_idxs.begin(),
        x_sorted_idxs.end(),
        [&y_sorted_points](const auto& i, const auto& j) {
            if (y_sorted_points(i, 0) == y_sorted_points(j, 0)) {
                if (y_sorted_points(i, 1) == y_sorted_points(j, 1))
                    return i < j; // Equal y_sorted_points are in the same order as y_sorted
                else
                    return y_sorted_points(i, 1) < y_sorted_points(j, 1);
            }
            return y_sorted_points(i, 0) < y_sorted_points(j, 0);
        });

}













// Sort points by y
//
// Then into the vector, we have xwise argsort of the points
//
// The sparse matrix size for each column is then the min of the idx and value contained when running along this argsort
//
// When merging we run along this argsort vector
//
// And we keep track of the largest contained value (y point) smaller than the y value of points to the right of the partition.
// Whenever this value increases with a decreasing x it must be a new predecessor that isn't also a predecessor of one of the other points and should be added to the matrix
template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency_2d(const Eigen::MatrixX<V>& points) {
    const uint64_t total_points = points.rows();
    const uint64_t dimensions = points.cols();

    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    adjacency.reserve(Eigen::VectorXi::Constant(total_points, 30 ? 30 < total_points : total_points));

    // sort by y to speed up comparison checks
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            if (points(i, 1) == points(j, 1))
                return points(i, 0) < points(j, 0);
            return points(i, 1) < points(j, 1);
        });

    // TODO don't actually have to build this. Could instead when looking up a point
    // just jump back through x_sorted then y_sorted
    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);

    // but work on these indices then sorted by x
    VectorXu x_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        x_sorted_idxs.begin(),
        x_sorted_idxs.end(),
        [&y_sorted_points](const auto& i, const auto& j) {
            if (y_sorted_points(i, 0) == y_sorted_points(j, 0)) {
                if (y_sorted_points(i, 1) == y_sorted_points(j, 1))
                    return i < j; // Equal y_sorted_points are in the same order as y_sorted
                else
                    return y_sorted_points(i, 1) < y_sorted_points(j, 1);
            }
            return y_sorted_points(i, 0) < y_sorted_points(j, 0);
        });

    VectorXu largest_y_below = VectorXu::Zero(total_points).array() - 1;
    largest_y_below(x_sorted_idxs(0)) = 0;
    std::vector<Eigen::Index> to_add;
    to_add.reserve(total_points);

    Eigen::VectorXd cmps = Eigen::VectorXd::Zero(total_points);
    for (Eigen::Index j = 1; j < total_points; ++j) {
        to_add.clear();
        Eigen::Index max_y = -1;
        const Eigen::Index j_val = x_sorted_idxs(j);

        // const auto lower_bound = largest_y_below(VectorXu::LinSpaced(j_val, 0, j_val - 1)).maxCoeff();
        Eigen::Index lower_bound = j_val;
        while (lower_bound > 0) {
            if (largest_y_below(lower_bound) >= 0) break;
            --lower_bound;
        }
        lower_bound = largest_y_below(lower_bound);

        Eigen::Index i = 0;
        for (; i < j - lower_bound; ++i) {
            const Eigen::Index i_val = x_sorted_idxs(j - i - 1);
            if (max_y < i_val && i_val < j_val) {
                to_add.push_back(j - i - 1);
                max_y = i_val;
            }
        }

        // std::cout << j << " " << lower_bound << " " << j - lower_bound << " " << j - (i-1) - 1 << " " << y_sorted_points(j_val, Eigen::all) << std::endl;
        // std::cout << "j: " << j << ", Lower bound: " << lower_bound << ", j_val: " << j_val << ", compared: " << i << ", " << to_add.size() << std::endl;

        cmps(j) = i;

        largest_y_below(j_val) = j;

        // to_add is populated with the largest values at the start
        // so add to the adjacency_matrix in reverse
        for (size_t idx = 0; idx < to_add.size(); ++idx) {
            const auto row = to_add[to_add.size() - idx - 1];
            adjacency.insert(row, j) = 1;
        }
    }

    std::cout << "Average compared " << cmps.mean() << std::endl;

    // TODO Is there some way to do this without all the comparisons again?
    // Correction for duplicate points
    bool has_equal_chain = false;
    Eigen::Index first_equal_index = 0;
    for (Eigen::Index j = 1; j < total_points; ++j) {
        const auto& p1 = y_sorted_points(x_sorted_idxs(j-1), Eigen::all);
        const auto& p2 = y_sorted_points(x_sorted_idxs(j), Eigen::all);

        if (!has_equal_chain && p1 == p2) {
            has_equal_chain = true;
            first_equal_index = j-1;
        } else if (has_equal_chain && p1 != p2) {
            has_equal_chain = false;
            adjacency.insert(j-1, first_equal_index) = 1;
        }
    }

    if (has_equal_chain) {
        adjacency.insert(total_points - 1, first_equal_index) = 1;
    }

    // Finalise Adjacency and Point Index Mappings
    adjacency.makeCompressed();
    gir::VectorXu idx_new = y_sorted_idxs(x_sorted_idxs);

    return std::make_tuple(
        std::move(adjacency),
        std::move(argsort(idx_new)),
        std::move(idx_new));
}

template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency_2d_2(const Eigen::MatrixX<V>& points) {
    const uint64_t total_points = points.rows();
    const uint64_t dimensions = points.cols();

    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    adjacency.reserve(Eigen::VectorXi::Constant(total_points, 30 ? 30 < total_points : total_points));

    // sort by y to speed up comparison checks
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            if (points(i, 1) == points(j, 1))
                return points(i, 0) < points(j, 0);
            return points(i, 1) < points(j, 1);
        });

    // TODO don't actually have to build this. Could instead when looking up a point
    // just jump back through x_sorted then y_sorted
    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);

    // but work on these indices then sorted by x
    VectorXu x_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        x_sorted_idxs.begin(),
        x_sorted_idxs.end(),
        [&y_sorted_points](const auto& i, const auto& j) {
            if (y_sorted_points(i, 0) == y_sorted_points(j, 0)) {
                if (y_sorted_points(i, 1) == y_sorted_points(j, 1))
                    return i < j; // Equal y_sorted_points are in the same order as y_sorted
                else
                    return y_sorted_points(i, 1) < y_sorted_points(j, 1);
            }
            return y_sorted_points(i, 0) < y_sorted_points(j, 0);
        });

    VectorXu largest_y_below = VectorXu::Zero(total_points).array() - 1;
    largest_y_below(x_sorted_idxs(0)) = 0; // add 0th value as not otherwise touched
    // std::vector<Eigen::Index> to_add;
    // to_add.reserve(total_points);
    Eigen::VectorX<bool> to_add(total_points);

    // Eigen::VectorXd cmps = Eigen::VectorXd::Zero(total_points);
    for (Eigen::Index j = 1; j < total_points; ++j) {
        to_add.setZero();
        Eigen::Index max_y = -1;
        const Eigen::Index j_val = x_sorted_idxs(j);

        Eigen::Index lower_bound = j_val;
        while (lower_bound > 0) {
            if (largest_y_below(lower_bound) >= 0) break;
            --lower_bound;
        }
        // Eigen::Index lower_bound = 0;
        // (largest_y_below.array() > -1).select(VectorXu::LinSpaced(j_val+1, 0, j_val), -1).maxCoeff(&lower_bound);
        lower_bound = largest_y_below(lower_bound);

        const auto to_check = VectorXu::LinSpaced(j - lower_bound, lower_bound,  j - 1);
        to_add(to_check).array() = x_sorted_idxs(to_check).array() < j_val;

        Eigen::Index i = 0;
        for (; i < j - lower_bound; ++i) {
            const Eigen::Index i_val = x_sorted_idxs(j - i - 1);
            if (to_add(j - i - 1)) {
                to_add(j - i - 1) = max_y < i_val;
                max_y = max_y < i_val ? i_val : max_y;
            }
        }

        // std::cout << j << " " << lower_bound << " " << j - lower_bound << " " << j - (i-1) - 1 << " " << y_sorted_points(j_val, Eigen::all) << std::endl;
        // std::cout << "j: " << j << ", Lower bound: " << lower_bound << ", j_val: " << j_val << ", compared: " << i << ", " << to_add.size() << std::endl;

        // cmps(j) = i;

        largest_y_below(j_val) = j;

        // to_add is populated with the largest values at the start
        // so add to the adjacency_matrix in reverse
        // for (size_t idx = 0; idx < to_add.size(); ++idx) {
        //     const auto row = to_add[to_add.size() - idx - 1];
        //     adjacency.insert(row, j) = 1;
        // }
        adjacency.col(j) = to_add.sparseView();
    }

    // std::cout << "Average compared " << cmps.mean() << std::endl;

    // TODO Is there some way to do this without all the comparisons again?
    // Correction for duplicate points
    bool has_equal_chain = false;
    Eigen::Index first_equal_index = 0;
    for (Eigen::Index j = 1; j < total_points; ++j) {
        const auto& p1 = y_sorted_points(x_sorted_idxs(j-1), Eigen::all);
        const auto& p2 = y_sorted_points(x_sorted_idxs(j), Eigen::all);

        if (!has_equal_chain && p1 == p2) {
            has_equal_chain = true;
            first_equal_index = j-1;
        } else if (has_equal_chain && p1 != p2) {
            has_equal_chain = false;
            adjacency.insert(j-1, first_equal_index) = 1;
        }
    }

    if (has_equal_chain) {
        adjacency.insert(total_points - 1, first_equal_index) = 1;
    }

    // Finalise Adjacency and Point Index Mappings
    adjacency.makeCompressed();
    gir::VectorXu idx_new = y_sorted_idxs(x_sorted_idxs);

    return std::make_tuple(
        std::move(adjacency),
        std::move(argsort(idx_new)),
        std::move(idx_new));
}

template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency_2d_3(const Eigen::MatrixX<V>& points) {
    const uint64_t total_points = points.rows();
    const uint64_t dimensions = points.cols();

    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    adjacency.reserve(Eigen::VectorXi::Constant(total_points, 30 ? 30 < total_points : total_points));

    // sort by y to speed up comparison checks
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            if (points(i, 1) == points(j, 1))
                return points(i, 0) < points(j, 0);
            return points(i, 1) < points(j, 1);
        });

    // TODO don't actually have to build this. Could instead when looking up a point
    // just jump back through x_sorted then y_sorted
    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);

    // but work on these indices then sorted by x
    VectorXu x_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        x_sorted_idxs.begin(),
        x_sorted_idxs.end(),
        [&y_sorted_points](const auto& i, const auto& j) {
            if (y_sorted_points(i, 0) == y_sorted_points(j, 0)) {
                if (y_sorted_points(i, 1) == y_sorted_points(j, 1))
                    return i < j; // Equal y_sorted_points are in the same order as y_sorted
                else
                    return y_sorted_points(i, 1) < y_sorted_points(j, 1);
            }
            return y_sorted_points(i, 0) < y_sorted_points(j, 0);
        });

    VectorXu largest_y_below = VectorXu::Zero(total_points).array() - 1;
    largest_y_below(x_sorted_idxs(0)) = 0; // add 0th value as not otherwise touched
    // std::vector<Eigen::Index> to_add;
    // to_add.reserve(total_points);
    Eigen::VectorX<bool> to_add(total_points);

    // Eigen::VectorXd cmps = Eigen::VectorXd::Zero(total_points);
    for (Eigen::Index j = 1; j < total_points; ++j) {
        to_add.setZero();
        // Eigen::Index max_y = -1;
        const Eigen::Index j_val = x_sorted_idxs(j);

        Eigen::Index idx = j_val - 1;
        Eigen::Index max_x = -1;
        while (idx >= 0) {
            if (largest_y_below(idx) > max_x) {
                max_x = largest_y_below(idx);
                to_add(max_x) = true;
            }
            --idx;
        }

        // Eigen::Index lower_bound = j_val;
        // while (lower_bound > 0) {
        //     if (largest_y_below(lower_bound) >= 0) break;
        //     --lower_bound;
        // }
        // // Eigen::Index lower_bound = 0;
        // // (largest_y_below.array() > -1).select(VectorXu::LinSpaced(j_val+1, 0, j_val), -1).maxCoeff(&lower_bound);
        // lower_bound = largest_y_below(lower_bound);

        // const auto to_check = VectorXu::LinSpaced(j - lower_bound, lower_bound,  j - 1);
        // to_add(to_check).array() = x_sorted_idxs(to_check).array() < j_val;

        // Eigen::Index i = 0;
        // for (; i < j - lower_bound; ++i) {
        //     const Eigen::Index i_val = x_sorted_idxs(j - i - 1);
        //     if (to_add(j - i - 1)) {
        //         to_add(j - i - 1) = max_y < i_val;
        //         max_y = max_y < i_val ? i_val : max_y;
        //     }
        // }

        // std::cout << j << " " << lower_bound << " " << j - lower_bound << " " << j - (i-1) - 1 << " " << y_sorted_points(j_val, Eigen::all) << std::endl;
        // std::cout << "j: " << j << ", Lower bound: " << lower_bound << ", j_val: " << j_val << ", compared: " << i << ", " << to_add.size() << std::endl;

        // cmps(j) = i;

        largest_y_below(j_val) = j;

        // to_add is populated with the largest values at the start
        // so add to the adjacency_matrix in reverse
        // for (size_t idx = 0; idx < to_add.size(); ++idx) {
        //     const auto row = to_add[to_add.size() - idx - 1];
        //     adjacency.insert(row, j) = 1;
        // }
        adjacency.col(j) = to_add.sparseView();
    }

    // std::cout << "Average compared " << cmps.mean() << std::endl;

    // TODO Is there some way to do this without all the comparisons again?
    // Correction for duplicate points
    bool has_equal_chain = false;
    Eigen::Index first_equal_index = 0;
    for (Eigen::Index j = 1; j < total_points; ++j) {
        const auto& p1 = y_sorted_points(x_sorted_idxs(j-1), Eigen::all);
        const auto& p2 = y_sorted_points(x_sorted_idxs(j), Eigen::all);

        if (!has_equal_chain && p1 == p2) {
            has_equal_chain = true;
            first_equal_index = j-1;
        } else if (has_equal_chain && p1 != p2) {
            has_equal_chain = false;
            adjacency.insert(j-1, first_equal_index) = 1;
        }
    }

    if (has_equal_chain) {
        adjacency.insert(total_points - 1, first_equal_index) = 1;
    }

    // Finalise Adjacency and Point Index Mappings
    adjacency.makeCompressed();
    gir::VectorXu idx_new = y_sorted_idxs(x_sorted_idxs);

    return std::make_tuple(
        std::move(adjacency),
        std::move(argsort(idx_new)),
        std::move(idx_new));
}

// For higher dimensions, we can perhaps apply the ideas from the paper
// Multidimensional Divide-and-conquer by Jon Louis Bentley.
// I think, we can precalculate the all points ecdf then the entries in
// the adjacency matrix for a given point, would be a subset of the points
// with the next lowest rank
//
// If one of the other 2d version above ends up being fast enough, it could
// be used for each plane before recombining, instead of recursing all
// the way down.
template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency_N_brute_force(const Eigen::MatrixX<V>& points) {
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
    VectorXu rev_degree_idxs = VectorXu::Zero(degree_idxs.rows());
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
    const auto dimensions = points.cols();

    // TODO specialised version for 1-d
    if (dimensions == 2) {
        return points_to_adjacency_2d_3(points);
    } else {
        return points_to_adjacency_N_brute_force(points);
    }
}

// Considering Recurisvely Cutting the space for better runtime in higher dimensions (very unfinished)
template<typename V>
void
points_to_adjacency_recursive_impl(
    // std::unordered_map<Eigen::Index, std::unordered_set<Eigen::Index>>& predecessors,
    Eigen::VectorX<bool>& predecessors,
    Eigen::SparseMatrix<bool>& adjacency,
    const Eigen::MatrixX<V>& points,
    const VectorXu& idxs
) {
    if (idxs.rows() <= 1) return;

    if (idxs.rows() == 2) {
        // TODO currently sorted by x so probably just have to check y.
        if ((points(idxs(0), Eigen::all).array() <= points(idxs(1), Eigen::all).array()).all()) {
            adjacency.insert(idxs(0), idxs(1)) = 1;
            // predecessors[idxs(1)] = std::move(std::unordered_set<Eigen::Index>{idxs(0)});
        } else if ((points(idxs(1), Eigen::all).array() <= points(idxs(0), Eigen::all).array()).all()) {
            adjacency.insert(idxs(1), idxs(0)) = 1;
            // predecessors[idxs(0)] = std::move(std::unordered_set<Eigen::Index>{idxs(1)});
        }
        return;
    }

    // std::cout << "\nidxs:\n" << idxs << std::endl;

    const uint64_t points_left = (idxs.rows() + 1) / 2;
    const uint64_t points_right = idxs.rows() / 2;
    VectorXu left_idxs = idxs(Eigen::seqN(0, points_left));
    VectorXu right_idxs = idxs(Eigen::seqN(points_left, points_right));

    // std::cout << "left:\n" << left_idxs << std::endl;
    // std::cout << "right:\n" << right_idxs << std::endl;

    points_to_adjacency_recursive_impl(predecessors, adjacency, points, left_idxs);
    points_to_adjacency_recursive_impl(predecessors, adjacency, points, right_idxs);

    // std::cout << "predecessors (" << predecessors.size() << "):\n";
    // for (auto const& pair: predecessors) {
    //     std::cout << "{" << pair.first << ": ";
    //     for (auto const& val: pair.second)
    //         std::cout << val << " ";
    //     std::cout << "}" << std::endl;
    // }

    // // should maybe make row major
    // // sorting of y could help
    // for (auto r_idx: right_idxs) {
    //     // std::vector<Eigen::Index> to_add;

    //     for (auto l_idx: left_idxs.reverse()) {
    //         if (points(l_idx, 1) <= points(r_idx, 1)) { // only need to compare y as was sorted by x

    //             bool should_add = true;
    //             for (
    //                 Eigen::SparseMatrix<bool>::ReverseInnerIterator it(adjacency, r_idx);
    //                 it;
    //                 --it
    //             ) {
    //                 if (it.value() && (points(l_idx, Eigen::all).array() <= points(it.row(), Eigen::all).array()).all()) {
    //                     should_add = false;
    //                     break;
    //                 }
    //             }

    //             if (should_add) adjacency.insert(l_idx, r_idx) = 1;

    //             // to_add.push_back(l_idx);

    //             // if (predecessors.count(r_idx)) {
    //             //     for (const auto predecessor: predecessors[r_idx]) {
    //             //         if (adjacency.coeff(l_idx, predecessor)) {
    //             //             should_add = false;
    //             //             break;
    //             //         }
    //             //     }

    //             //     predecessors[r_idx].insert(l_idx);
    //             // } else {
    //             //     predecessors[r_idx] = std::move(std::unordered_set<Eigen::Index>{l_idx});
    //             // }

    //             // if (should_add) adjacency.insert(l_idx, r_idx) = 1;
    //             // adjacency.insert(l_idx, r_idx) = 1;
    //         }
    //     }
    // }

    // Eigen::VectorX<bool> predecessors = Eigen::VectorX<bool>::Zero(points.rows());

    // should maybe make row major
    // sorting of y could help
    for (auto r_idx: right_idxs) {
        // predecessors = adjacency.col(r_idx).toDense();

        predecessors.setZero();
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(adjacency, r_idx);
            it;
            ++it
        ) {
            predecessors(it.row()) = it.value();
        }

        for (auto l_idx: left_idxs.reverse()) {
            if (points(l_idx, 1) <= points(r_idx, 1)) { // only need to compare y as was sorted by x
                predecessors(l_idx) = true;

                for (
                    Eigen::SparseMatrix<bool>::ReverseInnerIterator it(adjacency, r_idx);
                    it;
                    --it
                ) {
                    if (it.value() && (points(l_idx, Eigen::all).array() <= points(it.row(), Eigen::all).array()).all()) {
                        predecessors(l_idx) = false;
                        break;
                    }
                }

            }
        }

        adjacency.col(r_idx) = predecessors.sparseView();

    }
}

template<typename V>
std::tuple<Eigen::SparseMatrix<bool>, VectorXu, VectorXu>
points_to_adjacency_recursive(const Eigen::MatrixX<V>& points) {
    const uint64_t total_points = points.rows();
    const uint64_t dimensions = points.cols();

    // TODO could do a specialised version for 1d. Just need to sort
    // TODO trying just 2 dimensions first.
    // TODO do without recursion

    // sort by y to avoid help with merging later
    // VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    // std::sort(
    //     y_sorted_idxs.begin(),
    //     y_sorted_idxs.end(),
    //     [&points](const auto i, const auto j){
    //         Eigen::Index col = 0;
    //         while (col < points.cols()) {
    //             if ( points(i, points.cols() - 1 - col) != points(j, points.cols() - 1 - col) )
    //                 return points(i, points.cols() - 1 - col) < points(j, points.cols() - 1 - col);
    //             ++col;
    //         }
    //         return false;
    //     });

    VectorXu x_sorted_idxs = argsort(points);

    // std::unordered_map<Eigen::Index, std::unordered_set<Eigen::Index>> predecessors;
    Eigen::VectorX<bool> predecessors = Eigen::VectorX<bool>::Zero(points.rows());
    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    const Eigen::MatrixX<V> x_sorted_points = points(x_sorted_idxs, Eigen::all);
    points_to_adjacency_recursive_impl(
        predecessors,
        adjacency,
        x_sorted_points,
        VectorXu::LinSpaced(total_points, 0, total_points - 1));

    // std::cout << "adjacency:\n" << adjacency << std::endl;

    return std::make_tuple(
        std::move(adjacency),
        std::move(argsort(x_sorted_idxs)),
        std::move(x_sorted_idxs));
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

template<typename LossType>
std::pair<VectorXu, Eigen::VectorXd>
generalised_isotonic_regression(
    Eigen::SparseMatrix<bool> adjacency_matrix,
    Eigen::VectorXd y,
    Eigen::VectorXd weights,
    LossFunction<LossType> loss_fun,
    uint64_t max_iterations = 0
) {
    const uint64_t total_observations = y.rows();

    uint64_t group_count = 0;
    const double sentinal = 1e-30; // TODO handle properly

    // objective value of partitions that is used to decide
    // which cut to make at each iteration
    Eigen::VectorXd group_loss = Eigen::VectorXd::Zero(total_observations);

    // returned result
    VectorXu groups = VectorXu::Zero(total_observations);
    Eigen::VectorXd y_fit = Eigen::VectorXd::Zero(total_observations);

    // These iterations could potentially be done in parallel (except the first)
    for (uint64_t iteration = 1; max_iterations == 0 || iteration < max_iterations; ++iteration) {
        auto [max_cut_value, max_cut_idx] = argmax(group_loss);

        if (max_cut_value == sentinal) {
            break;
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

        } else if (no_constraints) {
            group_loss(considered_idxs).array() = sentinal;

            for (auto idx : considered_idxs) {
                y_fit(idx) = loss_fun.estimator(
                        y(idx, Eigen::all), weights(idx, Eigen::all));
                groups(idx) = ++group_count;
            }
            // y_fit(considered_idxs).array() = estimator;
            // groups(considered_idxs).array() = ++group_count;

        } else {
            const auto solution =
                minimum_cut(adjacency_matrix, derivative, considered_idxs); // TODO could pass num constrains in as calculating twice
            auto [left, right] = argpartition(solution);

            const bool no_cut = left.rows() == 0 || right.rows() == 0;

            if (no_cut) {
                group_loss(considered_idxs).array() = sentinal;
                y_fit(considered_idxs).array() = estimator;

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
            }
        }
    }

    return std::make_pair(std::move(groups), std::move(y_fit));
}

} // namespace gir
