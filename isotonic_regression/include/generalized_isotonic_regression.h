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

// Other non-recursive idea for faster 2-d
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

    // TODO broken for dimensions != 2
    // and probably for a single point

    // sort by y to avoid help with merging later
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto i, const auto j){
            Eigen::Index col = 0;
            while (col < points.cols()) {
                if ( points(i, points.cols() - 1 - col) != points(j, points.cols() - 1 - col) )
                    return points(i, points.cols() - 1 - col) < points(j, points.cols() - 1 - col);
                ++col;
            }
            return false;
        });

    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);

    // work on indices sorted by x
    const VectorXu x_sorted_idxs = argsort(y_sorted_idxs);

    // std::cout << "points:\n" << y_sorted_points << std::endl;
    // std::cout << "idxs:\n" << x_sorted_idxs << std::endl;

    // this is currently ordered accoreding to increasing x
    Eigen::SparseMatrix<bool, Eigen::ColMajor> adjacency(total_points, total_points); // Column Major
    // each column contains 1s in each row where the corresponding point is smaller
    // so we can upperbound the amount of 1s in each columns by the number of points
    // smaller in the y-dimension (contents of x_sorted_idxs) and the number smaller
    // in the x-dimension (index in x_sorted_idxs)
    //adjacency.reserve();
    // const auto to_reserve = Eigen::VectorXi::LinSpaced(total_points, 0, total_points-1).cwiseMax(10).cwiseMin(x_sorted_idxs.cast<int>());

    // std::cout << "to_reserve size: " << to_reserve.rows() << " " << to_reserve.cols() << ", to_reserve max: " << to_reserve.maxCoeff() << ", to_reserve min: " << to_reserve.minCoeff() << std::endl;

    adjacency.reserve(Eigen::VectorXi::Constant(total_points, 30)); // there seems to be problems reserving large amounts for specific columns
    // adjacency.reserve(total_points);                             // however the reservation also seems to change almost nothing timewise

    std::cout << "Finished reserving matrix" << std::endl;

    std::vector<Eigen::Index> to_add;
    to_add.reserve(total_points);

    Eigen::Index maxcol = 0;
    Eigen::Index maxrow = 0;
    Eigen::Index minrow = 10000000000000000;

    std::cout << "x_sorted size " << x_sorted_idxs.rows() << std::endl;

    VectorXu largest_y_below(total_points);

    // TODO the way this is now written could also just directly insert into
    // std::vector and use the insert from triplet method. might be faster?
    for (Eigen::Index j = 1; j < total_points; ++j) {
        to_add.clear();
        Eigen::Index max_y = -1;
        const Eigen::Index j_val = x_sorted_idxs(j);

        const auto lower_bound = largest_y_below(VectorXu::LinSpaced(j_val, 0, j_val - 1)).maxCoeff();

        for (Eigen::Index i = 0;  i < j; ++i) {
            const Eigen::Index i_val = x_sorted_idxs(j - i - 1);
            if (max_y < i_val && i_val < j_val) {
                to_add.push_back(j - i - 1);
                max_y = i_val;
                if (max_y >= lower_bound) break; // TODO is this valid logic?...
            }
        }

        largest_y_below(j_val) = max_y;

        for (size_t idx = 0; idx < to_add.size(); ++idx) {
            const auto row = to_add[to_add.size() - idx - 1];

            // if (j > maxcol) {
            //     maxcol = j;
            //     std::cout << "row: " << minrow << " -> " << maxrow << ", col: " << maxcol << ", out of " << total_points << ", with to add " << to_add.size() << std::endl;
            // }

            // if (row < minrow) {
            //     minrow = row;
            //     std::cout << "row: " << minrow << " -> " << maxrow << ", col: " << maxcol << ", out of " << total_points << ", with to add " << to_add.size() << std::endl;
            // }

            // if (row > maxrow) {
            //     maxrow = row;
            //     std::cout << "row: " << minrow << " -> " << maxrow << ", col: " << maxcol << ", out of " << total_points << ", with to add " << to_add.size() << std::endl;
            // }

            adjacency.insert(row, j) = 1;
        }

        // for(auto itr = to_add.rbegin(); itr != to_add.rend(); ++itr) {
        //     adjacency.insert(*itr, j) = 1;
        // }
    }

    // // - 1 as don't do anything with just the last entry when odd number of points
    // uint64_t window_size = 2;
    // for (Eigen::Index window_start = 0; window_start < total_points - 1; window_start += window_size) {
    //     if (x_sorted_idxs(window_start) < x_sorted_idxs(window_start + 1))
    //         adjacency.insert(window_start, window_start + 1) = 1;
    // }

    // // correct skipped point in the case of an odd number of points
    // if (total_points % 2) {
    //     if (x_sorted_idxs(total_points - 2) < x_sorted_idxs(total_points - 1)) adjacency.insert(total_points-2, total_points - 1);
    //     if (x_sorted_idxs(total_points - 3) < x_sorted_idxs(total_points - 1) && x_sorted_idxs(total_points - 3) > x_sorted_idxs(total_points - 2))
    //         adjacency.insert(total_points-3, total_points - 1);
    // }

    // // next we go through in increasing window sizes and correct the adjacency matrix by adding the missing dominated points
    // window_size *= 2;
    // for (Eigen::Index window_start = 0; window_start < total_points - 1; window_start += window_size) {
    //     const bool odd_point = window_start + window_size >= total_points - 1 && total_points % 2;
    //     const auto right_idxs = VectorXu::LinSpaced(window_size / 2 + odd_point, window_start + window_size / 2, window_start + window_size + odd_point - 1);

    //     // going from rightmost to leftmost point, x decreases. if y also decreases, that means the subsequent
    //     // point is smaller and so will have been already included as a predecessor of another point
    //     // therefore, we are only interested in those points where y increases above the current maximum.
    //     // VectorXu left_idxs = VectorXu::LinSpaced(window_size / 2, window_start, window_start + window_size / 2 - 1);
    //     std::vector<Eigen::Index> left_points;
    //     Eigen::Index max_y = x_sorted_idxs(window_start + window_size / 2 - 1);
    //     left_points.push_back(window_start + window_size / 2 - 1);
    //     for (Eigen::Index idx = window_start + window_size / 2 - 2; idx > window_start; --idx) {
    //         if (x_sorted_idxs(idx) > max_y) {
    //             left_points.push_back(idx);
    //             max_y = x_sorted_idxs(idx);
    //         }
    //     }
    //     const auto left_idxs = VectorXu::Map(&left_points[0], left_points.size());
    //     // VectorXu left_idxs(left_points.rbegin(), left_points.rend());

    //     std::cout << "left_idxs:\n" << left_idxs << std::endl;
    //     std::cout << "right_idxs:\n" << right_idxs << std::endl;

    //     // for (Eigen::Index right_point: right_idxs) {

    //     // }

    //     // const auto& previous_points = sorted_points(Eigen::all,
    //     //     VectorXu::LinSpaced(i, 0, i-1)).array();
    //     // const auto& current_point = sorted_points(Eigen::all,
    //     //     VectorXu::LinSpaced(i, i, i)).array();

    //     // is_predecessor(Eigen::seq(0, i-1)) = (
    //     //     previous_points <= current_point).colwise().all();

    //     // degree(i) = is_predecessor.count();
    // }

    adjacency.makeCompressed();

    std::cout << "total entries: " << adjacency.nonZeros() << std::endl;
    // std::cout << "adjacency:\n" << adjacency << std::endl;
    // std::cout << "first\n" << argsort(x_sorted_idxs) << std::endl;
    // std::cout << "second\n" << x_sorted_idxs << std::endl;

    return std::make_tuple(
        std::move(adjacency),
        std::move(argsort(x_sorted_idxs)), // TODO these aren't right anymore
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
