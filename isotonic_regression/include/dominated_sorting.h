#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <iostream> // TODO remove
#include <tuple>
#include <set>
#include <utility>

#include "utility.h"

namespace gir {

template<typename V>
void
total_dominated_by_impl(
    const Eigen::MatrixX<V>& points,
    VectorXu& dominated_by,
    const VectorXu& idxs
) {

    if (idxs.rows() <= 1) {
        return;
    }

    // TODO could do with less looping
    const double cut_val = median(points(idxs, 0));
    const bool equal_min = cut_val == points(idxs, 0).minCoeff();
    const bool equal_max = cut_val == points(idxs, 0).maxCoeff();

    if (equal_min && equal_max) {
        // 1 axis is entirely equal and the other is sorted
        // so just need to run through the points, and add
        // to each the number of predecessors
        u_int64_t total = 0;
        for (Eigen::Index i = 1; i < idxs.rows(); ++i) {
            ++total;
            dominated_by[idxs(i)] += total;
        }

        return;
    }

    VectorXu smaller;
    VectorXu larger;

    if (equal_max) {

        const auto [_smaller, _larger] = argpartition(points(idxs, 0).array() >= cut_val);
        smaller = idxs(_smaller);
        larger = idxs(_larger);

    } else {

        const auto [_smaller, _larger] = argpartition(points(idxs, 0).array() > cut_val);
        smaller = idxs(_smaller);
        larger = idxs(_larger);

    }

    std::cout << "points:\n" << points(idxs, Eigen::all) << std::endl;
    std::cout << "cut_val: " << cut_val << std::endl;
    std::cout << "smaller:\n" << points(smaller, Eigen::all) << std::endl;
    std::cout << "larger:\n" << points(larger, Eigen::all) << std::endl;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n" << std::endl;

    total_dominated_by_impl(points, dominated_by, smaller);
    total_dominated_by_impl(points, dominated_by, larger);

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

    Eigen::Index i = 0;
    Eigen::Index j = 0;
    u_int64_t total = 0;

    while (i < smaller.rows() && j < larger.rows()) {
        if (smaller(i) <= larger(j)) {
            ++total;
            ++i;
        } else if (smaller(i) > larger(j)) {
            dominated_by[larger(j)] += total;
            ++j;
        }
    }

    while(j < larger.rows()) {
        dominated_by[larger(j)] += total;
        ++j;
    }

    std::cout << "dominated_by:\n" << dominated_by << std::endl;

}

// TODO add dominates and dominated by
// TODO fininsh for more than 2 dimensions
template<typename V>
std::tuple<VectorXu, VectorXu, VectorXu>
total_dominated_by(
    const Eigen::MatrixX<V>& points
) {

    // TODO filter duplicate points.

    // without inverting we would end up with number of points dominated
    // instead of pareto fronts

    const uint64_t total_points = points.rows();
    VectorXu dominated_by = VectorXu::Zero(total_points);
    VectorXu idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);

    // sort by y to speed up comparison checks
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            if (points(i, 1) == points(j, 1))
                return points(i, 0) > points(j, 0);
            return points(i, 1) > points(j, 1);
        });

    const Eigen::MatrixX<V> y_sorted_points = -1 * points(y_sorted_idxs, Eigen::all).array();

    std::cout << "y_sorted_points:\n" << y_sorted_points << std::endl;

    total_dominated_by_impl(y_sorted_points, dominated_by, idxs);

    // this should be then sorted by rank?
    gir::VectorXu idx_new = y_sorted_idxs.reverse();

    return std::make_tuple(
        dominated_by.reverse(),
        argsort(idx_new),
        idx_new
    );
}

// From Paper Generalizing the Improved Run-Time Complexity
// Algorithm for Non-Dominated Sorting
template<typename V>
std::tuple<VectorXu, VectorXu, VectorXu, VectorXu>
split_b(
    const Eigen::MatrixX<V>& points,
    const VectorXu& L,
    const VectorXu& H,
    const Eigen::Index k
) {

    double cut_val;

    if (L.rows() > H.rows()) {
        cut_val = median(points(L, k));
    } else {
        cut_val = median(points(H, k));
    }

    const auto& [L1l, L2l] = argpartition(
        points(L, k).array() > cut_val
    );

    const auto& [L1r, L2r] = argpartition(
        points(L, k).array() >= cut_val
    );

    const auto& [H1l, H2l] = argpartition(
        points(H, k).array() > cut_val
    );

    const auto& [H1r, H2r] = argpartition(
        points(H, k).array() >= cut_val
    );

    if ((L1r.rows() + H1r.rows()) <= (L2l.rows() + H2l.rows())) {
        return std::make_tuple(
            std::move(L(L1l)),
            std::move(L(L2l)),
            std::move(H(H1l)),
            std::move(H(H2l))
        );
    } else {
        return std::make_tuple(
            std::move(L(L1r)),
            std::move(L(L2r)),
            std::move(H(H1r)),
            std::move(H(H2r))
        );
    }
}

template<typename V>
void
sweep_b(
    const Eigen::MatrixX<V>& points,
    const VectorXu& L,
    const VectorXu& H,
    VectorXu& pareto_rank
) {
    VectorXu T(0);
    Eigen::Index i = 1;

    for (Eigen::Index j = 0; j < H.rows(); ++j) {

        while (
            i <= L.rows() &&
            (points(L(i), VectorXu::LinSpaced(2, 0, 1)).array() <= points(H(j), VectorXu::LinSpaced(2, 0, 1)).array()).all()
        ) {

            const auto& R = T(find(
                (pareto_rank(T).array() == pareto_rank(L(i))) &&
                (points(T, 1).array() < points(L(i), 1))
            ));

            if (R.rows() == 0) {

                const auto& nonequal_rank = find(
                    pareto_rank(T).array() != pareto_rank(L(i))
                );

                VectorXu newT(nonequal_rank.rows() + 1);
                newT(VectorXu::LinSpaced(nonequal_rank.rows(), 0, nonequal_rank.rows()-1)) = T(nonequal_rank);
                newT(newT.rows() - 1) = L(i);

                T = newT;
            }

            ++i;
        }

        const auto& U = T(find(
            points(T, 1).array() <= points(H(j), 1)
        ));

        if (U.rows() > 0) {
            const auto r = pareto_rank(U).maxCoeff();
            pareto_rank(H(j)) = std::max(pareto_rank(H(j)), r + 1);
        }
    }
}

template<typename V>
void
n_d_helper_b(
    const Eigen::MatrixX<V>& points,
    const VectorXu& L,
    const VectorXu& H,
    const Eigen::Index k, // 0-indexed
    VectorXu& pareto_rank
) {

    if (L.rows() == 0 || H.rows() == 0) {
        return;

    } else if (L.rows() == 1 || H.rows() == 1) {
        for (const auto l : L) {
            for (const auto h: H) {
                if (
                    (points(l, VectorXu::LinSpaced(k + 1, 0, k)).array() <= // or < ?
                     points(h, VectorXu::LinSpaced(k + 1, 0, k)).array()
                    ).all()
                ) {
                    pareto_rank(h) = std::max(
                        pareto_rank(h),
                        pareto_rank(l) + 1
                    );
                }
            }
        }

    } else if (k == 1) { // so the last 2 and the first is sorted
        sweep_b(points, L, H, pareto_rank);

    } else if (points(L, k).maxCoeff() <= points(H, k).minCoeff()) {
        n_d_helper_b(points, L, H, k-1, pareto_rank);

    } else if (points(L, k).minCoeff() <= points(H, k).maxCoeff()) {
        const auto& [L1, L2, H1, H2] = split_b(points, L, H, k);
        n_d_helper_b(points, L1, H1, k, pareto_rank);
        n_d_helper_b(points, L1, H2, k-1, pareto_rank);
        n_d_helper_b(points, L2, H2, k, pareto_rank);

    }
}

template<typename V>
std::tuple<VectorXu, VectorXu>
split_a(
    const Eigen::MatrixX<V>& points,
    const VectorXu& S,
    const Eigen::Index k
) {
    const double cut_val = median(points(S, k));

    // cut_val points in _L
    const auto& [L1, H1] = argpartition(
        points(S, k).array() > cut_val
    );

    // cut_val points in _H
    const auto& [L2, H2] = argpartition(
        points(S, k).array() >= cut_val
    );

    if (L2.rows() <= H1.rows()) {
        return std::make_tuple(
            std::move(S(L1)),
            std::move(S(H1))
        );
    } else {
        return std::make_tuple(
            std::move(S(L2)),
            std::move(S(H2))
        );
    }
}

template<typename V>
void
sweep_a(
    const Eigen::MatrixX<V>& points,
    const VectorXu& S,
    VectorXu& pareto_rank
) {
    VectorXu T(1);
    T << S(0);

    for (Eigen::Index i = 1; i < S.rows(); ++i) {
        //std::cout << "T:\n" << T << std::endl;

        const auto& U = T(find(
            points(T, 1).array() <= points(S(i), 1)
        ));

        //std::cout << "U:\n" << U << std::endl;

        if (U.rows() > 0) {
            const auto r = pareto_rank(U).maxCoeff();
            pareto_rank(S(i)) = std::max(pareto_rank(S(i)), r + 1);
        }

        const auto& nonequal_rank = find(
            pareto_rank(T).array() != pareto_rank(S(i))
        );

        VectorXu newT(nonequal_rank.rows() + 1);
        newT(VectorXu::LinSpaced(nonequal_rank.rows(), 0, nonequal_rank.rows()-1)) = T(nonequal_rank);
        newT(newT.rows() - 1) = S(i);

        T = newT;
    }
}

template<typename V>
void
n_d_helper_a(
    const Eigen::MatrixX<V>& points,
    const VectorXu& S,
    const Eigen::Index k, // 0-indexed
    VectorXu& pareto_rank
) {
    std::cout << "nda: k: " << k << " and S:\n" << S << std::endl;
    if (S.rows() < 2) {
        std::cout << "nda: |S| < 2" << std::endl;
        return;

    } else if (S.rows() == 2) {
        std::cout << "nda: |S| == 2" << std::endl;
        if (
            (points(S(0), VectorXu::LinSpaced(k + 1, 0, k)).array() < // or <= ?
             points(S(1), VectorXu::LinSpaced(k + 1, 0, k)).array()
            ).all()
        ) {
            std::cout << "nda: s_1:k^0 == s_1:k^1" << std::endl;
            pareto_rank(S(1)) = std::max(
                pareto_rank(S(1)),
                pareto_rank(S(0)) + 1
            );
        }

    } else if (k == 1) { // so the last 2 and the first is sorted
        std::cout << "nda: k == 1" << std::endl;
        sweep_a(points, S, pareto_rank);

    } else if ((points(S, k).array() == points(S(0), k)).all()) {
        std::cout << "nda: |s_k| == 1" << std::endl;
        // all points along axis are identical, so move to next
        n_d_helper_a(points, S, k-1, pareto_rank);

    } else {
        std::cout << "nda: else" << std::endl;

        const auto& [L, H] = split_a(points, S, k);
        n_d_helper_a(points, L, k, pareto_rank);
        n_d_helper_b(points, L, H, k-1, pareto_rank);
        n_d_helper_a(points, H, k, pareto_rank);

    }
}

template<typename V>
std::tuple<VectorXu, VectorXu, VectorXu>
non_dominated_sort(
    const Eigen::MatrixX<V>& points
) {

    const uint64_t total_points = points.rows();
    VectorXu pareto_rank = VectorXu::Zero(total_points);
    VectorXu idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);

    // sort lexicographically
    VectorXu y_sorted_idxs = VectorXu::LinSpaced(total_points, 0, total_points - 1);
    std::sort(
        y_sorted_idxs.begin(),
        y_sorted_idxs.end(),
        [&points](const auto& i, const auto& j) {
            for (Eigen::Index k = 0; k < points.cols(); ++k) {
                if (points(i, k) != points(j, k)) {
                    return points(i, k) <= points(j, k);
                }
            }
            return true;
        });

    const Eigen::MatrixX<V> y_sorted_points = points(y_sorted_idxs, Eigen::all);
    std::cout << "y_sorted_points:\n" << y_sorted_points << std::endl;

    n_d_helper_a(y_sorted_points, idxs, y_sorted_points.cols() - 1, pareto_rank);

    return std::make_tuple(
        pareto_rank,
        argsort(y_sorted_idxs),
        y_sorted_idxs
    );
}


} // namespace gir
