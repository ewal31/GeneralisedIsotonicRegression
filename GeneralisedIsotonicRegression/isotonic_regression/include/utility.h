#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <utility>

using VectorXu = Eigen::VectorX<size_t>;


template<typename V>
VectorXu argsort(const Eigen::Matrix<V, Eigen::Dynamic, 1>& vec) {
    VectorXu idxs = VectorXu::LinSpaced(vec.size(), 0, vec.size() - 1);

    std::sort(
        idxs.begin(),
        idxs.end(),
        [&vec](const auto i, const auto j){
            return vec(i) < vec(j);
        });

    return idxs;
}

template<typename V>
VectorXu argsort(const Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    // TODO should rewrite to be outerindex-wise so it would work
    //      with both row and column major

    // rowwise lexicographic
    VectorXu idxs = VectorXu::LinSpaced(mat.rows(), 0, mat.rows() - 1);

    std::sort(
        idxs.begin(),
        idxs.end(),
        [&mat](const auto i, const auto j){
            size_t col = 0;
            while (col < mat.cols()) {
                if ( mat(i, col) != mat(j, col) )
                    return mat(i, col) < mat(j, col);
                ++col;
            }
            return false;
        });

    return idxs;
}

std::pair<VectorXu, VectorXu>
argpartition(const Eigen::VectorX<bool>& solution);

template<typename V>
std::pair<V, size_t>
argmax(Eigen::VectorX<V> vec) {
    auto max_val = vec.maxCoeff();
    auto location = std::find(vec.begin(), vec.end(), max_val);
    return std::make_pair(max_val, std::distance(vec.begin(), location));
}

template<typename V, typename UnaryPredicate>
VectorXu
find(Eigen::VectorX<V> vec, UnaryPredicate p) {
    VectorXu idxs = VectorXu::LinSpaced(vec.rows(), 0, vec.rows() - 1);
    auto partition_end = std::stable_partition(idxs.begin(), idxs.end(), p);
    idxs.conservativeResize(std::distance(idxs.begin(), partition_end));
    return idxs;
}
