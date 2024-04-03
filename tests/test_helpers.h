#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

template<typename V>
void REQUIRE_EQUAL(const Eigen::VectorX<V>& a, const Eigen::VectorX<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();
    if (!equal_sizes) {
        std::cout << "Unequal Vector Sizes: (" << a.rows() << ", " << a.cols();
        std::cout << ") != (" << b.rows() << ", " << b.cols() << ")" << std::endl;
    }

    REQUIRE( equal_sizes );

    REQUIRE( (a.array() == b.array()).all() );
}

template<typename V>
void REQUIRE_EQUAL(const Eigen::MatrixX<V>& a, const Eigen::MatrixX<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

    if (!equal_sizes) {
        std::cout << "Unequal Matrix Sizes: (" << a.rows() << ", " << a.cols();
        std::cout << ") != (" << b.rows() << ", " << b.cols() << ")" << std::endl;
    }

    REQUIRE( equal_sizes );

    REQUIRE( (a.cwiseEqual(b)).all() );
}

template<typename V>
void REQUIRE_EQUAL(const Eigen::SparseMatrix<V>& a, const Eigen::SparseMatrix<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

    if (!equal_sizes) {
        std::cout << "Unequal SparseMatrix Sizes: (" << a.rows() << ", " << a.cols();
        std::cout << ") != (" << b.rows() << ", " << b.cols() << ")" << std::endl;
    }

    REQUIRE( equal_sizes );

    if (!equal_sizes) {
        return;
    }

    for (size_t j = 0; j < a.outerSize(); ++j) {
        for (
            typename Eigen::SparseMatrix<V>::InnerIterator it(a, j);
            it;
            ++it
        ) {
            REQUIRE( it.value() == b.coeff(it.row(), it.col()) );
        }
    }

    for (size_t j = 0; j < a.outerSize(); ++j) {
        for (
            typename Eigen::SparseMatrix<V>::InnerIterator it(b, j);
            it;
            ++it
        ) {
            REQUIRE( it.value() == a.coeff(it.row(), it.col()) );
        }
    }
}
