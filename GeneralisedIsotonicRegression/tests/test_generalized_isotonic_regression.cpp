#include "generalized_isotonic_regression.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "generate_monotonic_points", "[isotonic_regression]" ) {
    SECTION( "1-dimensional" ) {
        auto [X, y] = generate_monotonic_points(5, 0.1, 1);

        REQUIRE( X.rows() == 5 );
        REQUIRE( X.cols() == 1 );
        REQUIRE( y.rows() == 5 );
        REQUIRE( y.cols() == 1 );
    }

    SECTION( "3-dimensional" ) {
        auto [X, y] = generate_monotonic_points(10, 0.1, 3);

        REQUIRE( X.rows() == 10 );
        REQUIRE( X.cols() == 3 );
        REQUIRE( y.rows() == 10 );
        REQUIRE( y.cols() == 1 );
    }

    SECTION( "is monotonic" ) {
        auto [X, y] = generate_monotonic_points(10, 0., 2);

        REQUIRE( X.rows() == 10 );
        REQUIRE( X.cols() == 2 );
        REQUIRE( y.rows() == 10 );
        REQUIRE( y.cols() == 1 );

        Eigen::VectorXi sorted_idxs = Eigen::VectorXi::LinSpaced(
            X.rows(),
            0,
            X.rows() - 1);

        // Sort by distance to 0
        std::sort(
            sorted_idxs.begin(),
            sorted_idxs.end(),
            [&X = X](const auto i, const auto j){
                return X.row(i).array().prod() <
                    X.row(j).array().prod();
            });

        for (auto i = 1; i < y.rows(); ++i) {
            REQUIRE( y(sorted_idxs(i-1)) <= y(sorted_idxs(i)) );
        }
    }
}

void REQUIRE_EQUAL(Eigen::SparseMatrix<bool> a, Eigen::SparseMatrix<bool> b) {
    for (size_t j = 0; j < a.outerSize(); ++j) {
        for (
            Eigen::SparseMatrix<bool>::InnerIterator it(a, j);
            it;
            ++it
        ) {
            REQUIRE( it.value() == b.coeffRef(it.row(), it.col()) );
        }
    }
}

TEST_CASE( "points_to_adjacency", "[isotonic_regression]" ) {
    SECTION( "Equal dimension" ) {
        Eigen::MatrixX<uint8_t> points(5, 2);
        points << 1, 1,
                  2, 1,
                  3, 1,
                  4, 1,
                  5, 1;

        /*
         * . x . . .
         * . . x . .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> expected_adjacency(5, 5);
        expected_adjacency.insert(0, 1) = true;
        expected_adjacency.insert(1, 2) = true;
        expected_adjacency.insert(2, 3) = true;
        expected_adjacency.insert(3, 4) = true;

        auto [adjacency_matrix, idx_original, idx_new] =
            points_to_adjacency(points);

        REQUIRE_EQUAL(expected_adjacency, adjacency_matrix);
    }

    SECTION( "1-dimensional" ) {
        Eigen::MatrixX<uint8_t> X(5, 1);
        X << 3,
             4,
             2,
             1,
             5;

        /*
         * . x . . .
         * . . x . .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> expected_adjacency(5, 5);
        expected_adjacency.insert(0, 1) = true;
        expected_adjacency.insert(1, 2) = true;
        expected_adjacency.insert(2, 3) = true;
        expected_adjacency.insert(3, 4) = true;

        auto [adjacency_matrix, idx_original, idx_new] =
            points_to_adjacency(X);

        REQUIRE_EQUAL(expected_adjacency, adjacency_matrix);
    }

    SECTION( "Multiple Paths" ) {
        Eigen::MatrixX<uint8_t> points(5, 2);
        points << 1, 1,
                  2, 3,
                  3, 2,
                  4, 4,
                  5, 5;

        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> expected_adjacency(5, 5);
        expected_adjacency.insert(0, 1) = true;
        expected_adjacency.insert(0, 2) = true;
        expected_adjacency.insert(1, 3) = true;
        expected_adjacency.insert(2, 3) = true;
        expected_adjacency.insert(3, 4) = true;

        auto [adjacency_matrix, idx_original, idx_new] =
            points_to_adjacency(points);

        REQUIRE_EQUAL(expected_adjacency, adjacency_matrix);
    }
}
