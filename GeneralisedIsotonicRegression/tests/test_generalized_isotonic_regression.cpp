#include "generalized_isotonic_regression.h"

#include <catch2/catch_test_macros.hpp>


template<typename V>
void REQUIRE_EQUAL(const Eigen::VectorX<V>& a, const Eigen::VectorX<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

    REQUIRE( equal_sizes );

    REQUIRE( (a.array() == b.array()).all() );
}

template<typename V>
void REQUIRE_EQUAL(const Eigen::SparseMatrix<V>& a, const Eigen::SparseMatrix<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

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
}

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

TEST_CASE( "adjacency_to_LP_standard_form", "[isotonic_regression]" ) {
    SECTION( "Simple Case 1" ) {
        /*
         * . x . . .
         * . . x . .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> adjacency_matrix(5, 5);

        adjacency_matrix.insert(0, 1) = true;
        adjacency_matrix.insert(1, 2) = true;
        adjacency_matrix.insert(2, 3) = true;
        adjacency_matrix.insert(3, 4) = true;
        adjacency_matrix.makeCompressed();

        /*
         *  1  0  0  0  0 -1  0  0  0  0  1  0  0  0
         *  0  1  0  0  0  0 -1  0  0  0 -1  1  0  0
         *  0  0  1  0  0  0  0 -1  0  0  0 -1  1  0
         *  0  0  0  1  0  0  0  0 -1  0  0  0 -1  1
         *  0  0  0  0  1  0  0  0  0 -1  0  0  0 -1
         */
        Eigen::SparseMatrix<int> expected(5, 14);
        for (int i = 0; i < 5; ++i) {
            expected.insert(i, i) = 1;
            expected.insert(i, i + 5) = -1;
        }
        expected.insert(0, 10) =  1;
        expected.insert(1, 10) = -1;
        expected.insert(1, 11) =  1;
        expected.insert(2, 11) = -1;
        expected.insert(2, 12) =  1;
        expected.insert(3, 12) = -1;
        expected.insert(3, 13) =  1;
        expected.insert(4, 13) = -1;
        expected.makeCompressed();

        auto result = adjacency_to_LP_standard_form(adjacency_matrix);

        REQUIRE_EQUAL(expected, result);
    }

    SECTION( "Simple Case 2" ) {
        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> adjacency_matrix(5, 5);
        adjacency_matrix.insert(0, 1) = true;
        adjacency_matrix.insert(0, 2) = true;
        adjacency_matrix.insert(1, 3) = true;
        adjacency_matrix.insert(2, 3) = true;
        adjacency_matrix.insert(3, 4) = true;
        adjacency_matrix.makeCompressed();

        /*
         * network by column:
         *  - node 1 to node 2
         *  - node 1 to node 3
         *  - node 2 to node 4
         *  - node 3 to node 4
         *  - node 4 to node 5
         * which correspond to the adjacency matrix above
         *
         * |   source    |     sink     |   network    |
         *  1  0  0  0  0 -1  0  0  0  0  1  1  0  0  0
         *  0  1  0  0  0  0 -1  0  0  0 -1  0  1  0  0
         *  0  0  1  0  0  0  0 -1  0  0  0 -1  0  1  0
         *  0  0  0  1  0  0  0  0 -1  0  0  0 -1 -1  1
         *  0  0  0  0  1  0  0  0  0 -1  0  0  0  0 -1
         */
        Eigen::SparseMatrix<int> expected(5, 15);
        for (int i = 0; i < 5; ++i) {
            expected.insert(i, i) = 1;
            expected.insert(i, i + 5) = -1;
        }
        expected.insert(0, 10) =  1;
        expected.insert(0, 11) =  1;
        expected.insert(1, 10) = -1;
        expected.insert(1, 12) =  1;
        expected.insert(2, 11) = -1;
        expected.insert(2, 13) =  1;
        expected.insert(3, 12) = -1;
        expected.insert(3, 13) = -1;
        expected.insert(3, 14) =  1;
        expected.insert(4, 14) = -1;
        expected.makeCompressed();

        auto network_flow_matrix = adjacency_to_LP_standard_form(adjacency_matrix);

        REQUIRE_EQUAL(expected, network_flow_matrix);
    }
}

TEST_CASE( "minimum_cut", "[isotonic_regression]" ) {
    /*
     * . x . . .
     * . . x . .
     * . . . x .
     * . . . . x
     * . . . . .
     */
    Eigen::SparseMatrix<bool> adjacency_matrix(5, 5);

    adjacency_matrix.insert(0, 1) = true;
    adjacency_matrix.insert(1, 2) = true;
    adjacency_matrix.insert(2, 3) = true;
    adjacency_matrix.insert(3, 4) = true;
    adjacency_matrix.makeCompressed();

    VectorXu idxs(5);
    idxs << 0, 1, 2, 3, 4;

    SECTION( "All 0 Loss" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 1, 1, 1;
        z << 0, 0, 0, 0, 0;

        Eigen::VectorX<bool> expected(5);
        expected << true, true, true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Simple Split (Right)" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 1, 1, 2
        z << -0.25, -0.25, -0.25, -0.25, 1;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, false, false, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Simple Split (Left)" ) {
        Eigen::VectorXd z(5); // y = 0.5, 1, 1, 1, 1
        z << -1, 0.25, 0.25, 0.25, 0.25;

        Eigen::VectorX<bool> expected(5);
        expected << false, true, true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Split Mid Point" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << -1, -1, 0.667, 0.667, 0.667;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Split Mid Point 2" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 1.5, 1.2;
        z << -0.51, -0.51, 1, 0.24, -0.21;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Reverse" ) {
        Eigen::VectorXd z(5); // y = 2, 2, 2, 1, 1;
        z << 0.667, 0.667, 0.667, -1, -1;

        Eigen::VectorX<bool> expected(5);
        expected << true, true, true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Index Subset 1" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << -1, -1, 0.667, 0.667, 0.667;

        VectorXu idxs(3);
        idxs << 1, 2, 3;

        Eigen::VectorX<bool> expected(3);
        expected << false, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Index Subset 2" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << -1, -1, 0.667, 0.667, 0.667;

        VectorXu idxs(3);
        idxs << 2, 3, 4;

        Eigen::VectorX<bool> expected(3);
        expected << true, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };
}

TEST_CASE( "generalised_isotonic_regression", "[isotonic_regression]" ) {
    SECTION( "Simple 3 Groups" ) {
        /*
         * . x . . .
         * . . x . .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::SparseMatrix<bool> adjacency_matrix(5, 5);
        adjacency_matrix.insert(0, 1) = true;
        adjacency_matrix.insert(1, 2) = true;
        adjacency_matrix.insert(2, 3) = true;
        adjacency_matrix.insert(3, 4) = true;
        adjacency_matrix.makeCompressed();

        Eigen::VectorXd y(5);
        y << 1, 1, 3, 5, 5;

        VectorXu expected_groups(5);
        expected_groups << 1, 1, 2, 3, 3;

        Eigen::VectorXd expected_y_fit(5);
        expected_y_fit << 1, 1, 3, 5, 5;

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y,
            LossFunction::L2);

        REQUIRE_EQUAL(expected_groups, groups);
        REQUIRE_EQUAL(expected_y_fit, y_fit);
    };
}
