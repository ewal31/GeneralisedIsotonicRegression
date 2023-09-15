#include "generalized_isotonic_regression.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

template<typename V>
void REQUIRE_EQUAL(const Eigen::VectorX<V>& a, const Eigen::VectorX<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

    REQUIRE( equal_sizes );

    REQUIRE( (a.array() == b.array()).all() );
}

template<typename V>
void REQUIRE_EQUAL(const Eigen::MatrixX<V>& a, const Eigen::MatrixX<V>& b) {
    bool equal_sizes = a.rows() == b.rows() && a.cols() == b.cols();

    REQUIRE( equal_sizes );

    REQUIRE( (a.cwiseEqual(b)).all() );
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

TEST_CASE( "is_monotonic", "[isotonic_regression]" ) {
    SECTION( "adjacency matrix (monotonic 1)" ) {
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

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 3, 5;
        REQUIRE( is_monotonic(adjacency_matrix, y) );
    }

    SECTION( "adjacency matrix (monotonic 2)" ) {
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

        Eigen::VectorX<double> y(5);
        y << 0, 2, 1, 2, 5;
        REQUIRE( is_monotonic(adjacency_matrix, y) );
    }

    SECTION( "adjacency matrix (not monotonic 1)" ) {
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

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 3, 2.9;
        REQUIRE( !is_monotonic(adjacency_matrix, y) );
    }

    SECTION( "adjacency matrix (not monotonic 2)" ) {
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

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 0.9, 5;
        REQUIRE( !is_monotonic(adjacency_matrix, y) );
    }

    SECTION( "points (monotonic 1)" ) {
        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::MatrixX<double> points(5, 2);
        points << 0, 0,
                  1, 2,
                  2, 1,
                  3, 3,
                  4, 4;

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 3, 5;
        REQUIRE( is_monotonic(points, y) );
    }

    SECTION( "points (monotonic 2)" ) {
        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::MatrixX<double> points(5, 2);
        points << 0, 0,
                  1, 2,
                  2, 1,
                  3, 3,
                  4, 4;

        Eigen::VectorX<double> y(5);
        y << 0, 2, 1, 3, 5;
        REQUIRE( is_monotonic(points, y) );
    }

    SECTION( "points (not monotonic 1)" ) {
        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::MatrixX<double> points(5, 2);
        points << 0, 0,
                  1, 2,
                  2, 1,
                  3, 3,
                  4, 4;

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 3, 2.9;
        REQUIRE( !is_monotonic(points, y) );
    }

    SECTION( "points (not monotonic 2)" ) {
        /*
         * . x x . .
         * . . . x .
         * . . . x .
         * . . . . x
         * . . . . .
         */
        Eigen::MatrixX<double> points(5, 2);
        points << 0, 0,
                  1, 2,
                  2, 1,
                  3, 3,
                  4, 4;

        Eigen::VectorX<double> y(5);
        y << 0, 1, 1, 0.9, 5;
        REQUIRE( !is_monotonic(points, y) );
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
    VectorXu considered_idxs(5);
    considered_idxs << 0, 1, 2, 3, 4;

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

        auto result = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

        auto result = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

        REQUIRE_EQUAL(expected, result);
    }

    SECTION( "Simple Case 2 (Subset 1)" ) {
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

        VectorXu considered_idxs(3);
        considered_idxs << 0, 1, 2;

        /*
         * |source |  sink  | network |
         *  1  0  0 -1  0  0  1  1
         *  0  1  0  0 -1  0 -1  0
         *  0  0  1  0  0 -1  0 -1
         */
        Eigen::SparseMatrix<int> expected(3, 8);
        for (int i = 0; i < 3; ++i) {
            expected.insert(i, i) = 1;
            expected.insert(i, i + 3) = -1;
        }
        expected.insert(0, 6) =  1;
        expected.insert(1, 6) =  -1;
        expected.insert(0, 7) =  1;
        expected.insert(2, 7) =  -1;
        expected.makeCompressed();

        auto result = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

        REQUIRE_EQUAL(expected, result);
    }

    SECTION( "Simple Case 2 (Subset 2)" ) {
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

        VectorXu considered_idxs(3);
        considered_idxs << 0, 2, 3;

        /*
         * |source |  sink  | network |
         *  1  0  0 -1  0  0  1  1
         *  0  1  0  0 -1  0 -1  0
         *  0  0  1  0  0 -1  0 -1
         */
        Eigen::SparseMatrix<int> expected(3, 8);
        for (int i = 0; i < 3; ++i) {
            expected.insert(i, i) = 1;
            expected.insert(i, i + 3) = -1;
        }
        expected.insert(0, 6) =  1;
        expected.insert(1, 6) =  -1;
        expected.insert(1, 7) =  1;
        expected.insert(2, 7) =  -1;
        expected.makeCompressed();

        auto result = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

        REQUIRE_EQUAL(expected, result);
    }

    SECTION( "Simple Case 3" ) {
        /*
         * . . x x .
         * . . . . x
         * . . . . .
         * . . . . .
         * . . . . .
         */
        Eigen::SparseMatrix<bool> adjacency_matrix(5, 5);
        adjacency_matrix.insert(0, 2) = true;
        adjacency_matrix.insert(0, 3) = true;
        adjacency_matrix.insert(1, 4) = true;
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
         * |   source    |     sink     | network |
         *  1  0  0  0  0 -1  0  0  0  0  1  1  0
         *  0  1  0  0  0  0 -1  0  0  0  0  0  1
         *  0  0  1  0  0  0  0 -1  0  0 -1  0  0
         *  0  0  0  1  0  0  0  0 -1  0  0 -1  0
         *  0  0  0  0  1  0  0  0  0 -1  0  0 -1
         */
        Eigen::SparseMatrix<int> expected(5, 13);
        for (int i = 0; i < 5; ++i) {
            expected.insert(i, i) = 1;
            expected.insert(i, i + 5) = -1;
        }
        expected.insert(0, 10) =  1;
        expected.insert(2, 10) = -1;
        expected.insert(0, 11) =  1;
        expected.insert(3, 11) = -1;
        expected.insert(1, 12) =  1;
        expected.insert(4, 12) = -1;
        expected.makeCompressed();

        auto result = adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

        REQUIRE_EQUAL(expected, result);
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
        z << -1, 0.667, 0.667;

        VectorXu idxs(3);
        idxs << 1, 2, 3;

        Eigen::VectorX<bool> expected(3);
        expected << false, true, true;

        auto result = minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Index Subset 2" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << 0.667, 0.667, 0.667;

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
        expected_groups << 0, 0, 1, 2, 2;

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

TEST_CASE( "full example", "[isotonic_regression]" ) {
    Eigen::MatrixXd points(5, 2);
    points << 0.762495,  0.392963,
              0.522416,  0.486052,
              0.796809,  0.0152406,
              0.979002,  0.271266,
              0.0711248, 0.310313;

    Eigen::VectorXd y(5);
    y << 0.301295,
         0.254428,
         0.0120925,
         0.265759,
         0.0222442;

    auto [adjacency_matrix, idx_original, idx_new] =
        points_to_adjacency(points);

    /*
        * . . x x .
        * . . . . x
        * . . . . .
        * . . . . .
        * . . . . .
        */
    Eigen::SparseMatrix<bool> expected_adjacency_matrix(5, 5);
    expected_adjacency_matrix.insert(0, 2) = true;
    expected_adjacency_matrix.insert(0, 3) = true;
    expected_adjacency_matrix.insert(1, 4) = true;
    expected_adjacency_matrix.makeCompressed();

    REQUIRE_EQUAL(expected_adjacency_matrix, adjacency_matrix);

    Eigen::MatrixXd sorted_points = points(idx_new, Eigen::all);
    Eigen::MatrixXd expected_sorted_points(5, 2);
    expected_sorted_points << 0.0711248, 0.310313,
                              0.796809,  0.0152406,
                              0.522416,  0.486052,
                              0.762495,  0.392963,
                              0.979002,  0.271266;

    REQUIRE_EQUAL(expected_sorted_points, sorted_points);

    Eigen::VectorXd sorted_y = y(idx_new);
    Eigen::VectorXd expected_sorted_y(5);
    expected_sorted_y << 0.0222442,
                         0.0120925,
                         0.254428,
                         0.301295,
                         0.265759;

    REQUIRE_EQUAL(expected_sorted_y, sorted_y);

    auto [groups, y_fit] = generalised_isotonic_regression(
        adjacency_matrix,
        y(idx_new),
        LossFunction::L2);

    VectorXu expected_groups(5);
    expected_groups << 0, 1, 2, 3, 4;

    REQUIRE_EQUAL(expected_groups, groups);
    REQUIRE_EQUAL(expected_sorted_y, y_fit);

    REQUIRE( is_monotonic(sorted_points, y_fit) );
    REQUIRE( is_monotonic(adjacency_matrix, y_fit) );
}

TEST_CASE( "random examples are monotonic", "[isotonic_regression]" ) {
    for (size_t iteration = 0; iteration < 10; ++iteration) {
        size_t dimensions = std::rand() % 10;

        auto [X, y] = generate_monotonic_points(1000, 0.01, dimensions);

        REQUIRE( X.rows() == y.rows() );
        REQUIRE( X.cols() == dimensions );
        REQUIRE( y.cols() == 1 );

        auto [adjacency_matrix, idx_original, idx_new] =
            points_to_adjacency(X);

        REQUIRE( X.rows() == adjacency_matrix.rows() );
        REQUIRE( X.rows() == adjacency_matrix.cols() );
        REQUIRE( X.rows() == idx_original.rows() );
        REQUIRE( X.rows() == idx_new.rows() );

        Eigen::MatrixXd sorted_points = X(idx_new, Eigen::all);
        Eigen::VectorXd sorted_ys = y(idx_new);

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            LossFunction::L2);

        REQUIRE( is_monotonic(sorted_points, y_fit) );
        REQUIRE( is_monotonic(adjacency_matrix, y_fit) );
    }
}
