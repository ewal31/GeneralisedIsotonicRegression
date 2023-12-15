#include "generalized_isotonic_regression.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include <random>

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
        REQUIRE( gir::is_monotonic(adjacency_matrix, y) );
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
        REQUIRE( gir::is_monotonic(adjacency_matrix, y) );
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
        REQUIRE( !gir::is_monotonic(adjacency_matrix, y) );
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
        REQUIRE( !gir::is_monotonic(adjacency_matrix, y) );
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
        REQUIRE( gir::is_monotonic(points, y) );
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
        REQUIRE( gir::is_monotonic(points, y) );
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
        REQUIRE( !gir::is_monotonic(points, y) );
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
        REQUIRE( !gir::is_monotonic(points, y) );
    }
}


TEST_CASE( "generate_monotonic_points", "[isotonic_regression]" ) {
    SECTION( "1-dimensional" ) {
        auto [X, y] = gir::generate_monotonic_points(5, 0.1, 1);

        REQUIRE( X.rows() == 5 );
        REQUIRE( X.cols() == 1 );
        REQUIRE( y.rows() == 5 );
        REQUIRE( y.cols() == 1 );
    }

    SECTION( "3-dimensional" ) {
        auto [X, y] = gir::generate_monotonic_points(10, 0.1, 3);

        REQUIRE( X.rows() == 10 );
        REQUIRE( X.cols() == 3 );
        REQUIRE( y.rows() == 10 );
        REQUIRE( y.cols() == 1 );
    }

    SECTION( "is monotonic" ) {
        auto [X, y] = gir::generate_monotonic_points(10, 0., 2);

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
            gir::points_to_adjacency(points);

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
            gir::points_to_adjacency(X);

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
            gir::points_to_adjacency(points);

        REQUIRE_EQUAL(expected_adjacency, adjacency_matrix);
    }
}

TEST_CASE( "adjacency_to_LP_standard_form", "[isotonic_regression]" ) {
    gir::VectorXu considered_idxs(5);
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

        auto result = gir::adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

        auto result = gir::adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

        gir::VectorXu considered_idxs(3);
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

        auto result = gir::adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

        gir::VectorXu considered_idxs(3);
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

        auto result = gir::adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

        auto result = gir::adjacency_to_LP_standard_form(adjacency_matrix, considered_idxs);

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

    gir::VectorXu idxs(5);
    idxs << 0, 1, 2, 3, 4;

    SECTION( "Simple Split (Right)" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 1, 1, 2
        z << -0.25, -0.25, -0.25, -0.25, 1;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, false, false, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Simple Split (Left)" ) {
        Eigen::VectorXd z(5); // y = 0.5, 1, 1, 1, 1
        z << -1, 0.25, 0.25, 0.25, 0.25;

        Eigen::VectorX<bool> expected(5);
        expected << false, true, true, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Split Mid Point" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << -1, -1, 0.667, 0.667, 0.667;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, true, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Split Mid Point 2" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 1.5, 1.2;
        z << -0.51, -0.51, 1, 0.24, -0.21;

        Eigen::VectorX<bool> expected(5);
        expected << false, false, true, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Reverse" ) {
        Eigen::VectorXd z(5); // y = 2, 2, 2, 1, 1;
        z << 0.667, 0.667, 0.667, -1, -1;

        Eigen::VectorX<bool> expected(5);
        expected << true, true, true, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Index Subset 1" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << -1, 0.667, 0.667;

        gir::VectorXu idxs(3);
        idxs << 1, 2, 3;

        Eigen::VectorX<bool> expected(3);
        expected << false, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

        REQUIRE_EQUAL(expected, result);
    };

    SECTION( "Index Subset 2" ) {
        Eigen::VectorXd z(5); // y = 1, 1, 2, 2, 2;
        z << 0.667, 0.667, 0.667;

        gir::VectorXu idxs(3);
        idxs << 2, 3, 4;

        Eigen::VectorX<bool> expected(3);
        expected << true, true, true;

        auto result = gir::minimum_cut(adjacency_matrix, z, idxs);

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

        Eigen::VectorXd expected_y_fit(5);
        expected_y_fit << 1, 1, 3, 5, 5;

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y,
            Eigen::VectorXd::Constant(y.rows(), 1),
            gir::L2());

        REQUIRE_EQUAL(expected_y_fit, y_fit);

        // something like 0, 0, 1, 2, 2
        REQUIRE( (groups(0) == groups(1) && groups(1) != groups(2)) );
        REQUIRE( (groups(3) == groups(4) && groups(3) != groups(2)) );
        REQUIRE( groups(0) != groups(4) );
    };
}

TEST_CASE( "full example", "[isotonic_regression]" ) {
    SECTION("5 Points") {
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
            gir::points_to_adjacency(points);

        // /*
        //  * . . x x .
        //  * . . . . x
        //  * . . . . .
        //  * . . . . .
        //  * . . . . .
        //  */
        // Eigen::SparseMatrix<bool> expected_adjacency_matrix(5, 5);
        // expected_adjacency_matrix.insert(0, 2) = true;
        // expected_adjacency_matrix.insert(0, 3) = true;
        // expected_adjacency_matrix.insert(1, 4) = true;
        // expected_adjacency_matrix.makeCompressed();

        // REQUIRE_EQUAL(expected_adjacency_matrix, adjacency_matrix);

        // Eigen::MatrixXd sorted_points = points(idx_new, Eigen::all);
        // Eigen::MatrixXd expected_sorted_points(5, 2);
        // expected_sorted_points << 0.0711248, 0.310313,
        //                           0.796809,  0.0152406,
        //                           0.522416,  0.486052,
        //                           0.762495,  0.392963,
        //                           0.979002,  0.271266;

        // REQUIRE_EQUAL(expected_sorted_points, sorted_points);

        // Eigen::VectorXd sorted_y = y(idx_new);
        // Eigen::VectorXd expected_sorted_y(5);
        // expected_sorted_y << 0.0222442,
        //                      0.0120925,
        //                      0.254428,
        //                      0.301295,
        //                      0.265759;

        // REQUIRE_EQUAL(expected_sorted_y, sorted_y);
        REQUIRE( adjacency_matrix.nonZeros() == 3 );
        REQUIRE( adjacency_matrix.coeff(idx_original(4), idx_original(0)) == true );
        REQUIRE( adjacency_matrix.coeff(idx_original(4), idx_original(1)) == true );
        REQUIRE( adjacency_matrix.coeff(idx_original(2), idx_original(3)) == true );

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            Eigen::VectorXd::Constant(y.rows(), 1),
            gir::L2());

        // REQUIRE_EQUAL(expected_sorted_y, y_fit);
        const Eigen::VectorXd reordered_y_fit = y_fit(idx_original, Eigen::all);

        REQUIRE_EQUAL( y, reordered_y_fit );
        REQUIRE( gir::is_monotonic(points, reordered_y_fit) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit) );

        // something like 0, 1, 2, 3, 4
        REQUIRE( std::unique(groups.begin(), groups.end()) == groups.end() );

        // REQUIRE( gir::is_monotonic(sorted_points, y_fit) );
        // REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit) );
    }

    SECTION("Already Monotonic") {
        uint64_t grid_width = 5;

        Eigen::MatrixXd points(grid_width * grid_width, 2);
        Eigen::Index k = 0;
        for (uint64_t i = 0; i < grid_width; ++i) {
            for (uint64_t j = 0; j < grid_width; ++j) {
                points(k, 0) = i;
                points(k, 1) = j;
                ++k;
            }
        }

        Eigen::VectorXd y = points(Eigen::all, 0).array() + points(Eigen::all, 1).array();

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        /*
         * . x . . .
         * . . x . .
         * . . . x .
         * . . . . x  ->
         * . . . . .
         *     |    \
         *     v     v
         */
        // std::cout << adjacency_matrix << std::endl;
        // for (Eigen::Index j = 0; j < adjacency_matrix.outerSize(); ++j) {
        //     for (
        //         Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j);
        //         it;
        //         ++it
        //     ) {
        //         REQUIRE( it.row() == it.col() - 1 );
        //     }
        // }
        // REQUIRE( adjacency_matrix.nonZeros() == grid_width * grid_width - 1 );

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            Eigen::VectorXd::Constant(y.rows(), 1),
            gir::L2());

        REQUIRE_EQUAL(Eigen::VectorXd(y(idx_new)), y_fit);

        REQUIRE( gir::is_monotonic(Eigen::MatrixXd(points(idx_new, Eigen::all)), y_fit) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit) );
    }

    SECTION("Comparison With UniIsoRegression R Package 1") {
        Eigen::MatrixXd points(8, 2);
        points << 0, 0,   0, 1,   0, 2,   0, 3,
                  1, 0,   1, 1,   1, 2,   1, 3;

        Eigen::VectorXd y(8);
        y << 2, 4, 3, 1,
             5, 7, 9, 0;

        Eigen::VectorXd weight = Eigen::VectorXd::Constant(y.rows(), 1);

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        // L2 Norm
        gir::L2 l2_norm;

        auto [groups_L2, y_fit_L2] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            l2_norm);

        Eigen::VectorXd uni_iso_y_L2(8);
        uni_iso_y_L2 << 2, 2.636364, 2.636364, 2.636364,
                        5, 5,        5,        5;

        Eigen::VectorXd y_fit_L2_reordered =
            Eigen::VectorXd(y_fit_L2(idx_original, Eigen::all));

        REQUIRE(
            l2_norm.loss(y, y_fit_L2_reordered, weight) <=
                l2_norm.loss(y, uni_iso_y_L2, weight) );

        REQUIRE( gir::is_monotonic(points, y_fit_L2_reordered) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_L2) );

        // Pseudo L1 Norm
        gir::L1 l1_norm;

        auto [groups_L1, y_fit_L1] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            l1_norm);

        Eigen::VectorXd uni_iso_y_L1(8);
        uni_iso_y_L1 << 2, 3, 3, 3,
                        5, 7, 7, 7;

        Eigen::VectorXd y_fit_L1_reordered =
            Eigen::VectorXd(y_fit_L1(idx_original, Eigen::all));

        REQUIRE(
            l1_norm.loss(y, y_fit_L1_reordered, weight) <=
                l1_norm.loss(y, uni_iso_y_L1, weight) );

        REQUIRE( gir::is_monotonic(points, y_fit_L1_reordered) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_L1) );
    }

    SECTION("Small difference") {
        // const auto default_precision{std::cout.precision()};
        // std::cout << std::setprecision(20);

        Eigen::MatrixXd points(8, 2);
        points << 0, 0,   0, 1,   0, 2,   0, 3,
                  1, 0,   1, 1,   1, 2,   1, 3;

        Eigen::VectorXd y(8);
        y << 1, 1, 1, 1,
             1, 1, 1, 1 + 1e-10;

        Eigen::VectorXd weight = Eigen::VectorXd::Constant(y.rows(), 1);

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            gir::L2());

        REQUIRE_EQUAL ( y, Eigen::VectorXd(y_fit(idx_original)) );

        gir::VectorXu reordered_groups = groups(idx_original);
        REQUIRE ( (reordered_groups(gir::VectorXu::LinSpaced(7, 0, 6)).array() == reordered_groups(0)).all() );
        REQUIRE ( (reordered_groups(gir::VectorXu::LinSpaced(7, 0, 6)).array() != reordered_groups(8)).all() );

        // std::cout << std::setprecision(default_precision);
    }

    SECTION("Duplicate Points 1") {
        Eigen::MatrixXd points(4, 2);
        points << 0, 0,   0, 0,   0, 0,   0, 0;

        Eigen::VectorXd y(4);
        y << 0.1, 0.4, 0.9, 1; // 0.6 mean

        Eigen::VectorXd weight = Eigen::VectorXd::Constant(y.rows(), 1);

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            gir::L2());

        REQUIRE ( (y_fit.array() == 0.6).all() );
        REQUIRE ( (groups.array() == 0).all() );
    }

    SECTION("Duplicate Points 2") {
        Eigen::MatrixXd points(8, 2);
        points << 0, 0,   0, 1,   1, 0,   0.5, 0.5,
                  1, 1,   1, 1,   1, 1,   1,   1;

        Eigen::VectorXd y(8);
        y <<  0.1, 0.1, 0.9, 0.9, // will be pulled up by the last two points
              0.1, 0.2, 0.9, 1; // 0.6 mean

        Eigen::VectorXd weight = Eigen::VectorXd::Constant(y.rows(), 1);

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            gir::L2(),
            2);

        Eigen::VectorXd reordered_y_fit = y_fit(idx_original);
        gir::VectorXu reordered_groups = groups(idx_original);

        // points 2 + 3 push the mean up
        REQUIRE ( (reordered_y_fit({4, 5, 6, 7}).array() - (.9 + .9 + .1 + .2 + .9 + 1)/6 < 1e-6 ).all() );
        REQUIRE ( (reordered_groups({4, 5, 6, 7}).array() == reordered_groups[7]).all() );
    }

    SECTION("Duplicate Points 3") {
        Eigen::MatrixXd points(8, 2);
        points <<   0,   0,   0, 0,   0, 0,     0,   0,
                  0.1, 0.1,   0, 1,   1, 0,   0.5, 0.5;

        Eigen::VectorXd y(8);
        y <<  0.1, 0.1, 0.9, 0.9, // 0.5 mean
              0.1, 0.2, 0.9, 1;  // will be pushed down by the first two points

        Eigen::VectorXd weight = Eigen::VectorXd::Constant(y.rows(), 1);

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(points);

        auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weight,
            gir::L2(),
            2);

        Eigen::VectorXd reordered_y_fit = y_fit(idx_original);
        gir::VectorXu reordered_groups = groups(idx_original);

        // points 4 + 5 push the mean down
        REQUIRE ( (reordered_y_fit({0, 1, 2, 3}).array() - (.1 + .1 + .9 + .9 + .1 + .2)/6 < 1e-6 ).all() );
        REQUIRE ( (reordered_groups({0, 1, 2, 3}).array() == reordered_groups[0]).all() );
    }
}

TEST_CASE( "random examples are monotonic", "[isotonic_regression]" ) {
    SECTION( "Normal" ) {
        for (size_t iteration = 0; iteration < 5; ++iteration) {
            size_t dimensions = std::rand() % 10 + 1;

            auto [X, y] = gir::generate_monotonic_points(1000, 0.01, dimensions);

            REQUIRE( X.rows() == y.rows() );
            REQUIRE( X.cols() == dimensions );
            REQUIRE( y.cols() == 1 );

            auto [adjacency_matrix, idx_original, idx_new] =
                gir::points_to_adjacency(X);

            REQUIRE( X.rows() == adjacency_matrix.rows() );
            REQUIRE( X.rows() == adjacency_matrix.cols() );
            REQUIRE( X.rows() == idx_original.rows() );
            REQUIRE( X.rows() == idx_new.rows() );

            Eigen::MatrixXd sorted_points = X(idx_new, Eigen::all);
            Eigen::VectorXd sorted_ys = y(idx_new);
            Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

            REQUIRE( X.rows() == sorted_points.rows() );
            REQUIRE( sorted_points.cols() == dimensions );

            REQUIRE( X.rows() == sorted_ys.rows() );
            REQUIRE( sorted_ys.cols() == 1 );

            REQUIRE( X.rows() == weights.rows() );
            REQUIRE( weights.cols() == 1 );

            auto [groups, y_fit] = generalised_isotonic_regression(
                adjacency_matrix,
                sorted_ys,
                weights,
                gir::L2());

            REQUIRE( X.rows() == y_fit.rows() );
            REQUIRE( y_fit.cols() == 1 );

            REQUIRE( gir::is_monotonic(sorted_points, y_fit) );
            REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit) );
        }
    }

    SECTION( "Duplicates" ) {
        for (size_t iteration = 0; iteration < 5; ++iteration) {
            size_t dimensions = std::rand() % 10 + 1;

            auto [X, y] = gir::generate_monotonic_points(1000, 0.01, dimensions);

            // Am only changing X here, so should make scores worse
            // There could also be others that are identical due to random
            gir::VectorXu identical_points_1 = gir::VectorXu::LinSpaced(5, 0, 4);
            gir::VectorXu identical_points_2 = gir::VectorXu::LinSpaced(10, 20, 29);

            for (auto i : identical_points_1)
                X(i, Eigen::all).array() = X(identical_points_1(0), Eigen::all).array();

            for (auto i : identical_points_2)
                X(i, Eigen::all).array() = X(identical_points_2(0), Eigen::all).array();

            REQUIRE( X.rows() == y.rows() );
            REQUIRE( X.cols() == dimensions );
            REQUIRE( y.cols() == 1 );

            auto [adjacency_matrix, idx_original, idx_new] =
                gir::points_to_adjacency(X);

            REQUIRE( X.rows() == adjacency_matrix.rows() );
            REQUIRE( X.rows() == adjacency_matrix.cols() );
            REQUIRE( X.rows() == idx_original.rows() );
            REQUIRE( X.rows() == idx_new.rows() );

            Eigen::MatrixXd sorted_points = X(idx_new, Eigen::all);
            Eigen::VectorXd sorted_ys = y(idx_new);
            Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

            REQUIRE( X.rows() == sorted_points.rows() );
            REQUIRE( sorted_points.cols() == dimensions );

            REQUIRE( X.rows() == sorted_ys.rows() );
            REQUIRE( sorted_ys.cols() == 1 );

            REQUIRE( X.rows() == weights.rows() );
            REQUIRE( weights.cols() == 1 );

            auto [groups, y_fit] = generalised_isotonic_regression(
                adjacency_matrix,
                sorted_ys,
                weights,
                gir::L2());

            REQUIRE( X.rows() == y_fit.rows() );
            REQUIRE( y_fit.cols() == 1 );

            REQUIRE( gir::is_monotonic(sorted_points, y_fit) );
            REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit) );

            gir::VectorXu reordered_groups = groups(idx_original, Eigen::all);
            Eigen::VectorXd reordered_y_fit = y_fit(idx_original, Eigen::all);

            REQUIRE ( (reordered_groups(identical_points_1).array() == reordered_groups(identical_points_1(0))).all() );
            REQUIRE ( (reordered_y_fit(identical_points_1).array() == reordered_y_fit(identical_points_1(0))).all() );

            REQUIRE ( (reordered_groups(identical_points_2).array() == reordered_groups(identical_points_2(0))).all() );
            REQUIRE ( (reordered_y_fit(identical_points_2).array() == reordered_y_fit(identical_points_2(0))).all() );
        }
    }

    SECTION( "Brute Force vs Specialised 1d Method" ) {

        for (size_t iteration = 0; iteration < 5; ++iteration) {
            size_t dimensions = 1;

            auto [X, y] = gir::generate_monotonic_points(50, 0.01, dimensions);
            Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

            // Brute Force
            auto [adjacency_matrix_b, idx_original_b, idx_new_b] =
                gir::points_to_adjacency_N_brute_force(X);

            Eigen::MatrixXd sorted_points_b = X(idx_new_b, Eigen::all);
            Eigen::VectorXd sorted_ys_b = y(idx_new_b);

            auto [groups_b, y_fit_b] = generalised_isotonic_regression(
                adjacency_matrix_b,
                sorted_ys_b,
                weights,
                gir::L2());

            REQUIRE( gir::is_monotonic(sorted_points_b, y_fit_b) );
            REQUIRE( gir::is_monotonic(adjacency_matrix_b, y_fit_b) );

            // 1d Specialised
            auto [adjacency_matrix_s, idx_original_s, idx_new_s] =
                gir::points_to_adjacency_1d(X);

            Eigen::MatrixXd sorted_points_s = X(idx_new_s, Eigen::all);
            Eigen::VectorXd sorted_ys_s = y(idx_new_s);

            auto [groups_s, y_fit_s] = generalised_isotonic_regression(
                adjacency_matrix_s,
                sorted_ys_s,
                weights,
                gir::L2());

            REQUIRE( gir::is_monotonic(sorted_points_s, y_fit_s) );
            REQUIRE( gir::is_monotonic(adjacency_matrix_s, y_fit_s) );

            Eigen::VectorXd reordered_y_fit_b = y_fit_b(idx_original_b, Eigen::all);
            Eigen::VectorXd reordered_y_fit_s = y_fit_s(idx_original_s, Eigen::all);

            REQUIRE( ((reordered_y_fit_b.array() - reordered_y_fit_s.array()) < 1e-12).all() );
        }
    }

    SECTION( "Brute Force vs Specialised 2d Method" ) {

        for (size_t iteration = 0; iteration < 5; ++iteration) {
            size_t dimensions = 2;

            auto [X, y] = gir::generate_monotonic_points(50, 0.01, dimensions);
            Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

            // Brute Force
            auto [adjacency_matrix_b, idx_original_b, idx_new_b] =
                gir::points_to_adjacency_N_brute_force(X);

            Eigen::MatrixXd sorted_points_b = X(idx_new_b, Eigen::all);
            Eigen::VectorXd sorted_ys_b = y(idx_new_b);

            auto [groups_b, y_fit_b] = generalised_isotonic_regression(
                adjacency_matrix_b,
                sorted_ys_b,
                weights,
                gir::L2());

            REQUIRE( gir::is_monotonic(sorted_points_b, y_fit_b) );
            REQUIRE( gir::is_monotonic(adjacency_matrix_b, y_fit_b) );

            // 2d Specialised
            auto [adjacency_matrix_s, idx_original_s, idx_new_s] =
                gir::points_to_adjacency_2d(X);

            Eigen::MatrixXd sorted_points_s = X(idx_new_s, Eigen::all);
            Eigen::VectorXd sorted_ys_s = y(idx_new_s);

            auto [groups_s, y_fit_s] = generalised_isotonic_regression(
                adjacency_matrix_s,
                sorted_ys_s,
                weights,
                gir::L2());

            REQUIRE( gir::is_monotonic(sorted_points_s, y_fit_s) );
            REQUIRE( gir::is_monotonic(adjacency_matrix_s, y_fit_s) );

            Eigen::VectorXd reordered_y_fit_b = y_fit_b(idx_original_b, Eigen::all);
            Eigen::VectorXd reordered_y_fit_s = y_fit_s(idx_original_s, Eigen::all);

            REQUIRE( ((reordered_y_fit_b.array() - reordered_y_fit_s.array()) < 1e-12).all() );
        }
    }
}

TEST_CASE( "comparing loss functions", "[isotonic_regression]" ) {
    for (size_t iteration = 0; iteration < 5; ++iteration) {
        size_t dimensions = std::rand() % 10 + 1;

        auto [X, y] = gir::generate_monotonic_points(1000, 0.01, dimensions);

        REQUIRE( X.rows() == y.rows() );
        REQUIRE( X.cols() == dimensions );
        REQUIRE( y.cols() == 1 );

        auto [adjacency_matrix, idx_original, idx_new] =
            gir::points_to_adjacency(X);

        REQUIRE( X.rows() == adjacency_matrix.rows() );
        REQUIRE( X.rows() == adjacency_matrix.cols() );
        REQUIRE( X.rows() == idx_original.rows() );
        REQUIRE( X.rows() == idx_new.rows() );

        Eigen::MatrixXd sorted_points = X(idx_new, Eigen::all);
        Eigen::VectorXd sorted_ys = y(idx_new);
        Eigen::VectorXd equal_weights = Eigen::VectorXd::Constant(y.rows(), 1);

        REQUIRE( X.rows() == sorted_points.rows() );
        REQUIRE( sorted_points.cols() == dimensions );

        REQUIRE( X.rows() == sorted_ys.rows() );
        REQUIRE( sorted_ys.cols() == 1 );

        REQUIRE( X.rows() == equal_weights.rows() );
        REQUIRE( equal_weights.cols() == 1 );

        // L2 loss without weights
        auto [groups_l2, y_fit_l2] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            equal_weights,
            gir::L2());

        REQUIRE( X.rows() == y_fit_l2.rows() );
        REQUIRE( y_fit_l2.cols() == 1 );

        REQUIRE( gir::is_monotonic(sorted_points, y_fit_l2) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_l2) );


        double l2_loss = gir::L2().estimator(sorted_ys, equal_weights);
        double l2_loss_weighted = gir::L2_WEIGHTED().estimator(sorted_ys, equal_weights);
        REQUIRE( l2_loss == l2_loss_weighted );

        const auto& l2_loss_deriv = gir::L2().derivative(l2_loss, sorted_ys, equal_weights);
        const auto& l2_loss_weighted_deriv = gir::L2_WEIGHTED().derivative(l2_loss_weighted, sorted_ys, equal_weights);
        REQUIRE_EQUAL( l2_loss_deriv, l2_loss_weighted_deriv );

        // L2 loss weighted
        auto [groups_l2_weighted, y_fit_l2_weighted] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            equal_weights,
            gir::L2_WEIGHTED());

        REQUIRE( X.rows() == y_fit_l2_weighted.rows() );
        REQUIRE( y_fit_l2_weighted.cols() == 1 );

        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_l2_weighted) );
        REQUIRE( gir::is_monotonic(sorted_points, y_fit_l2_weighted) );


        REQUIRE_EQUAL( groups_l2, groups_l2_weighted );
        REQUIRE_EQUAL( y_fit_l2, y_fit_l2_weighted );

        // The y values in generate_monotonic_points are generated via the prod function and then noise is added.
        // So we add a bias to those that are above the true y generated by the prod function.
        Eigen::VectorXd bias_upwards_weights = (sorted_ys.array() > sorted_points.rowwise().prod().array()).select(equal_weights.array() * 2, equal_weights);
        // or similarly below
        Eigen::VectorXd bias_downwards_weights = (sorted_ys.array() < sorted_points.rowwise().prod().array()).select(equal_weights.array() * 2, equal_weights);

        auto [groups_l2_weighted_bias_up, y_fit_l2_weighted_bias_up] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            bias_upwards_weights,
            gir::L2_WEIGHTED());

        auto [groups_l2_weighted_bias_down, y_fit_l2_weighted_bias_down] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            bias_downwards_weights,
            gir::L2_WEIGHTED());

        REQUIRE( X.rows() == y_fit_l2_weighted_bias_up.rows() );
        REQUIRE( y_fit_l2_weighted_bias_up.cols() == 1 );

        REQUIRE( X.rows() == y_fit_l2_weighted_bias_down.rows() );
        REQUIRE( y_fit_l2_weighted_bias_down.cols() == 1 );

        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_l2_weighted_bias_up) );
        REQUIRE( gir::is_monotonic(sorted_points, y_fit_l2_weighted_bias_up) );

        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_l2_weighted_bias_down) );
        REQUIRE( gir::is_monotonic(sorted_points, y_fit_l2_weighted_bias_down) );

        REQUIRE( (y_fit_l2_weighted_bias_up.array() >= y_fit_l2.array()).all() );
        REQUIRE( (y_fit_l2_weighted_bias_down.array() <= y_fit_l2.array()).all() );

        // Huber loss without weights
        auto [groups_huber, y_fit_huber] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            equal_weights,
            gir::HUBER(0.1));

        REQUIRE( X.rows() == y_fit_huber.rows() );
        REQUIRE( y_fit_huber.cols() == 1 );

        REQUIRE( gir::is_monotonic(sorted_points, y_fit_huber) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_huber) );

        auto [groups_huber_2, y_fit_huber_2] = generalised_isotonic_regression(
            adjacency_matrix,
            sorted_ys,
            equal_weights,
            gir::HUBER(2.1));

        REQUIRE( X.rows() == y_fit_huber_2.rows() );
        REQUIRE( y_fit_huber_2.cols() == 1 );

        REQUIRE( gir::is_monotonic(sorted_points, y_fit_huber_2) );
        REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_huber_2) );

        // L1 loss without weights
        // auto [groups_l1, y_fit_l1] = generalised_isotonic_regression(
        //     adjacency_matrix,
        //     sorted_ys,
        //     equal_weights,
        //     gir::L1());

        // REQUIRE( X.rows() == y_fit_l1.rows() );
        // REQUIRE( y_fit_l1.cols() == 1 );

        // REQUIRE( gir::is_monotonic(sorted_points, y_fit_l1) );
        // REQUIRE( gir::is_monotonic(adjacency_matrix, y_fit_l1) );
    }
}
