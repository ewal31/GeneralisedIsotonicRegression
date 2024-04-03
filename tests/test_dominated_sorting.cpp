#include "dominated_sorting.h"
#include "test_helpers.h"

#include "generalized_isotonic_regression.h" // TODO delete
#include <Eigen/Core>
#include <Eigen/SparseCore> // TODO delete
#include <catch2/catch_test_macros.hpp>


TEST_CASE( "total_dominated_by", "[dominated_sorting]" ) {

    SECTION( "Simple Example (variation over single dimension)" ) {

        Eigen::MatrixX<double> points(5, 2);
        points << 1, 1,
                  2, 1,
                  3, 1,
                  4, 1,
                  5, 1;

        gir::VectorXu expected(5);
        expected << 4,
                    3,
                    2,
                    1,
                    0;

        auto [pareto_rank, idx_orig, idx_new] = gir::total_dominated_by(points);

        const gir::VectorXu pareto_rank_orig = pareto_rank(idx_orig, Eigen::all);

        std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

        REQUIRE_EQUAL(expected, pareto_rank_orig);
    }

    SECTION( "Simple Example 2" ) {

        //    ^
        //    |
        //  2 |                 x
        //    |
        //    |                       should have 3 wavefronts
        //    |
        //  1 x        x
        //    |      x
        //    |
        //    |
        //    ---------x------------->
        //    0        1        2

        Eigen::MatrixX<double> points(5, 2);
        points << 0,       1   ,
                  1,       0   ,
                  0.99,    0.99,
                  1,       1   ,
                  2,       2   ;

        gir::VectorXu expected(5);
        expected << 2,
                    2,
                    2,
                    1,
                    0;

        auto [pareto_rank, idx_orig, idx_new] = gir::total_dominated_by(points);

        const gir::VectorXu pareto_rank_orig = pareto_rank(idx_orig, Eigen::all);

        std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

        REQUIRE_EQUAL(expected, pareto_rank_orig);
    }

    SECTION( "Simple Example 3" ) {

        //       ^
        //       |
        // (0,5) x
        //       |
        //       |                 should have 2 wavefronts
        //       |
        //       |
        //       |          (1,1)
        //       x           x
        //       |     x
        //       |       x
        //       |         x
        //       |
        //       ------------x-----------------x---->
        //                                   (5.0)

        Eigen::MatrixX<double> points(8, 2);
        points << 0  ,  1  ,
                  1  ,  0  ,
                  0.5,  0.9,
                  0.9,  0.5,
                  0.8,  0.8,
                  1,      1,
                  0,      5,
                  5,      0;

        gir::VectorXu expected(8);
        expected << 2,
                    2,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0;

        auto [pareto_rank, idx_orig, idx_new] = gir::total_dominated_by(points);

        const gir::VectorXu pareto_rank_orig = pareto_rank(idx_orig, Eigen::all);

        std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

        REQUIRE_EQUAL(expected, pareto_rank_orig);
    }

}

TEST_CASE( "non_dominated_sort", "[dominated_sorting]" ) {

    SECTION( "Simple Example (variation over single dimension)" ) {
        Eigen::MatrixX<double> points(5, 2);
        points << 1, 1,
                  2, 1,
                  3, 1,
                  4, 1,
                  5, 1;

        gir::VectorXu expected(5);
        expected << 0,
                    1,
                    2,
                    3,
                    4;

        auto [pareto_rank, idx_orig, idx_new] = gir::non_dominated_sort(points);

        const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

        REQUIRE_EQUAL(expected, pareto_rank_reordered);
    }

    SECTION( "Simple Example 2" ) {

        //    ^
        //    |
        //  2 |                 x
        //    |
        //    |                       should have 3 wavefronts
        //    |
        //  1 x        x
        //    |      x
        //    |
        //    |
        //    ---------x------------->
        //    0        1        2

        Eigen::MatrixX<double> points(5, 2);
        points << 0,       1   ,
                  1,       0   ,
                  0.99,    0.99,
                  1,       1   ,
                  2,       2   ;

        gir::VectorXu expected(5);
        expected << 0,
                    0,
                    0,
                    1,
                    2;

        auto [pareto_rank, idx_orig, idx_new] = gir::non_dominated_sort(points);

        const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

        REQUIRE_EQUAL(expected, pareto_rank_reordered);
    }

    SECTION( "2 waves with 2 extremal points" ) {

        //       ^
        //       |
        // (0,5) x
        //       |
        //       |                 should have 2 wavefronts
        //       |
        //       |
        //       |          (1,1)
        //       x           x
        //       |     x
        //       |       x
        //       |         x
        //       |
        //       ------------x-----------------x---->
        //                                   (5.0)

        Eigen::MatrixX<double> points(8, 2);
        points << 0  ,  1  ,
                  1  ,  0  ,
                  0.5,  0.9,
                  0.9,  0.5,
                  0.8,  0.8,
                  1,      1,
                  0,      5,
                  5,      0;

        gir::VectorXu expected(8);
        expected << 0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1;

        auto [pareto_rank, idx_orig, idx_new] = gir::non_dominated_sort(points);

        const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

        REQUIRE_EQUAL(expected, pareto_rank_reordered);
    }

    SECTION( "Many wavefronts" ) {

        //   ^
        //   |
        // 4 x     x -         x -   x
        //   | \      \            \
        // 3 x   - x   \       x -   x
        //   | \     \  \          \
        // 2 x   - x  \  x -   x -   x
        //   |\     \  \     \     \
        // 1 |  \    \   x -   x     x
        //   |    \    \    \
        // 0 ------x-----x-----x-------------------------
        //   0     1     2     3     4

        Eigen::MatrixX<double> points(19, 2);
        points << 0,  2,  // 7
                  1,  0,  // 7
                  0,  3,  // 6
                  1,  2,  // 6
                  2,  0,  // 6
                  0,  4,  // 5
                  1,  3,  // 5
                  2,  1,  // 5
                  3,  0,  // 5
                  1,  4,  // 4
                  2,  2,  // 4
                  3,  1,  // 4
                  3,  2,  // 3
                  4,  1,  // 3
                  3,  3,  // 2
                  4,  2,  // 2
                  3,  4,  // 1
                  4,  3,  // 1
                  4,  4;  // 0

        gir::VectorXu expected(19);
        expected << 0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    7;

        auto [pareto_rank, idx_orig, idx_new] = gir::non_dominated_sort(points);

        const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

        REQUIRE_EQUAL(expected, pareto_rank_reordered);
    }

}
