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

        // std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

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

        // std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

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

        // std::cout << "pareto_rank:\n" << pareto_rank_orig << std::endl;

        REQUIRE_EQUAL(expected, pareto_rank_orig);
    }

}

TEST_CASE( "non_dominated_sort", "[dominated_sorting]" ) {

    // SECTION( "Simple Example (variation over single dimension)" ) {
    //     Eigen::MatrixX<double> points(5, 2);
    //     points << 1, 1,
    //               2, 1,
    //               3, 1,
    //               4, 1,
    //               5, 1;

    //     gir::VectorXu expected(5);
    //     expected << 4,
    //                 3,
    //                 2,
    //                 1,
    //                 0;

    //     auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //     const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //     // std::cout << "pareto_rank\n" << pareto_rank << std::endl;
    //     // std::cout << "pareto_rank_reordered\n" << pareto_rank_reordered << std::endl;
    //     // std::cout << "expected\n" << expected << std::endl;

    //     REQUIRE_EQUAL(expected, pareto_rank_reordered);
    // }

    // SECTION( "Simple Example 2" ) {

    //     //    ^
    //     //    |
    //     //  2 |                 x
    //     //    |
    //     //    |                       should have 3 wavefronts
    //     //    |
    //     //  1 x        x
    //     //    |      x
    //     //    |
    //     //    |
    //     //    ---------x------------->
    //     //    0        1        2

    //     Eigen::MatrixX<double> points(5, 2);
    //     points << 0,       1   ,
    //               1,       0   ,
    //               0.99,    0.99,
    //               1,       1   ,
    //               2,       2   ;

    //     gir::VectorXu expected(5);
    //     expected << 2,
    //                 2,
    //                 2,
    //                 1,
    //                 0;

    //     auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //     const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //     // std::cout << "pareto_rank\n" << pareto_rank << std::endl;
    //     // std::cout << "pareto_rank_reordered\n" << pareto_rank_reordered << std::endl;
    //     // std::cout << "expected\n" << expected << std::endl;

    //     REQUIRE_EQUAL(expected, pareto_rank_reordered);
    // }

    // SECTION( "2 waves with 2 extremal points" ) {

    //     //       ^
    //     //       |
    //     // (0,5) x
    //     //       |
    //     //       |                 should have 2 wavefronts
    //     //       |
    //     //       |
    //     //       |          (1,1)
    //     //       x           x
    //     //       |     x
    //     //       |       x
    //     //       |         x
    //     //       |
    //     //       ------------x-----------------x---->
    //     //                                   (5.0)

    //     Eigen::MatrixX<double> points(8, 2);
    //     points << 0  ,  1  ,
    //               1  ,  0  ,
    //               0.5,  0.9,
    //               0.9,  0.5,
    //               0.8,  0.8,
    //               1,      1,
    //               0,      5,
    //               5,      0;

    //     gir::VectorXu expected(8);
    //     expected << 1,
    //                 1,
    //                 1,
    //                 1,
    //                 1,
    //                 0,
    //                 0,
    //                 0;

    //     auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //     const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //     REQUIRE_EQUAL(expected, pareto_rank_reordered);
    // }

    SECTION( "Many wavefronts" ) {

        Eigen::MatrixX<double> points(19, 2);
        points << 1,  0,  // 0
                  0,  2,  // 1
                  1,  2,  // 1
                  2,  0,  // 1
                  2,  1,  // 1
                  2,  2,  // 1
                  0,  3,  // 2
                  1,  3,  // 2
                  3,  0,  // 2
                  3,  1,  // 2
                  3,  2,  // 2
                  3,  3,  // 2
                  0,  4,  // 3
                  1,  4,  // 3
                  3,  4,  // 3
                  4,  1,  // 3
                  4,  2,  // 3
                  4,  3,  // 3
                  4,  4;  // 3

        gir::VectorXu expected(19);
        expected << 0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3;

        auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

        const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

        std::cout << "pareto_rank\n" << pareto_rank << std::endl;
        std::cout << "pareto_rank_reordered\n" << pareto_rank_reordered << std::endl;
        std::cout << "expected\n" << expected << std::endl;

        REQUIRE_EQUAL(expected, pareto_rank_reordered);
    }

    // SECTION( "2 waves with 2 extremal points, with duplicate point" ) {

    //     //       ^
    //     //       |
    //     // (0,5) x
    //     //       |
    //     //       |                 should have 2 wavefronts
    //     //       |
    //     //       |
    //     //       |          (1,1)
    //     //       x           x
    //     //       |     x
    //     //       |       x
    //     //       |         x
    //     //       |
    //     //       ------------x-----------------x---->
    //     //                                   (5.0)

    //     Eigen::MatrixX<double> points(9, 2);
    //     points << 0  ,  1  ,
    //               1  ,  0  ,
    //               0.5,  0.9,
    //               0.9,  0.5,
    //               0.8,  0.8,
    //               1,    1  ,
    //               0,    5  ,
    //               5,    0  ,
    //               0  ,  1  ; // duplicate

    //     gir::VectorXu expected(9);
    //     expected << 1,
    //                 1,
    //                 1,
    //                 1,
    //                 1,
    //                 0,
    //                 0,
    //                 0,
    //                 1; // duplicate

    //     for (Eigen::Index i = 0; i < 8; ++i) {
    //         // std::cout << "duplicate point: " << i << std::endl;

    //         // Make last point a duplicate of one of the
    //         // previous points
    //         points(8, 0) = points(i, 0);
    //         points(8, 1) = points(i, 1);

    //         expected(8) = expected(i);

    //         auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //         const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //         // std::cout << "pareto_rank\n" << pareto_rank << std::endl;
    //         // std::cout << "pareto_rank_reordered\n" << pareto_rank_reordered << std::endl;
    //         // std::cout << "expected\n" << expected << std::endl;

    //         REQUIRE_EQUAL(expected, pareto_rank_reordered);

    //     }

    // }

    // SECTION( "2 waves with 2 extremal points, with two duplicate points" ) {

    //     //       ^
    //     //       |
    //     // (0,5) x
    //     //       |
    //     //       |                 should have 2 wavefronts
    //     //       |
    //     //       |
    //     //       |          (1,1)
    //     //       x           x
    //     //       |     x
    //     //       |       x
    //     //       |         x
    //     //       |
    //     //       ------------x-----------------x---->
    //     //                                   (5.0)

    //     Eigen::MatrixX<double> points(10, 2);
    //     points << 0  ,  1  ,
    //               1  ,  0  ,
    //               0.5,  0.9,
    //               0.9,  0.5,
    //               0.8,  0.8,
    //               1,    1  ,
    //               0,    5  ,
    //               5,    0  ,
    //               0  ,  1  , // duplicate
    //               0  ,  1  ; // duplicate

    //     gir::VectorXu expected(10);
    //     expected << 1,
    //                 1,
    //                 1,
    //                 1,
    //                 1,
    //                 0,
    //                 0,
    //                 0,
    //                 0, // duplicate
    //                 0; // duplicate

    //     for (Eigen::Index i = 0; i < 8; ++i) {

    //         // First duplicate
    //         points(8, 0) = points(i, 0);
    //         points(8, 1) = points(i, 1);
    //         expected(8) = expected(i);

    //         for (Eigen::Index j = 0; j < 8; ++j) {

    //             // Second duplicate
    //             points(9, 0) = points(j, 0);
    //             points(9, 1) = points(j, 1);
    //             expected(9) = expected(j);

    //             auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //             const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //             // std::cout << "pareto_rank\n" << pareto_rank << std::endl;
    //             // std::cout << "pareto_rank_reordered\n" << pareto_rank_reordered << std::endl;
    //             // std::cout << "expected\n" << expected << std::endl;

    //             REQUIRE_EQUAL(expected, pareto_rank_reordered);

    //         }

    //     }

    // }

    // SECTION( "Random Points" ) {

    //     /*     ┌──────────────────────────────────────────────────┐
    //      *   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀p8⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀p10⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠁p1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀-⠀-⠀-⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀p3⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀p4⠀ \⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀--⠀⠀⠀⠀front 1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀--⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀p2⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀--⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀front 2⠀⠀⠀⠀⠀--⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀-⠀-⠀-⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀------⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀p9⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀p6⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀│
    //      *     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ - -⠀⡀⠀p7⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀│
    //      *   0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀p5⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ \⠀⠀⠀⠀⠀│
    //      *     └──────────────────────────────────────────────────┘
    //      *     ⠀0.1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀1⠀
    //      *
    //      *  p5 is in a front by itself (front 3)
    //      *  p8 and p10 are in a wavefront together (front 0)
    //      *
    //      */
    //     Eigen::MatrixX<double> points(10, 2);
    //     points << 0.118816,   0.794008, // 1
    //               0.298615,   0.362457, // 2
    //               0.346182,   0.643438, // 1
    //               0.351713,   0.565865, // 1
    //               0.457662,  0.0339997, // 3
    //               0.512422,   0.107742, // 2
    //               0.623951,  0.0586508, // 2
    //               0.717142,   0.963573, // 0
    //               0.801575,    0.18501, // 1
    //               0.981288,   0.959583; // 0

    //     gir::VectorXu expected(10);
    //     expected << 1,
    //                 2,
    //                 1,
    //                 1,
    //                 3,
    //                 2,
    //                 2,
    //                 0,
    //                 1,
    //                 0;

    //     auto [pareto_rank, idx_orig, idx_new, unique_idxs] = gir::non_dominated_sort(points);

    //     const gir::VectorXu pareto_rank_reordered = pareto_rank(idx_orig);

    //     REQUIRE_EQUAL(expected, pareto_rank_reordered);

    // }

    // TODO test more than 2 dimensions
}
