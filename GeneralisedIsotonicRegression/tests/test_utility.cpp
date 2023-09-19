#include "utility.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "argsort Eigen::VectorX", "[argsort]" ) {
    Eigen::VectorXi input(5);
    Eigen::VectorXi expected(5);
    expected << 0, 1, 2, 3, 4;

    SECTION( "already ordered" ) {
        input << 0, 1, 2, 3, 4;
        auto idxs = argsort(input);
        REQUIRE( input(idxs) == expected );
    }

    SECTION( "in reverse" ) {
        input << 4, 3, 2, 1, 0;
        auto idxs = argsort(input);
        REQUIRE( input(idxs) == expected );
    }

    SECTION( "out of order 1" ) {
        input << 1, 2, 4, 0, 3;
        auto idxs = argsort(input);
        REQUIRE( input(idxs) == expected );
    }

    SECTION( "out of order 2" ) {
        input << 2, 1, 0, 4, 3;
        auto idxs = argsort(input);
        REQUIRE( input(idxs) == expected );
    }

    SECTION( "another type" ) {
        Eigen::VectorX<char> input(5);
        input << 'z', 'a', 'x', 'e', 'e';

        Eigen::VectorX<char> expected(5);
        expected << 'a', 'e', 'e', 'x', 'z';

        auto idxs = argsort(input);
        REQUIRE( input(idxs) == expected );
    }
}

TEST_CASE( "argsort Eigen::MatrixX", "[argsort]" ) {
    Eigen::MatrixXi input(5, 2);
    Eigen::MatrixXi expected(5, 2);
    expected << 0, 1,
                2, 2,
                2, 3,
                3, 0,
                5, 5;

    SECTION( "already ordered" ) {
        input << 0, 1,
                 2, 2,
                 2, 3,
                 3, 0,
                 5, 5;
        auto idxs = argsort(input);
        REQUIRE( input(idxs, Eigen::all) == expected );
    }

    SECTION( "in reverse" ) {
        input << 5, 5,
                 3, 0,
                 2, 3,
                 2, 2,
                 0, 1;
        auto idxs = argsort(input);
        REQUIRE( input(idxs, Eigen::all) == expected );
    }

    SECTION( "out of order 1" ) {
        input << 3, 0,
                 2, 3,
                 2, 2,
                 5, 5,
                 0, 1;
        auto idxs = argsort(input);
        REQUIRE( input(idxs, Eigen::all) == expected );
    }

    SECTION( "out of order 2" ) {
        input << 0, 1,
                 5, 5,
                 2, 2,
                 2, 3,
                 3, 0;
        auto idxs = argsort(input);
        REQUIRE( input(idxs, Eigen::all) == expected );
    }

    SECTION( "another type" ) {
        Eigen::MatrixX<char> input(5, 2);
        input << 'z', 'a',
                 'b', 'b',
                 'y', 'a',
                 'r', 'q',
                 'x', 'w';

        Eigen::MatrixX<char> expected(5, 2);
        expected << 'b', 'b',
                    'r', 'q',
                    'x', 'w',
                    'y', 'a',
                    'z', 'a';

        auto idxs = argsort(input);

        REQUIRE( input(idxs, Eigen::all) == expected );
    }
}

TEST_CASE( "argpartition", "[argpartition]" ) {
    Eigen::VectorX<bool> input(5);

    SECTION( "all true" ) {
        input << true, true, true, true, true;

        VectorXu expected_left = VectorXu(0);
        VectorXu expected_right = VectorXu(5);
        expected_right << 0, 1, 2, 3, 4;

        auto [left, right] = argpartition(input);

        REQUIRE( expected_left == left);
        REQUIRE( expected_right == right );
    }

    SECTION( "all false" ) {
        input << false, false, false, false, false;

        VectorXu expected_left = VectorXu(5);
        VectorXu expected_right = VectorXu(0);
        expected_left << 0, 1, 2, 3, 4;

        auto [left, right] = argpartition(input);

        REQUIRE( expected_left == left);
        REQUIRE( expected_right == right );
    }

    SECTION( "partition 1" ) {
        input << false, false, false, true, true;

        VectorXu expected_left = VectorXu(3);
        VectorXu expected_right = VectorXu(2);
        expected_left << 0, 1, 2;
        expected_right << 3, 4;

        auto [left, right] = argpartition(input);

        REQUIRE( expected_left == left);
        REQUIRE( expected_right == right );
    }

    SECTION( "partition 2" ) {
        input << false, true, false, true, false;

        VectorXu expected_left = VectorXu(3);
        VectorXu expected_right = VectorXu(2);
        expected_left << 0, 2, 4;
        expected_right << 1, 3;

        auto [left, right] = argpartition(input);

        REQUIRE( expected_left == left);
        REQUIRE( expected_right == right );
    }

    SECTION( "partition 3" ) {
        input << true, true, false, true, false;

        VectorXu expected_left = VectorXu(2);
        VectorXu expected_right = VectorXu(3);
        expected_left << 2, 4;
        expected_right << 0, 1, 3;

        auto [left, right] = argpartition(input);

        REQUIRE( expected_left == left);
        REQUIRE( expected_right == right );
    }
}
