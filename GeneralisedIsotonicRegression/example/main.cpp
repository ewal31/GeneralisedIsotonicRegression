#include <iostream>

#include <generalized_isotonic_regression.h>

int main() {
    auto [X, y] = gir::generate_monotonic_points(5, 1e-3, 2);

    std::cout << "point positions:\n" << X << '\n' << std::endl;
    std::cout << "monotonic axis:\n" << y << '\n' << std::endl;

    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    std::cout << "adjacency matrix:\n" << adjacency_matrix << '\n' << std::endl;
    std::cout << "adjacency matrix ordering:\n" << idx_new << '\n' << std::endl;
    std::cout << "points reordered\n";
    std::cout << X(idx_new, Eigen::all) << "\n" << std::endl;
    std::cout << y(idx_new) << "\n" << std::endl;

    auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            Eigen::VectorXd::Constant(y.rows(), 1),
            gir::LossFunction::L2);

    std::cout << "fit y values:\n" << y_fit << "\n" << std::endl;

    return 0;
}
