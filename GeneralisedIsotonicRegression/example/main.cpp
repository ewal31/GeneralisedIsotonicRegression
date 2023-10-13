#include <iostream>

#include <generalized_isotonic_regression.h>

int main() {
    auto [X, y] = gir::generate_monotonic_points(10, 1e-2, 2);
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(y.rows(), 1);

    std::cout << "point positions:\n" << X << '\n' << std::endl;
    std::cout << "monotonic axis:\n" << y << '\n' << std::endl;

    auto [adjacency_matrix, idx_original, idx_new] =
        gir::points_to_adjacency(X);

    std::cout << "adjacency matrix:\n" << adjacency_matrix << '\n' << std::endl;
    std::cout << "adjacency matrix ordering:\n" << idx_new << '\n' << std::endl;
    std::cout << "points reordered\n";
    std::cout << X(idx_new, Eigen::all) << "\n" << std::endl;
    std::cout << y(idx_new) << "\n" << std::endl;

    gir::L2 loss_function;

    auto [groups, y_fit] = generalised_isotonic_regression(
            adjacency_matrix,
            y(idx_new),
            weights,
            loss_function);

    std::cout << "fit y values:\n" << y_fit << "\n" << std::endl;

    double total_loss = loss_function.loss(
        y(idx_new),
        y_fit,
        weights);

    std::cout << "with total loss: " << total_loss << "\n" << std::endl;

    return 0;
}
