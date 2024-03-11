#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <generalized_isotonic_regression.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/pybind11.h>

#include <stdexcept>


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


std::pair<gir::VectorXu, Eigen::VectorXd>
py_generalised_isotonic_regression (
    const Eigen::SparseMatrix<bool>& adjacency_matrix,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& _weights,
    std::string loss_function,
    uint64_t max_iterations,
    double delta, // Huber Loss
    double p      // P-Norm
) {

    // TODO check arguments are all correct sizes

    transform(
        loss_function.begin(),
        loss_function.end(),
        loss_function.begin(),
        ::toupper
    );

    const Eigen::VectorXd weights = _weights.rows() == 0 ? Eigen::VectorXd::Ones(y.rows()) : _weights;

    if (loss_function == "L2") {

        return gir::generalised_isotonic_regression(
            adjacency_matrix, y, weights,
            gir::L2_WEIGHTED(), max_iterations
        );

    } else if (loss_function == "POISSON") {

        return gir::generalised_isotonic_regression(
            adjacency_matrix, y, weights,
            gir::POISSON(), max_iterations
        );

    } else if (loss_function == "PNORM") {

        if (p <= 1 || p >= 2)
            throw std::invalid_argument("When using the P-Norm, set the p argument to a value between 1 and 2");
        return gir::generalised_isotonic_regression(
            adjacency_matrix, y, weights,
            gir::PNORM(p), max_iterations
        );

    } else if (loss_function == "HUBER") {

        if (delta <= 0)
            throw std::invalid_argument("When using the Huber, set the delta argument to a value above 0");
        return gir::generalised_isotonic_regression(
            adjacency_matrix, y, weights,
            gir::HUBER(delta), max_iterations
        );

    } else {
        throw std::invalid_argument("The loss function " + loss_function + " is unsupported.");
    }
}

double
py_calculate_loss (
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& y_fit,
    const Eigen::VectorXd& _weights,
    std::string loss_function,
    double delta, // Huber Loss
    double p      // P-Norm
) {

    transform(
        loss_function.begin(),
        loss_function.end(),
        loss_function.begin(),
        ::toupper
    );

    const Eigen::VectorXd weights = _weights.rows() == 0 ? Eigen::VectorXd::Ones(y.rows()) : _weights;

    if (loss_function == "L2") {
        return gir::L2_WEIGHTED().loss(y, y_fit, weights);

    } else if (loss_function == "POISSON") {
        return gir::POISSON().loss(y, y_fit, weights);

    } else if (loss_function == "PNORM") {
        if (p <= 1 || p >= 2)
            throw std::invalid_argument("When using the P-Norm, set the p argument to a value between 1 and 2");

        return gir::PNORM(p).loss(y, y_fit, weights);

    } else if (loss_function == "HUBER") {
        if (delta <= 0)
            throw std::invalid_argument("When using the Huber, set the delta argument to a value above 0");

        return gir::HUBER(delta).loss(y, y_fit, weights);

    } else {
        throw std::invalid_argument("The loss function " + loss_function + " is unsupported.");
    }
}

namespace py = pybind11;

PYBIND11_MODULE(multivariate_isotonic_regression, m) {

    m.def("generate_monotonic_points", &gir::generate_monotonic_points,
        py::arg("total"),
        py::arg("sigma") = 0.1,
        py::arg("dimensions") = 2,
        R"pbdoc(
            Generates Monotonic Points with added noise.

            Arguments:
            * total number of points
            * standard deviation of noise
            * number of dimensions of the x values

            Returns a tuple containing:
            * the points/x values
            * the observations/y values corresponding to each x
        )pbdoc"
    );

    m.def("points_to_adjacency",
        [](const Eigen::MatrixXd& points) { return gir::points_to_adjacency(points); },
        R"pbdoc(
            Builds a graph from unnorded points, where each row is
            considered a point.

            A point is considered larger than all others that it is
            greater than or equal to across all dimensions.

            Edges are only included for points that are direct
            predecessors/successors of one another.

            In one dimension, this means that for the points [1, 2, 4]
            we have an edge from 1 -> 2 and 2 -> 4 but not 1 -> 4.

            Arguments:
            * A multidimensional array, the rows of which are points

            Returns a tuple containing:
            * A sparse adjacency matrix, with 1's indicating edges
            * original_indexes: adjacency(original_indexes(i), original_indexes(j)) shows whether there is an edge between points(i) and points(j)
            * new_indexes: adjacency(i, j) compares points(new_indexes(i)) and points(new_indexes(j))
        )pbdoc"
    );

    m.def("generalised_isotonic_regression", &py_generalised_isotonic_regression,
        py::arg("adjaceny_matrix"),
        py::arg("y"),
        py::arg("weights") = Eigen::VectorXd(0),
        py::arg("loss_function") = "L2",
        py::arg("max_iterations") = 0,
        py::kw_only(),
        py::arg("delta") = 0,
        py::arg("p") = 0,
        R"pbdoc(
            Runs the Generalised Isotonic Regression Algorithm
            on the provided adjacency matrix, with corresponding
            observations, y, and weights,

            Arguments:
            * A sparse adjacency matrix
            * y (observations)
            * weights
            * loss function ["L2", "POISSON", "PNORM", "HUBER"]
            * max iterations

            Keyword Arguments:
            * delta, defines the proportion of the Huber loss that is linear/quadratic
            * p, modulates the p-norm between estimate a median when closer to 1 and mean when closer to 2

            Returns a tuple containing:
            * A sparse adjacency matrix, with 1's indicating edges
            * original_indexes: adjacency(original_indexes(i), original_indexes(j)) shows whether there is an edge between points(i) and points(j)
            * new_indexes: adjacency(i, j) compares points(new_indexes(i)) and points(new_indexes(j))
        )pbdoc"
    );

    m.def("calculate_loss", &py_calculate_loss,
        py::arg("y"),
        py::arg("y_fit"),
        py::arg("weights") = Eigen::VectorXd(0),
        py::arg("loss_function") = "L2",
        py::kw_only(),
        py::arg("delta") = 0,
        py::arg("p") = 0,
        R"pbdoc(
            Calculate the total loss between y and y_fit
            weighted by the weights according to the specified
            loss function.

            Arguments:
            * a vector y
            * a second vector y_fit
            * weights
            * loss function ["L2", "POISSON", "PNORM", "HUBER"]

            Keyword Arguments:
            * delta, defines the proportion of the Huber loss that is linear/quadratic
            * p, modulates the p-norm between estimate a median when closer to 1 and mean when closer to 2

            Returns a tuple containing:
            * A sparse adjacency matrix, with 1's indicating edges
            * original_indexes: adjacency(original_indexes(i), original_indexes(j)) shows whether there is an edge between points(i) and points(j)
            * new_indexes: adjacency(i, j) compares points(new_indexes(i)) and points(new_indexes(j))
        )pbdoc"
    );

    m.def("is_monotonic",
        [](const Eigen::MatrixXd points, const Eigen::VectorXd y, const double tolerance) {
            return gir::is_monotonic(points, y, tolerance);
        },
        py::arg("points"),
        py::arg("y"),
        py::arg("tolerance") = 1e-6,
        R"pbdoc(
            Check if the points y are increasing monotonically
            with the values x

            This is an exhaustive bruteforce implementation and not
            suitable for large datasets.

            Arguments:
            * n-dimensional array x where each row is a point
            * 1-dimensional array y

            Returns a boolean True or False
        )pbdoc"
    );

    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
}
