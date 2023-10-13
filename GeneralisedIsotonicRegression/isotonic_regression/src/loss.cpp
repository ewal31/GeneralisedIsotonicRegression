#include "loss.h"

namespace gir {

Eigen::VectorXd normalise(const Eigen::VectorXd& loss_derivative) {
    if (loss_derivative.cwiseAbs().maxCoeff() > 0)
        // normalise largest coefficient to +/- 1
        return loss_derivative / loss_derivative.cwiseAbs().maxCoeff();
    return loss_derivative;
}

double median(Eigen::VectorXd vals) {
    const auto middle = vals.begin() + vals.size() / 2;
    std::nth_element(vals.begin(), middle, vals.end());
    if (vals.size() % 2 == 0) {
        const auto middle2 = std::max_element(vals.begin(), middle);
        return (*middle2 + *middle) / 2;
    } else {
        return *middle;
    }
}

} // namespace gir
