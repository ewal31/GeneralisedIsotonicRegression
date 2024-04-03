#include "loss.h"

namespace gir {

Eigen::VectorXd normalise(const Eigen::VectorXd& loss_derivative) {
    if (loss_derivative.cwiseAbs().maxCoeff() > 0)
        // normalise largest coefficient to +/- 1
        return loss_derivative / loss_derivative.cwiseAbs().maxCoeff();
    return loss_derivative;
}

} // namespace gir
