#include "utility.h"

namespace gir {

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

std::pair<VectorXu, VectorXu>
argpartition(const Eigen::VectorX<bool>& solution) {
    auto total_true = solution.count();

    VectorXu p1(solution.size() - total_true);
    VectorXu p2(total_true);
    Eigen::Index p1_idx = 0;
    Eigen::Index p2_idx = 0;
    for (Eigen::Index itr = 0; itr < solution.rows(); ++itr) {
        if (solution(itr)) {
            p2[p2_idx++] = itr;
        } else {
            p1[p1_idx++] = itr;
        }
    }

    return std::make_pair(std::move(p1), std::move(p2));
}

VectorXu
find(const Eigen::VectorX<bool>& solution) {
    auto total_true = solution.count();

    VectorXu p(total_true);
    Eigen::Index p_idx = 0;
    for (Eigen::Index itr = 0; itr < solution.rows(); ++itr) {
        if (solution(itr)) {
            p[p_idx++] = itr;
        }
    }

    return p;
}

} // namespace gir
