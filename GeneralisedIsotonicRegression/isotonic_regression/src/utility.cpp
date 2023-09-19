#include "utility.h"

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
