#include "utility.h"

std::pair<VectorXu, VectorXu>
argpartition(const Eigen::VectorX<bool>& solution) {
    auto total_true = solution.count();

    VectorXu p1(solution.size() - total_true);
    VectorXu p2(total_true);
    size_t p1_idx = 0;
    size_t p2_idx = 0;
    for (size_t itr = 0; itr < solution.rows(); ++itr) {
        if (solution(itr)) {
            p2[p2_idx++] = itr;
        } else {
            p1[p1_idx++] = itr;
        }
    }

    return std::make_pair(std::move(p1), std::move(p2));
}
