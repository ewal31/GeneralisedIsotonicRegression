#include "generalized_isotonic_regression.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <Highs.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>


enum class LossFunction {
    L2
};

double calculate_loss_estimator(
    const LossFunction loss,
    const Eigen::VectorXd& vals
) {
    switch (loss) {
        case LossFunction::L2: return vals.mean();
    }
}

// TODO change to general normalise
Eigen::VectorXd normalise_loss_derivative(const Eigen::VectorXd& loss_derivative) {
    if (loss_derivative.cwiseAbs().maxCoeff() > 0)
        // normalise largest coefficient to +/- 1
        return loss_derivative / loss_derivative.cwiseAbs().maxCoeff();
    return loss_derivative;
}

Eigen::VectorXd calculate_loss_derivative(const LossFunction loss, const double loss_estimator, const Eigen::VectorXd& vals) {
    switch (loss) {
        case LossFunction::L2: return normalise_loss_derivative(2 * (vals.array() - loss_estimator));
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
generate_monotonic_points(size_t total, double sigma, size_t dimensions) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<> point_distribution(0, 1);
    std::normal_distribution<double> noise_distribution(0., sigma);

    Eigen::MatrixXd points = Eigen::MatrixXd::NullaryExpr(
        total,
        dimensions,
        [&point_distribution, &generator](){
            return point_distribution(generator);
        });

    Eigen::VectorXd regressed_values = points.rowwise().prod() +
        Eigen::MatrixXd::NullaryExpr(
            total,
            1,
            [&noise_distribution, &generator](){
                return noise_distribution(generator);
            });

    return std::make_pair(std::move(points), std::move(regressed_values));
}

// // TODO rename
Eigen::VectorX<bool>
GIRP_cutproblem_wrapper(const Eigen::SparseMatrix<bool>& adjacency_matrix, const Eigen::VectorXd& y, const Eigen::VectorX<uint16_t>& considered_idxs, LossFunction loss) {
    // There are other ways to get this value, such as the degree vector, but to keep things general
    // it is being recalculated here.
    // uint16_t min_constraint_row = adjacency_matrix.rows();
    // uint16_t max_constraint_row = 0;

    // for (int j = 0; j < adjacency_matrix.outerSize(); ++j) { // column if column-wise
    //     Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j);
    //     min_constraint_row = it ? it.row() < min_constraint_row ? it.row() : min_constraint_row : min_constraint_row;

    //     Eigen::SparseMatrix<bool>::ReverseInnerIterator rit(adjacency_matrix, j);
    //     max_constraint_row = rit ? rit.row() > max_constraint_row ? rit.row() : max_constraint_row : max_constraint_row;
    // }

    // uint16_t total_constraints = max_constraint_row >= min_constraint_row ? max_constraint_row - min_constraint_row + 1 : 0;
    // uint16_t total_observations = y.rows();

    uint16_t total_constraints = adjacency_matrix.nonZeros();
    uint16_t total_observations = y.rows();

    // There are other options here see GIRP_estimator and cut_value but just doing L2 norm for now
    double estimator = calculate_loss_estimator(loss, y);
    Eigen::VectorXd network_flow_constraint = calculate_loss_derivative(loss, estimator, y); // this was the first derivative from paper I think

    // std::cout << "min_constraint_row:\n" << min_constraint_row << '\n' << std::endl;
    // std::cout << "max_constraint_row:\n" << max_constraint_row << '\n' << std::endl;
    std::cout << "total_constraints:\n" << total_constraints << '\n' << std::endl;
    std::cout << "total_observations:\n" << total_observations << '\n' << std::endl;
    std::cout << "Estimator:\n" << estimator << '\n' << std::endl;
    std::cout << "flow_constraint:\n" << network_flow_constraint << '\n' << std::endl;

    // Already optimal if there is no chance for improvement?
    if (network_flow_constraint.isApproxToConstant(0)) {
        return Eigen::VectorX<bool>::Constant(total_observations, true);
    }

    uint16_t idx = 0;
    std::vector<Eigen::Triplet<double>> row_coefficients_list; // TODO reserve
    std::vector<Eigen::Triplet<double>> col_coefficients_list; // TODO reserve
    for (int j = 0; j < adjacency_matrix.outerSize(); ++j) {
        for (Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j); it; ++it) {
            row_coefficients_list.emplace_back(it.row(), idx + 2 * total_observations, 1);
            col_coefficients_list.emplace_back(it.col(), idx + 2 * total_observations, 1);
            ++idx;
        }

        // Add source and sink identity matrix parts at the same time
        row_coefficients_list.emplace_back(j, j, 1);
        col_coefficients_list.emplace_back(j, j + total_observations, 1);
    }

    Eigen::SparseMatrix<double> row_coefficients(total_observations, total_observations * 2 + total_constraints);
    row_coefficients.setFromTriplets(row_coefficients_list.begin(), row_coefficients_list.end());

    Eigen::SparseMatrix<double> col_coefficients(total_observations, total_observations * 2 + total_constraints);
    col_coefficients.setFromTriplets(col_coefficients_list.begin(), col_coefficients_list.end());

    std::cout << "row_coefficients:\n" << row_coefficients << '\n' << std::endl;
    std::cout << "col_coefficients:\n" << col_coefficients << '\n' << std::endl;

    Eigen::SparseMatrix<double> Aineq = row_coefficients - col_coefficients;

    std::cout << "rows:\n" << row_coefficients << '\n' << std::endl;
    std::cout << "cols:\n" << col_coefficients << '\n' << std::endl;
    std::cout << "Aineq:\n" << Aineq << '\n' << std::endl;

    double infinity = 1.0e30; // TODO surely there is a better option
    HighsModel model;
    model.lp_.num_col_ = Aineq.cols();
    model.lp_.num_row_ = Aineq.rows();
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.offset_ = 0;

    std::vector<double> cost_coefs(Aineq.cols());
    for (int i = 0; i < total_observations * 2; ++i) cost_coefs[i] = 1;
    for (int i = 0; i < total_constraints; ++i) cost_coefs[i + total_observations * 2] = 0;
    model.lp_.col_cost_ = cost_coefs;
    model.lp_.col_lower_ = std::vector<double>(Aineq.cols(), 0);
    model.lp_.col_upper_ = std::vector<double>(Aineq.cols(), infinity);

    std::vector<double> equality_vector(network_flow_constraint.begin(), network_flow_constraint.end());
    model.lp_.row_lower_ = equality_vector;
    model.lp_.row_upper_ = equality_vector;

    std::vector<int> column_start_positions(Aineq.cols() + 1);
    std::vector<int> nonzero_row_index(Aineq.nonZeros());
    std::vector<double> nonzero_values(Aineq.nonZeros());
    idx = 0;
    for (int j = 0; j < Aineq.outerSize(); ++j) {
        column_start_positions[j] = idx;
        for (Eigen::SparseMatrix<double>::InnerIterator it(Aineq, j); it; ++it) {
            nonzero_row_index[idx] = it.row();
            nonzero_values[idx] = it.value();
            ++idx;
        }
    }
    column_start_positions[column_start_positions.size()-1] = idx;

    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = column_start_positions;
    model.lp_.a_matrix_.index_ = nonzero_row_index;
    model.lp_.a_matrix_.value_ = nonzero_values;

    std::cout << "model summary:\n";
    std::cout << "columns:\n" << model.lp_.num_col_ << '\n';
    std::cout << "rows:\n" << model.lp_.num_row_ << '\n';
    std::cout << "offset:\n" << model.lp_.offset_ << '\n';
    std::cout << "cost coefficients:\n" << fmt::format("{}", model.lp_.col_cost_) << '\n';
    std::cout << "column lower bound:\n" << fmt::format("{}", model.lp_.col_lower_) << '\n';
    std::cout << "column upper bound:\n" << fmt::format("{}", model.lp_.col_upper_) << '\n';
    std::cout << "row lower bound:\n" << fmt::format("{}", model.lp_.row_lower_) << '\n';
    std::cout << "row upper bound:\n" << fmt::format("{}", model.lp_.row_upper_) << '\n';
    std::cout << "A column starts:\n" << fmt::format("{}", model.lp_.a_matrix_.start_) << '\n';
    std::cout << "A row index:\n" << fmt::format("{}", model.lp_.a_matrix_.index_) << '\n';
    std::cout << "A value:\n" << fmt::format("{}", model.lp_.a_matrix_.value_) << '\n';

    // Create a Highs instance
    Highs highs;
    HighsStatus return_status;

    // Pass the model to HiGHS
    return_status = highs.passModel(model);
    assert(return_status == HighsStatus::kOk);

    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 4); // Primal

    // Get a const reference to the LP data in HiGHS
    const HighsLp& lp = highs.getLp();

    // Solve the model
    return_status = highs.run();
    assert(return_status == HighsStatus::kOk);

    // Get the model status
    const HighsModelStatus& model_status = highs.getModelStatus();
    assert(model_status==HighsModelStatus::kOptimal);

    const HighsInfo& info = highs.getInfo();
    assert(info.dual_solution_status == 2); // feasible
    std::cout << "Simplex iteration count: " << info.simplex_iteration_count << '\n' << std::endl;
    std::cout << "Objective function value: " << info.objective_function_value << '\n' << std::endl;
    std::cout << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << '\n' << std::endl;
    std::cout << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << " -> " << info.dual_solution_status << '\n' << std::endl;
    std::cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << '\n' << std::endl;

    fmt::println("Solution col: \n{}\n", highs.getSolution().col_value);
    fmt::println("Solution row: \n{}\n", highs.getSolution().row_value);
    fmt::println("Dual col: \n{}\n", highs.getSolution().col_dual);
    fmt::println("Dual row: \n{}\n", highs.getSolution().row_dual);

    // Might need to handle a value that is exactly 0 in a different way.
    Eigen::VectorX<bool> solution = Eigen::VectorXd::Map(
        &highs.getSolution().row_dual[0],
        highs.getSolution().row_dual.size()
    ).array() > 0; // 0 left = 1 right

    return solution;
}

// void
// generalised_isotonic_regression(
//     const Eigen::SparseMatrix<bool>& adjacency_matrix,
//     const Eigen::VectorXd& y,
//     const LossFunction loss
// ) {
//     uint16_t total_observations = y.rows();
// 
//     Eigen::VectorX<uint16_t> group = Eigen::VectorX<uint16_t>::Zero(total_observations); // each group will have a different id
//     Eigen::VectorXd cut_values = Eigen::VectorXd::Zero(total_observations); // objective value of partition problems. dused to decide which cut to make at each iteration
// 
//     Eigen::VectorXi cutgroups1 = Eigen::VectorXi::Zero(total_observations);
//     Eigen::VectorXi cutgroups2 = Eigen::VectorXi::Zero(total_observations);
//     uint16_t cutcount1 = 0;
//     uint16_t cutcount2 = 0;
//     uint16_t iteration = 1;
//     double max_cut_values = 0;
// 
//     Eigen::VectorXd yfits = Eigen::VectorXd::Zero(total_observations);
//     // cutInds=[];            % output variable to track of cuts along the path to the final solution
//     // violation_flag=0;      % output flag to check that isotonicity is maintained
// 
//     // find initial partitioning
//     Eigen::VectorX<uint16_t> considered_idxs = Eigen::VectorX<uint16_t>::LinSpaced(total_observations, 0, total_observations - 1);
//     Eigen::VectorX<bool> solution =  GIRP_cutproblem_wrapper(adjacency_matrix, y, considered_idxs, loss);
// 
//     auto [left, right] = argpartition(solution);
// 
//     std::cout << "left:\n" << left << '\n' << std::endl;
//     std::cout << "right:\n" << right << '\n' << std::endl;
// 
//     double estimator = calculate_loss_estimator(loss, y);
//     cut_values.array() = calculate_loss_derivative(loss, estimator, y(right)).sum() - calculate_loss_derivative(loss, estimator, y(left)).sum();
// 
//     cutgroups1(left).array() = cutcount1 + 1;
//     cutgroups1(right).array() = cutcount1 + 2;
//     cutgroups2.array() = cutcount2 + 1;
// 
//     cutcount1 += 2;
//     cutcount2 += 1;
//     max_cut_values = 0;
// 
//     std::cout << "cut_values:\n" << cut_values << '\n' << std::endl;
//     std::cout << "cutgroups1:\n" << cutgroups1 << '\n' << std::endl;
//     std::cout << "cutgroups2:\n" << cutgroups2 << '\n' << std::endl;
// 
// 
//     const auto& [max_cut_value, indCutValue] = find_first_max(cut_values);
// 
//     auto indsToCut = find_where(cutgroups2,
//         [&cutgroups2, &indCutValue = indCutValue](int i){ // Initialisation not required with g++ or cpp 20
//             return cutgroups2(i) == cutgroups2(indCutValue);
//         });
// 
//     std::cout << "max_cut_value:\n" << max_cut_value << '\n' << std::endl;
//     std::cout << "indCutValue:\n" << indCutValue << '\n' << std::endl;
//     std::cout << "indsToCut:\n" << indsToCut << '\n' << std::endl;
// 
//     // Only really need the unique values and I don't think the order is important.
//     // could probably just throw into a set.
//     Eigen::VectorXi tmp = cutgroups1(indsToCut);
//     std::sort(tmp.begin(), tmp.end());
//     tmp.conservativeResize(std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end())));
// 
//     std::cout << "tmp:\n" << tmp << '\n' << std::endl;
// 
//     int count = 0;
//     yfits.array() = calculate_loss_estimator(loss, y); // don't really think this is necessary
// 
//     for (auto to_cut: tmp) {
//         fmt::println("Processing group {}", to_cut);
// 
//         indsToCut = find_where(cutgroups1,
//             [&cutgroups1, &to_cut](int i){ // Initialisation not required with g++ or cpp 20
//                 return cutgroups1(i) == to_cut;
//             });
// 
//         auto groupsize = indsToCut.rows();
//         group(indsToCut).array() = ++count; // divide these groups in the current solution
//         yfits(indsToCut).array() = calculate_loss_estimator(loss, y(indsToCut));
// 
//         std::cout << "indsToCut:\n" << indsToCut << '\n' << std::endl;
//         std::cout << "groupsize:\n" << groupsize << '\n' << std::endl;
//         std::cout << "group:\n" << group << '\n' << std::endl;
//         std::cout << "yfits:\n" << yfits << '\n' << std::endl;
// 
//         Eigen::VectorXi inds_to_cut_sorted_idxs = Eigen::VectorXi::LinSpaced(indsToCut.rows(), 0, indsToCut.rows() - 1);
//         std::sort(
//             inds_to_cut_sorted_idxs.begin(),
//             inds_to_cut_sorted_idxs.end(),
//             [&indsToCut](uint16_t i, uint16_t j){
//                 return indsToCut(i) < indsToCut(j);
//             }
//         );
// 
//         // create a copy of adjacency reordered to be the same order as degree_idxs.
//         Eigen::SparseMatrix<bool> adjacency_subset(indsToCut.rows(), indsToCut.rows());
//         adjacency_subset.reserve(Eigen::VectorXi::Constant(indsToCut.rows(), indsToCut.rows()));
// 
//         // TODO needs to be redone
//         int new_j = 0;
//         for (int j : indsToCut) { // column if column-wise
//             for (Eigen::SparseMatrix<bool>::InnerIterator it(adjacency_matrix, j); it; ++it) {
//                 auto row = std::find(indsToCut.begin(), indsToCut.end(), it.row());
//                 if (row != indsToCut.end()) {
//                     adjacency_subset.insert(std::distance(indsToCut.begin(), row), new_j) = it.value();
//                 }
//             }
//             ++new_j;
//         }
//         adjacency_subset.makeCompressed();
// 
//         Eigen::VectorX<bool> solution = GIRP_cutproblem_wrapper(
//             adjacency_subset,
//             y(indsToCut(inds_to_cut_sorted_idxs)),
//             considered_idxs,
//             loss
//         );
// 
//         std::cout << "solution:\n" << solution << '\n' << std::endl;
//         std::cout << "inds_to_cut_sorted_idxs:\n" << inds_to_cut_sorted_idxs << '\n' << std::endl;
// 
//         if (solution.count() != solution.rows() && solution.count() != 0) {
//             auto [left, right] = argpartition(solution);
//             double estimator = calculate_loss_estimator(loss, y(inds_to_cut_sorted_idxs));
//             cut_values.array()(inds_to_cut_sorted_idxs) = calculate_loss_derivative(loss, estimator, y(right)).sum() - calculate_loss_derivative(loss, estimator, y(left)).sum();
// 
//             // indleft=find(x<=-.9);indright=find(x>=.9);
//             // estimator=GIRP_estimator(y(indsToCut([indleft;indright])),options,indsToCut([indleft;indright]));
//             // cutvalues(indsToCut([indleft;indright]))=sum(cut_value(y(indsToCut(indright)),estimator,options,indsToCut(indright)))-sum(cut_value(y(indsToCut(indleft)),estimator,options,indsToCut(indleft)));
//             // cutgroups1(indsToCut(indleft))=cutcount1+1;
//             // cutgroups1(indsToCut(indright))=cutcount1+2;
//             // cutgroups2([indsToCut(indleft);indsToCut(indright)])=cutcount2+1;
//             // cutcount1=cutcount1+2;
//             // cutcount2=cutcount2+1;
// 
//         } else {
//             cut_values(indsToCut).array() = -1e30;
//         }
//     }
// 
//     // sort( vec.begin(), vec.end() );
//     // vec.erase( unique( vec.begin(), vec.end() ), vec.end() );
//     // cutgroups1(indsToCut).array();
// 
//     // temp=sort(unique(cutgroups1(indsToCut)),'ascend');
// 
// 
// 
//     // Eigen::VectorXi indsToCut(cutgroups2);
//     // auto to_find = cutgroups2(indCutValue);
//     // auto partition_end = std::stable_partition(indsToCut.begin(), indsToCut.end(), [&to_find](int n){return n == to_find;});
//     // indsToCut.resize(std::distance(indsToCut.begin(), partition_end));
// 
//     // New partition and select code?? should be using std::distance
//     // auto asdf = std::stable_partition(cutgroups1.begin(), cutgroups1.end(), [](int n){return n > 1;});
//     // std::cout << "cutgroups1:\n" << cutgroups1 << '\n' << std::endl;
//     // std::cout << "=:\n" << asdf - cutgroups1.begin() << '\n' << std::endl;
//     // cutgroups1.conservativeResize(asdf - cutgroups1.begin());
//     // std::cout << "cutgroups1:\n" << cutgroups1 << '\n' << std::endl;
//     // (cut_values.array() == cut_values.maxCoeff()).select(Eigen::VectorXi::LinSpaced(cut_values.rows(), 0, cut_values.rows() - 1), -1);
// 
// }

void run() {
    // auto [X, y] = generate_monotonic_points(5, 1e-10, 2);

    /* position_matrix_to_adjacency_matrix simple test 2.
     *
     * xt::xarray<double> points {
     *     {1, 1},
     *     {2, 3},
     *     {3, 2},
     *     {4, 4},
     *     {5, 5}
     * };
     *
     * should produce
     *
     * {{0, 1, 1, 0, 0},
     *  {0, 0, 0, 1, 0},
     *  {0, 0, 0, 1, 0},
     *  {0, 0, 0, 0, 1},
     *  {0, 0, 0, 0, 0}}
     */

    Eigen::MatrixX<uint8_t> X(5, 1);
    X << 1,
         2,
         3,
         4,
         5;

    Eigen::VectorXd y(5);
    // regressed_values << 1, 1, 1.2, 1, 1;
    y << 1, 1, 3, 5, 5;

    std::cout << "points:\n" << X << '\n' << std::endl;
    std::cout << "regressed_values:\n" << y << '\n' << std::endl;

    auto [adjacency_matrix, idx_original, idx_new] =
        points_to_adjacency(X);

    std::cout << "adjacency_matrix:\n" << adjacency_matrix << '\n' << std::endl;
    std::cout << "idx_original:\n" << idx_original << '\n' << std::endl;
    std::cout << "idx_new:\n" << idx_new << '\n' << std::endl;
    std::cout << "points reordered to adjacency_matrix\n";
    std::cout << X(idx_new, Eigen::all) << "\n" << std::endl;
    std::cout << y(idx_new) << "\n" << std::endl;
    std::cout << "Just checking this works\n" << std::endl;

    //generalised_isotonic_regression(adjacency_matrix, y, LossFunction::L2);
}
