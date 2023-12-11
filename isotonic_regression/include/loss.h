#pragma once

#include <Eigen/Core>

#include <iostream>
#include <stdlib.h>
#include <type_traits>

namespace gir {

Eigen::VectorXd normalise(const Eigen::VectorXd& loss_derivative);

double median(Eigen::VectorXd vals);

template <typename T>
int8_t sign(
        const T val,
        typename std::enable_if<std::is_arithmetic_v<T>>::type* = 0
) {
    return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
}

template <typename T> Eigen::VectorXi
sign(const Eigen::VectorX<T>& vals) {
    return vals.unaryExp(
        [](const auto val){
            return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
        });
}

// CRTP
template <typename Derived>
class LossFunction {
 public:
    double loss(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& y_fit,
        const Eigen::VectorXd& weights
    ) const {
        return (static_cast<Derived const*>(this))->
            loss(y, y_fit, weights);
    }

    // template <typename V = Derived>
    // double estimator(
    //     const Eigen::VectorXd& vals,
    //     const Eigen::VectorXd& weights,
    //     typename std::enable_if<std::is_member_function_pointer_v<decltype(&V::estimator)>>::type* = 0
    // ) const
    // {
    //     std::cout << "Using specialised method" << std::endl;
    //     return (static_cast<Derived const*>(this))->
    //         estimator(vals, weights);
    // }

    // double estimator(
    //     const Eigen::VectorXd& vals,
    //     const Eigen::VectorXd& weights,
    //     ...
    // ) const
    // {
    //     std::cout << "Not using specialised method" << std::endl;
    //     return binary_search_based_estimator(
    //         vals,
    //         weights);
    // }

    double estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights
    ) const
    {
        return (static_cast<Derived const*>(this))->
            estimator(vals, weights);
    }

    double binary_search_based_estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        const double precision = 1e-9;
        double left = vals.minCoeff();
        double right = vals.maxCoeff();
        double mid = (left + right) / 2;
        double loss_left = derivative(left, vals, weights).sum();
        double loss_mid = derivative(mid, vals, weights).sum();

        while (abs(left - right) > precision) {

            if (sign(loss_left) == sign(loss_mid)) {
                left = mid;
                loss_left = derivative(left, vals, weights).sum();
            } else {
                right = mid;
            }

            mid = (left + right) / 2;
            loss_mid = derivative(mid, vals, weights).sum();
        }

        return mid;
    }

    Eigen::VectorXd derivative(
        const double loss_estimate,
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return normalise(
                (static_cast<Derived const*>(this))->
                    derivative(loss_estimate, vals, weights));
    }

    Eigen::VectorXd derivative(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return this->derivative(
            this->estimator(vals, weights),
            vals,
            weights);
    }

 private:
    LossFunction() {}
    friend Derived;
};

class L1 : public LossFunction<L1> {
 private:
     inline static bool warned = false;
 public:
    double loss(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& y_fit,
        const Eigen::VectorXd& weights) const
    {
        return (y.array() - y_fit.array()).cwiseAbs().sum();
    }

    double estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return median(vals);
    }

    Eigen::VectorXd derivative(
        const double loss_estimate,
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        if (!warned) {
            std::cout << "L1 Norm isn't technically everwhere differentiable, so use at own risk" << std::endl;
            warned = true;
        }
        return (vals.array() == loss_estimate).
            select(0, (vals.array() < loss_estimate).
                    select(-1, Eigen::VectorXd::Ones(vals.rows()))).array();
    }
};

class L2 : public LossFunction<L2> {
 public:
    double loss(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& y_fit,
        const Eigen::VectorXd& weights) const
    {
        return std::sqrt((y.array() - y_fit.array()).pow(2).sum());
    }

    double estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return vals.mean();
    }

    Eigen::VectorXd derivative(
        const double loss_estimate,
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return 2 * (vals.array() - loss_estimate);
    }
};

class L2_WEIGHTED : public LossFunction<L2_WEIGHTED> {
 public:
    double loss(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& y_fit,
        const Eigen::VectorXd& weights) const
    {
        const auto&& diff = (weights.array() * (y.array() - y_fit.array()));
        return std::sqrt(diff.pow(2).sum());
    }

    double estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return weights.cwiseProduct(vals).sum() / weights.sum();
    }

    Eigen::VectorXd derivative(
        const double loss_estimate,
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return 2 * (vals.array() - loss_estimate) * weights.array();
    }
};

class HUBER : public LossFunction<HUBER> {
 private:
    const double delta;
 public:
    explicit HUBER(const double delta): delta(delta) {}

    double loss(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& y_fit,
        const Eigen::VectorXd& weights) const
    {
        const Eigen::ArrayXd diff = y.array() - y_fit.array();
        const auto losses = (diff.cwiseAbs() <= delta)
            .select(
                    weights.array() * 0.5 * diff.pow(2),
                    weights.array() * delta * (diff.cwiseAbs() - 0.5 * delta))
            / weights.sum();

        return losses.sum();
    }

    double estimator(
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        return binary_search_based_estimator(vals, weights);
    }

    Eigen::VectorXd derivative(
        const double loss_estimate,
        const Eigen::VectorXd& vals,
        const Eigen::VectorXd& weights) const
    {
        const Eigen::ArrayXd diff = vals.array() - loss_estimate;
        return (diff.cwiseAbs() <= delta)
            .select(
                    weights.array() * diff,
                    weights.array() * sign(diff) * delta)
            / weights.sum();
    }
};

} // namespace gir
