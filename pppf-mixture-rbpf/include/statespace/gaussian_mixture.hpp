#pragma once

#include <Eigen/Dense>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace statespace {

struct LinearGaussianTransition {
  Eigen::MatrixXd A;
  Eigen::VectorXd a;
  Eigen::MatrixXd Q;
};

struct GaussianMixtureTransition {
  std::vector<LinearGaussianTransition> components;
  std::vector<double> weights;  // same size as components; should sum to 1

  int size() const { return static_cast<int>(components.size()); }

  void check() const {
    if (components.empty()) throw std::invalid_argument("mixture: empty");
    if (weights.size() != components.size()) throw std::invalid_argument("mixture: weight size");
    const double s = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(std::abs(s - 1.0) < 1e-8)) throw std::invalid_argument("mixture: weights not normalized");
  }
};

}  // namespace statespace

