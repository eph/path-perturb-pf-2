#pragma once

#include "util/rng.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <vector>

namespace filters {

inline double ess(const Eigen::VectorXd& w_norm) {
  const double s2 = w_norm.array().square().sum();
  return (s2 > 0.0) ? (1.0 / s2) : 0.0;
}

inline std::vector<int> systematic_resample(const Eigen::VectorXd& w_norm, util::Rng& rng) {
  const int N = static_cast<int>(w_norm.size());
  if (N <= 0) throw std::invalid_argument("systematic_resample: N<=0");
  const double sumw = w_norm.sum();
  if (!(std::abs(sumw - 1.0) < 1e-8)) throw std::invalid_argument("systematic_resample: w not normalized");

  std::vector<int> idx(N);
  const double u0 = rng.uniform() / static_cast<double>(N);
  double cdf = w_norm(0);
  int j = 0;
  for (int i = 0; i < N; ++i) {
    const double u = u0 + static_cast<double>(i) / static_cast<double>(N);
    while (u > cdf && j < N - 1) {
      ++j;
      cdf += w_norm(j);
    }
    idx[i] = j;
  }
  return idx;
}

}  // namespace filters

