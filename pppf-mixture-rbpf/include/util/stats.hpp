#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace util {

inline double log_sum_exp(const Eigen::VectorXd& logv) {
  if (logv.size() == 0) return -std::numeric_limits<double>::infinity();
  const double m = logv.maxCoeff();
  return m + std::log((logv.array() - m).exp().sum());
}

inline double ess_from_logw(const Eigen::VectorXd& logw) {
  const double lse = log_sum_exp(logw);
  const Eigen::ArrayXd w = (logw.array() - lse).exp();
  const double sumsq = w.square().sum();
  if (!(sumsq > 0.0)) return 0.0;
  return 1.0 / sumsq;
}

inline double mean(const std::vector<double>& x) {
  if (x.empty()) throw std::invalid_argument("mean: empty");
  const double s = std::accumulate(x.begin(), x.end(), 0.0);
  return s / static_cast<double>(x.size());
}

inline double stdev(const std::vector<double>& x) {
  if (x.size() < 2) return 0.0;
  const double m = mean(x);
  double acc = 0.0;
  for (double v : x) acc += (v - m) * (v - m);
  return std::sqrt(acc / static_cast<double>(x.size() - 1));
}

inline double log_mvnorm_pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean,
                             const Eigen::MatrixXd& cov) {
  const int d = static_cast<int>(x.size());
  if (mean.size() != d || cov.rows() != d || cov.cols() != d) {
    throw std::invalid_argument("log_mvnorm_pdf: dim mismatch");
  }
  Eigen::LLT<Eigen::MatrixXd> llt(cov);
  if (llt.info() != Eigen::Success) {
    // Add a tiny jitter and retry.
    Eigen::MatrixXd cov_j = cov + 1e-12 * Eigen::MatrixXd::Identity(d, d);
    llt.compute(cov_j);
    if (llt.info() != Eigen::Success) throw std::runtime_error("log_mvnorm_pdf: cov not PD");
  }
  const Eigen::VectorXd diff = x - mean;
  const Eigen::VectorXd sol = llt.solve(diff);
  const double quad = diff.dot(sol);
  const Eigen::MatrixXd L = llt.matrixL();
  const double logdet = 2.0 * L.diagonal().array().log().sum();
  return -0.5 * (static_cast<double>(d) * std::log(2.0 * M_PI) + logdet + quad);
}

}  // namespace util
