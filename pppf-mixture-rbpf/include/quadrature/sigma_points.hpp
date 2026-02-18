#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace quadrature {

struct SigmaPoints {
  Eigen::MatrixXd points;   // (2d+1) x d
  Eigen::VectorXd w_mean;   // (2d+1)
  Eigen::VectorXd w_cov;    // (2d+1)
};

// Unscented transform sigma points for x ~ N(mean, cov).
//
// Standard parameters:
//   alpha = 1e-2, kappa = 0, beta = 2
//
// Points:
//   x0 = mean
//   xi = mean ± column_i( sqrt((d+lambda) * cov) )
//
// Weights:
//   w0m = lambda/(d+lambda)
//   w0c = w0m + (1 - alpha^2 + beta)
//   wi  = 1/(2(d+lambda))
inline SigmaPoints unscented_sigma_points(const Eigen::VectorXd& mean,
                                          const Eigen::MatrixXd& cov, double alpha = 1e-2,
                                          double kappa = 0.0, double beta = 2.0) {
  const int d = static_cast<int>(mean.size());
  if (cov.rows() != d || cov.cols() != d) throw std::invalid_argument("sigma_points: dim mismatch");
  if (d <= 0) throw std::invalid_argument("sigma_points: d<=0");
  if (!(alpha > 0.0)) throw std::invalid_argument("sigma_points: alpha<=0");

  const double lambda = alpha * alpha * (static_cast<double>(d) + kappa) - static_cast<double>(d);
  const double c = static_cast<double>(d) + lambda;
  if (!(c > 0.0)) throw std::invalid_argument("sigma_points: d+lambda<=0");

  Eigen::LLT<Eigen::MatrixXd> llt(cov);
  if (llt.info() != Eigen::Success) {
    Eigen::MatrixXd cov_j = cov + 1e-12 * Eigen::MatrixXd::Identity(d, d);
    llt.compute(cov_j);
    if (llt.info() != Eigen::Success) throw std::runtime_error("sigma_points: cov not PD");
  }

  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::MatrixXd S = std::sqrt(c) * L;

  SigmaPoints sp;
  sp.points.resize(2 * d + 1, d);
  sp.w_mean.resize(2 * d + 1);
  sp.w_cov.resize(2 * d + 1);

  sp.points.row(0) = mean.transpose();
  sp.w_mean(0) = lambda / c;
  sp.w_cov(0) = sp.w_mean(0) + (1.0 - alpha * alpha + beta);

  const double wi = 1.0 / (2.0 * c);
  for (int i = 0; i < d; ++i) {
    const Eigen::VectorXd col = S.col(i);
    sp.points.row(1 + i) = (mean + col).transpose();
    sp.points.row(1 + d + i) = (mean - col).transpose();
    sp.w_mean(1 + i) = wi;
    sp.w_mean(1 + d + i) = wi;
    sp.w_cov(1 + i) = wi;
    sp.w_cov(1 + d + i) = wi;
  }

  return sp;
}

inline SigmaPoints unscented_sigma_points_1d(double mean, double var, double alpha = 1e-2,
                                             double kappa = 0.0, double beta = 2.0) {
  if (!(var >= 0.0)) throw std::invalid_argument("sigma_points_1d: var<0");
  Eigen::VectorXd m(1);
  m(0) = mean;
  Eigen::MatrixXd C(1, 1);
  C(0, 0) = var;
  return unscented_sigma_points(m, C, alpha, kappa, beta);
}

}  // namespace quadrature

