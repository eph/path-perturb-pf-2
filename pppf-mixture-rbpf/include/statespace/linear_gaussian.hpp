#pragma once

#include "util/stats.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace statespace {

struct LinearGaussianModel {
  // Transition: x_{t+1} = A x_t + a + η,   η ~ N(0, Q)
  Eigen::MatrixXd A;
  Eigen::VectorXd a;
  Eigen::MatrixXd Q;

  // Measurement: y_t = H x_t + d + ε,     ε ~ N(0, R)
  Eigen::MatrixXd H;
  Eigen::VectorXd d;
  Eigen::MatrixXd R;

  int state_dim() const { return static_cast<int>(a.size()); }
  int obs_dim() const { return static_cast<int>(d.size()); }

  void check_dims() const {
    const int n = state_dim();
    const int m = obs_dim();
    if (A.rows() != n || A.cols() != n) throw std::invalid_argument("LG: A dim");
    if (Q.rows() != n || Q.cols() != n) throw std::invalid_argument("LG: Q dim");
    if (H.rows() != m || H.cols() != n) throw std::invalid_argument("LG: H dim");
    if (R.rows() != m || R.cols() != m) throw std::invalid_argument("LG: R dim");
  }
};

struct KalmanState {
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

inline void kalman_predict(const Eigen::MatrixXd& A, const Eigen::VectorXd& a,
                           const Eigen::MatrixXd& Q, KalmanState* st) {
  st->mean = A * st->mean + a;
  st->cov = A * st->cov * A.transpose() + Q;
}

inline double kalman_update(const Eigen::MatrixXd& H, const Eigen::VectorXd& d,
                            const Eigen::MatrixXd& R, const Eigen::VectorXd& y, KalmanState* st) {
  const Eigen::VectorXd yhat = H * st->mean + d;
  const Eigen::VectorXd v = y - yhat;
  Eigen::MatrixXd S = H * st->cov * H.transpose() + R;

  Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() != Eigen::Success) {
    S += 1e-12 * Eigen::MatrixXd::Identity(S.rows(), S.cols());
    llt.compute(S);
    if (llt.info() != Eigen::Success) throw std::runtime_error("kalman_update: S not PD");
  }

  const Eigen::MatrixXd K = st->cov * H.transpose() * llt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  st->mean = st->mean + K * v;
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(st->cov.rows(), st->cov.cols());
  st->cov = (I - K * H) * st->cov * (I - K * H).transpose() + K * R * K.transpose();

  return util::log_mvnorm_pdf(y, yhat, S);
}

inline double normal_pdf(double z) {
  constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934;  // 1/sqrt(2*pi)
  return kInvSqrt2Pi * std::exp(-0.5 * z * z);
}

inline double normal_cdf(double z) { return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))); }

inline double log_normal_pdf(double x, double mean, double var) {
  constexpr double kLog2Pi = 1.83787706640934533908193770912;  // log(2*pi)
  const double v = std::max(var, 1e-300);
  const double z = (x - mean) / std::sqrt(v);
  return -0.5 * (kLog2Pi + std::log(v) + z * z);
}

inline double log_normal_cdf(double z) {
  // Stable-ish log(Phi(z)) with a Mills-ratio tail approximation.
  if (z > 8.0) return 0.0;
  if (z < -10.0) {
    constexpr double kLogSqrt2Pi = 0.918938533204672741780329736406;  // 0.5*log(2*pi)
    const double zz = -z;
    return -0.5 * z * z - std::log(zz) - kLogSqrt2Pi;
  }
  const double p = normal_cdf(z);
  return (p > 0.0) ? std::log(p) : -std::numeric_limits<double>::infinity();
}

inline double log_normal_ccdf(double z) {
  // log(1 - Phi(z))
  if (z < -8.0) return 0.0;
  if (z > 10.0) {
    constexpr double kLogSqrt2Pi = 0.918938533204672741780329736406;  // 0.5*log(2*pi)
    return -0.5 * z * z - std::log(z) - kLogSqrt2Pi;
  }
  const double p = normal_cdf(z);
  const double q = 1.0 - p;
  if (q <= 0.0) return -std::numeric_limits<double>::infinity();
  return std::log(q);
}

inline double log_sum_exp2(double a, double b) {
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// Moment-projected update for a censored (lower-bounded) scalar measurement:
//   y = max(lower, intercept + h x) + eps,  eps ~ N(0, meas_var),
// where the prior on x is Gaussian (KalmanState).
//
// Returns log p(y | prior) and updates (mean, cov) to match posterior moments.
inline double kalman_update_censored_lower(const Eigen::RowVectorXd& h, double intercept,
                                          double lower, double meas_var, double y,
                                          KalmanState* st) {
  if (!st) throw std::invalid_argument("kalman_update_censored_lower: st null");
  if (!(meas_var > 0.0)) throw std::invalid_argument("kalman_update_censored_lower: meas_var<=0");
  if (h.size() != st->mean.size()) throw std::invalid_argument("kalman_update_censored_lower: h dim");
  if (st->cov.rows() != st->mean.size() || st->cov.cols() != st->mean.size()) {
    throw std::invalid_argument("kalman_update_censored_lower: cov dim");
  }

  const double yprime = y - intercept;
  const double c = lower - intercept;

  const Eigen::VectorXd hcol = h.transpose();
  const double m = (h * st->mean)(0);
  const Eigen::VectorXd Ph = st->cov * hcol;
  const double s2 = (h * Ph)(0);

  if (!(std::isfinite(s2)) || s2 < 1e-14) {
    const double mu = std::max(c, m);
    return log_normal_pdf(yprime, mu, meas_var);
  }

  const double s = std::sqrt(std::max(0.0, s2));
  const double alpha = (c - m) / s;
  const double log_w0 = log_normal_pdf(yprime, c, meas_var) + log_normal_cdf(alpha);

  const double var_y = s2 + meas_var;
  const double log_p_lin = log_normal_pdf(yprime, m, var_y);
  const double v_tilde = (s2 * meas_var) / (s2 + meas_var);
  const double sd_tilde = std::sqrt(std::max(0.0, v_tilde));
  const double mu_tilde = (meas_var * m + s2 * yprime) / (s2 + meas_var);
  const double beta = (c - mu_tilde) / std::max(1e-300, sd_tilde);
  const double log_mass1 = log_normal_ccdf(beta);
  const double log_w1 = log_p_lin + log_mass1;

  const double logZ = log_sum_exp2(log_w0, log_w1);
  const double w0 = std::exp(log_w0 - logZ);
  const double w1 = std::exp(log_w1 - logZ);

  // Component 0: u <= c. Likelihood is constant in u on this region, so posterior is truncated prior.
  double m0 = c;
  double v0 = 0.0;
  if (w0 > 1e-14) {
    const double P0 = normal_cdf(alpha);
    if (P0 > 1e-14) {
      const double lam0 = normal_pdf(alpha) / P0;
      m0 = m - s * lam0;
      v0 = s2 * (1.0 - alpha * lam0 - lam0 * lam0);
      v0 = std::max(0.0, v0);
    } else {
      // Extremely rare tail; weight is already tiny.
      m0 = c;
      v0 = 0.0;
    }
  }

  // Component 1: u > c. Posterior under linear measurement is N(mu_tilde, v_tilde), truncated to (c, +inf).
  double m1 = c;
  double v1 = 0.0;
  if (w1 > 1e-14) {
    const double mass1 = 1.0 - normal_cdf(beta);
    if (mass1 > 1e-14) {
      const double lam1 = normal_pdf(beta) / mass1;
      m1 = mu_tilde + sd_tilde * lam1;
      v1 = v_tilde * (1.0 + beta * lam1 - lam1 * lam1);
      v1 = std::max(0.0, v1);
    } else {
      m1 = c;
      v1 = 0.0;
    }
  }

  const double mpost = w0 * m0 + w1 * m1;
  const double vpost = w0 * (v0 + (m0 - mpost) * (m0 - mpost)) +
                       w1 * (v1 + (m1 - mpost) * (m1 - mpost));

  // Moment-projected update for x using posterior moments of u = h x.
  const Eigen::VectorXd k = Ph / s2;
  st->mean = st->mean + k * (mpost - m);
  st->cov = st->cov + (Ph * Ph.transpose()) * ((vpost - s2) / (s2 * s2));

  return logZ;
}

}  // namespace statespace
