#pragma once

#include "util/stats.hpp"

#include <Eigen/Dense>

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

}  // namespace statespace

