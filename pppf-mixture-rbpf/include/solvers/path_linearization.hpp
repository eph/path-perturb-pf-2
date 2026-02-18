#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace solvers {

inline Eigen::MatrixXd finite_diff_jacobian(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x0,
    double eps = 1e-6) {
  const int n = static_cast<int>(x0.size());
  const Eigen::VectorXd f0 = f(x0);
  const int m = static_cast<int>(f0.size());
  Eigen::MatrixXd J(m, n);
  Eigen::VectorXd xp = x0;
  Eigen::VectorXd xm = x0;
  for (int j = 0; j < n; ++j) {
    const double h = eps * std::max(1.0, std::abs(x0(j)));
    xp(j) = x0(j) + h;
    xm(j) = x0(j) - h;
    const Eigen::VectorXd fp = f(xp);
    const Eigen::VectorXd fm = f(xm);
    if (fp.size() != m || fm.size() != m) throw std::runtime_error("finite_diff_jacobian: size");
    J.col(j) = (fp - fm) / (2.0 * h);
    xp(j) = x0(j);
    xm(j) = x0(j);
  }
  return J;
}

}  // namespace solvers

