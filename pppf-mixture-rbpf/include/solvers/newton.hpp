#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace solvers {

struct NewtonOptions {
  int max_iter = 50;
  double tol = 1e-10;
  double fd_eps = 1e-6;
  double step_shrink = 0.5;      // backtracking
  int max_backtrack = 20;
};

struct NewtonResult {
  bool success = false;
  int iterations = 0;
  double final_residual_norm = std::numeric_limits<double>::infinity();
};

// Functor must be callable as:
//   f(x, &residual)                 OR
//   f(x, &residual, &jacobian)
//
// If Jacobian is not provided, a finite-difference Jacobian is used.
template <typename Functor>
inline NewtonResult newton_solve(Eigen::VectorXd* x, Functor f, const NewtonOptions& opt = {}) {
  if (!x) throw std::invalid_argument("newton_solve: x is null");
  const int n = static_cast<int>(x->size());
  if (n == 0) throw std::invalid_argument("newton_solve: empty x");

  Eigen::VectorXd r(n);
  Eigen::MatrixXd J(n, n);

  constexpr bool kHasJac =
      std::is_invocable<Functor, const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::MatrixXd*>::value;
  static_assert(std::is_invocable<Functor, const Eigen::VectorXd&, Eigen::VectorXd*>::value,
                "newton_solve: Functor must be callable as f(x, residual*)");

  auto fd_jac = [&](const Eigen::VectorXd& xin, const Eigen::VectorXd& r0,
                    Eigen::MatrixXd* Jout) {
    Jout->setZero();
    Eigen::VectorXd xp = xin;
    Eigen::VectorXd xm = xin;
    Eigen::VectorXd rp(n), rm(n);
    for (int j = 0; j < n; ++j) {
      const double h = opt.fd_eps * std::max(1.0, std::abs(xin(j)));
      xp(j) = xin(j) + h;
      xm(j) = xin(j) - h;
      f(xp, &rp);
      f(xm, &rm);
      Jout->col(j) = (rp - rm) / (2.0 * h);
      xp(j) = xin(j);
      xm(j) = xin(j);
    }
  };

  NewtonResult out;
  for (int it = 0; it < opt.max_iter; ++it) {
    if constexpr (kHasJac) {
      f(*x, &r, &J);
    } else {
      f(*x, &r);
      fd_jac(*x, r, &J);
    }

    const double rn = r.norm();
    out.iterations = it;
    out.final_residual_norm = rn;
    if (rn < opt.tol) {
      out.success = true;
      return out;
    }

    Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-r);
    if (!dx.allFinite()) throw std::runtime_error("newton_solve: non-finite step");

    // Backtracking to ensure residual decreases.
    double step = 1.0;
    Eigen::VectorXd x_try(n), r_try(n);
    for (int bt = 0; bt <= opt.max_backtrack; ++bt) {
      x_try = *x + step * dx;
      f(x_try, &r_try);
      if (r_try.norm() <= (1.0 - 1e-4 * step) * rn || bt == opt.max_backtrack) {
        *x = x_try;
        r = r_try;
        break;
      }
      step *= opt.step_shrink;
    }
  }
  out.success = false;
  return out;
}

}  // namespace solvers
