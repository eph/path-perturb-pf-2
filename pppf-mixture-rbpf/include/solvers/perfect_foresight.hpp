#pragma once

#include "solvers/newton.hpp"

#include <Eigen/Dense>

#include <stdexcept>

namespace solvers {

// Minimal "extended path" helper: solve a stacked nonlinear system over a horizon.
//
// The caller provides a residual functor over the stacked vector Z.
// This is intentionally generic; model-specific helpers live in include/models/.
template <typename ResidualFunctor>
inline Eigen::VectorXd solve_extended_path(const Eigen::VectorXd& z0, ResidualFunctor residual,
                                           const NewtonOptions& opt = {}) {
  Eigen::VectorXd z = z0;
  auto f = [&](const Eigen::VectorXd& zin, Eigen::VectorXd* rout) { residual(zin, rout); };
  const NewtonResult res = newton_solve(&z, f, opt);
  if (!res.success) {
    throw std::runtime_error("solve_extended_path: Newton failed, residual=" +
                             std::to_string(res.final_residual_norm));
  }
  return z;
}

}  // namespace solvers

