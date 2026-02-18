#pragma once

#include "models/nk_occbin.hpp"
#include "quadrature/sigma_points.hpp"
#include "solvers/path_linearization.hpp"
#include "statespace/gaussian_mixture.hpp"

#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace models {

// State ordering used in this repo: z = [x, pi, r^n, nu]^T.
inline Eigen::Vector4d nk_transition_markov(const NkParams& p, const Eigen::Vector4d& z_prev,
                                            double eps_r, double eps_nu) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  const NkPath path = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);

  Eigen::Vector4d z_next;
  z_next << path.x[0], path.pi[0], r_next, nu_next;
  return z_next;
}

inline Eigen::Vector4d nk_transition_markov_anticipated(const NkParams& p, const Eigen::Vector4d& z_prev,
                                                        double eps_r, double eps_nu,
                                                        double omega_r_next) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  // Anticipated shock object: insert a (single) future innovation at t+2 inside the OccBin path solve.
  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (p.horizon > 0) eps_r_path[0] = omega_r_next;

  const NkPath path = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);

  Eigen::Vector4d z_next;
  z_next << path.x[0], path.pi[0], r_next, nu_next;
  return z_next;
}

inline double nk_policy_rate(const NkParams& p, const Eigen::Vector4d& z) {
  const double i_rule = p.i_ss + p.phi_pi * z(1) + p.phi_x * z(0) + z(3);
  return std::max(p.i_lower, i_rule);
}

inline Eigen::Vector3d nk_observables(const NkParams& p, const Eigen::Vector4d& z) {
  // y = [pi, x, i]
  Eigen::Vector3d y;
  y << z(1), z(0), nk_policy_rate(p, z);
  return y;
}

// Builds a UT-indexed mixture of linear-Gaussian transitions around a reference z_prev.
//
// - Discrete mixture index integrates over a 1D anticipated-shock object omega (standard normal)
//   inserted into the perfect-foresight/OccBin path solve to approximate expectations.
// - Unexpected one-step Gaussian innovations (eps_r, eps_nu) are integrated analytically via Q.
inline statespace::GaussianMixtureTransition build_pppf_mixture(const NkParams& p,
                                                                const Eigen::Vector4d& z_ref,
                                                                double omega_mean = 0.0,
                                                                double omega_var = 1.0,
                                                                double unexpected_eps_r_mean = 0.0,
                                                                double unexpected_eps_r_var = 1.0,
                                                                double fd_eps = 1e-6) {
  // For a *probability* mixture (RBPF needs to sample indices), we require nonnegative weights.
  // The common UKF choice (alpha=1e-2, kappa=0) yields a large negative w0 in 1D, so we instead
  // use a UT parameterization with lambda=2 => weights {2/3, 1/6, 1/6} and points {μ, μ±sqrt(3)σ}.
  const double alpha_ut = 1.0;
  const double kappa_ut = 2.0;
  const double beta_ut = 2.0;
  const auto sp =
      quadrature::unscented_sigma_points_1d(omega_mean, omega_var, alpha_ut, kappa_ut, beta_ut);
  const int K = static_cast<int>(sp.points.rows());

  statespace::GaussianMixtureTransition mix;
  mix.components.resize(K);
  mix.weights.resize(K);

  for (int k = 0; k < K; ++k) {
    const double omega_node = sp.points(k, 0);
    mix.weights[k] = sp.w_mean(k);

    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      const Eigen::Vector4d z_out = nk_transition_markov_anticipated(p, z4, 0.0, 0.0, omega_node);
      return z_out;
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    Eigen::Vector4d z_mean = nk_transition_markov_anticipated(p, z_ref, 0.0, 0.0, omega_node);

    // Unexpected-shock mean shift for IRFs (filter uses eps_r_mean=0).
    const double h = 1e-6;
    const Eigen::Vector4d zrp = nk_transition_markov_anticipated(p, z_ref, +h, 0.0, omega_node);
    const Eigen::Vector4d zrm = nk_transition_markov_anticipated(p, z_ref, -h, 0.0, omega_node);
    const Eigen::Vector4d Br = (zrp - zrm) / (2.0 * h);  // derivative wrt eps_r
    z_mean += Br * unexpected_eps_r_mean;

    const Eigen::Vector4d a = z_mean - A * z_ref;

    // Noise mapping for unexpected eps_r and eps_nu.
    const Eigen::Vector4d znp = nk_transition_markov_anticipated(p, z_ref, 0.0, +h, omega_node);
    const Eigen::Vector4d znm = nk_transition_markov_anticipated(p, z_ref, 0.0, -h, omega_node);
    const Eigen::Vector4d Bn = (znp - znm) / (2.0 * h);  // derivative wrt eps_nu

    Eigen::Matrix4d Q = unexpected_eps_r_var * (Br * Br.transpose()) + (Bn * Bn.transpose());
    Q += 1e-10 * Eigen::Matrix4d::Identity();

    mix.components[k].A = A;
    mix.components[k].a = a;
    mix.components[k].Q = Q;
  }

  // Normalize weights (guard against tiny drift).
  double sw = 0.0;
  for (double w : mix.weights) sw += w;
  for (double& w : mix.weights) w /= sw;
  return mix;
}

}  // namespace models
