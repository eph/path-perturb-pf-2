#pragma once

#include "models/nk_occbin.hpp"
#include "quadrature/sigma_points.hpp"
#include "solvers/path_linearization.hpp"
#include "statespace/gaussian_mixture.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <optional>
#include <vector>

namespace models {

enum class PppfRegimeMode {
  EndogenousByNode = 0,
  FixedBind0FromMean = 1,
  MixtureBind0 = 2,
};

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

// One-step transition with an anticipated-shock object omega embedded inside the EP/OccBin path solve.
//
// Timing convention:
// - eps_r, eps_nu are the realized one-step innovations that move (r^n, nu) from t to t+1.
// - omega encodes a *short horizon* of anticipated innovations starting at t+2 (i.e., inside the
//   horizon path solve conditional on r_{t+1}, nu_{t+1}). This avoids double-counting eps_{t+1}.
//
// Omega layout:
//   omega = [omega_r(0..L-1), omega_nu(0..L-1)] with length 2L (L may be 0).
// These are inserted into eps_*_path[0..L-1], which correspond to innovations at t+2..t+L+1.
inline Eigen::Vector4d nk_transition_markov_anticipated(const NkParams& p, const Eigen::Vector4d& z_prev,
                                                        double eps_r, double eps_nu,
                                                        const Eigen::VectorXd& omega) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (omega.size() > 0) {
    if (omega.size() % 2 != 0) throw std::invalid_argument("nk_transition_markov_anticipated: omega odd");
    const int L = static_cast<int>(omega.size() / 2);
    if (L > p.horizon) throw std::invalid_argument("nk_transition_markov_anticipated: omega L>horizon");
    for (int i = 0; i < L; ++i) {
      eps_r_path[i] = omega(i);
      eps_nu_path[i] = omega(L + i);
    }
  }

  const NkPath path = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);

  Eigen::Vector4d z_next;
  z_next << path.x[0], path.pi[0], r_next, nu_next;
  return z_next;
}

inline Eigen::Vector4d nk_transition_markov_anticipated_conditional_bind0(
    const NkParams& p, const Eigen::Vector4d& z_prev, double eps_r, double eps_nu,
    const Eigen::VectorXd& omega, int bind0) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (omega.size() > 0) {
    if (omega.size() % 2 != 0) throw std::invalid_argument("nk_transition_bind0: omega odd");
    const int L = static_cast<int>(omega.size() / 2);
    if (L > p.horizon) throw std::invalid_argument("nk_transition_bind0: omega L>horizon");
    for (int i = 0; i < L; ++i) {
      eps_r_path[i] = omega(i);
      eps_nu_path[i] = omega(L + i);
    }
  }

  const NkPath path =
      solve_nk_occbin_path_conditional_bind0(p, r_next, nu_next, eps_r_path, eps_nu_path, bind0);

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
// - Discrete mixture index integrates over a low-dimensional anticipated-shock object omega
//   (Gaussian) inserted into the perfect-foresight/OccBin path solve to approximate expectations.
// - Unexpected one-step Gaussian innovations (eps_r, eps_nu) are integrated analytically via Q.
inline statespace::GaussianMixtureTransition build_pppf_mixture(const NkParams& p,
                                                                const Eigen::Vector4d& z_ref,
                                                                int omega_horizon = 1,
                                                                double omega_var = 1.0,
                                                                double unexpected_eps_r_mean = 0.0,
                                                                double unexpected_eps_r_var = 1.0,
                                                                double unexpected_eps_nu_var = 1.0,
                                                                PppfRegimeMode regime_mode =
                                                                    PppfRegimeMode::EndogenousByNode,
                                                                double fd_eps = 1e-6) {
  if (omega_horizon < 0) throw std::invalid_argument("build_pppf_mixture: omega_horizon<0");
  if (!(omega_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: omega_var<0");
  if (!(unexpected_eps_r_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: eps_r_var<0");
  if (!(unexpected_eps_nu_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: eps_nu_var<0");

  const int L = std::min(omega_horizon, p.horizon);
  const int d = 2 * L;

  const auto normal_cdf = [&](double z) { return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))); };

  // Special case: omega is degenerate (or absent) => a single component (exactly), which is
  // important for the "all methods coincide" no-ELB sanity check.
  if (d == 0 || omega_var == 0.0) {
    const Eigen::VectorXd omega0 = Eigen::VectorXd::Zero(d);
    statespace::GaussianMixtureTransition mix;
    mix.components.resize(1);
    mix.weights = {1.0};

    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      const Eigen::Vector4d z_out = nk_transition_markov_anticipated(p, z4, 0.0, 0.0, omega0);
      return z_out;
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    Eigen::Vector4d z_mean = nk_transition_markov_anticipated(p, z_ref, 0.0, 0.0, omega0);

    const double h = 1e-6;
    const Eigen::Vector4d zrp = nk_transition_markov_anticipated(p, z_ref, +h, 0.0, omega0);
    const Eigen::Vector4d zrm = nk_transition_markov_anticipated(p, z_ref, -h, 0.0, omega0);
    const Eigen::Vector4d Br = (zrp - zrm) / (2.0 * h);
    z_mean += Br * unexpected_eps_r_mean;

    const Eigen::Vector4d a = z_mean - A * z_ref;

    const Eigen::Vector4d znp = nk_transition_markov_anticipated(p, z_ref, 0.0, +h, omega0);
    const Eigen::Vector4d znm = nk_transition_markov_anticipated(p, z_ref, 0.0, -h, omega0);
    const Eigen::Vector4d Bn = (znp - znm) / (2.0 * h);

    Eigen::Matrix4d Q =
        unexpected_eps_r_var * (Br * Br.transpose()) + unexpected_eps_nu_var * (Bn * Bn.transpose());
    Q += 1e-10 * Eigen::Matrix4d::Identity();

    mix.components[0].A = A;
    mix.components[0].a = a;
    mix.components[0].Q = Q;
    return mix;
  }

  // For a *probability* mixture (RBPF needs to sample indices), we require nonnegative weights.
  // Using alpha=1 and kappa=2 yields lambda=2 (hence w0>0) and positive weights in any dimension.
  const Eigen::VectorXd omega_mean = Eigen::VectorXd::Zero(d);
  const Eigen::MatrixXd omega_cov = omega_var * Eigen::MatrixXd::Identity(d, d);
  const double alpha_ut = 1.0;
  const double kappa_ut = 2.0;
  const double beta_ut = 2.0;
  const auto sp = quadrature::unscented_sigma_points(omega_mean, omega_cov, alpha_ut, kappa_ut, beta_ut);
  const int K = static_cast<int>(sp.points.rows());

  statespace::GaussianMixtureTransition mix;
  mix.components.clear();
  mix.weights.clear();

  int bind0_fixed = 0;
  if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
    std::vector<double> eps_r_path(p.horizon, 0.0);
    std::vector<double> eps_nu_path(p.horizon, 0.0);
    const double r_next = p.rho_r * z_ref(2);
    const double nu_next = p.rho_nu * z_ref(3);
    const NkPath path0 = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);
    bind0_fixed = path0.bind[0];
  }

  struct LinTrans {
    Eigen::MatrixXd A;
    Eigen::Vector4d a;
    Eigen::Matrix4d Q;
    Eigen::Vector4d z_mean;
  };

  auto linearize_one = [&](const Eigen::VectorXd& omega_node,
                           std::optional<int> bind0_override) -> LinTrans {
    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      if (bind0_override.has_value()) {
        return nk_transition_markov_anticipated_conditional_bind0(p, z4, 0.0, 0.0, omega_node,
                                                                  *bind0_override);
      }
      if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
        return nk_transition_markov_anticipated_conditional_bind0(p, z4, 0.0, 0.0, omega_node, bind0_fixed);
      }
      return nk_transition_markov_anticipated(p, z4, 0.0, 0.0, omega_node);
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    Eigen::Vector4d z_mean;
    if (bind0_override.has_value()) {
      z_mean = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, 0.0, omega_node,
                                                                  *bind0_override);
    } else if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
      z_mean = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, 0.0, omega_node, bind0_fixed);
    } else {
      z_mean = nk_transition_markov_anticipated(p, z_ref, 0.0, 0.0, omega_node);
    }

    const double h = 1e-6;
    Eigen::Vector4d zrp, zrm, znp, znm;
    if (bind0_override.has_value()) {
      zrp = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, +h, 0.0, omega_node, *bind0_override);
      zrm = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, -h, 0.0, omega_node, *bind0_override);
      znp = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, +h, omega_node, *bind0_override);
      znm = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, -h, omega_node, *bind0_override);
    } else if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
      zrp = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, +h, 0.0, omega_node, bind0_fixed);
      zrm = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, -h, 0.0, omega_node, bind0_fixed);
      znp = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, +h, omega_node, bind0_fixed);
      znm = nk_transition_markov_anticipated_conditional_bind0(p, z_ref, 0.0, -h, omega_node, bind0_fixed);
    } else {
      zrp = nk_transition_markov_anticipated(p, z_ref, +h, 0.0, omega_node);
      zrm = nk_transition_markov_anticipated(p, z_ref, -h, 0.0, omega_node);
      znp = nk_transition_markov_anticipated(p, z_ref, 0.0, +h, omega_node);
      znm = nk_transition_markov_anticipated(p, z_ref, 0.0, -h, omega_node);
    }

    const Eigen::Vector4d Br = (zrp - zrm) / (2.0 * h);
    z_mean += Br * unexpected_eps_r_mean;
    const Eigen::Vector4d a = z_mean - A * z_ref;
    const Eigen::Vector4d Bn = (znp - znm) / (2.0 * h);

    Eigen::Matrix4d Q =
        unexpected_eps_r_var * (Br * Br.transpose()) + unexpected_eps_nu_var * (Bn * Bn.transpose());
    Q += 1e-10 * Eigen::Matrix4d::Identity();
    return LinTrans{A, a, Q, z_mean};
  };

  for (int k = 0; k < K; ++k) {
    const Eigen::VectorXd omega_node = sp.points.row(k).transpose();
    const double w_node = sp.w_mean(k);
    if (w_node < 0.0) {
      throw std::runtime_error("build_pppf_mixture: negative quadrature weight (not a mixture)");
    }

    if (regime_mode != PppfRegimeMode::MixtureBind0) {
      const LinTrans lt = linearize_one(omega_node, std::nullopt);
      mix.components.push_back(statespace::LinearGaussianTransition{lt.A, lt.a, lt.Q});
      mix.weights.push_back(w_node);
      continue;
    }

    const LinTrans lt_slack = linearize_one(omega_node, /*bind0_override=*/0);
    const LinTrans lt_bind = linearize_one(omega_node, /*bind0_override=*/1);

    const Eigen::RowVector4d h_rule =
        (Eigen::RowVector4d() << p.phi_x, p.phi_pi, 0.0, 1.0).finished();
    const double mu_rule = p.i_ss + (h_rule * lt_slack.z_mean)(0);
    const double var_rule = (h_rule * lt_slack.Q * h_rule.transpose())(0, 0);
    const double sd_rule = std::sqrt(std::max(0.0, var_rule));

    double p_bind = 0.0;
    if (sd_rule < 1e-12) {
      p_bind = (mu_rule <= p.i_lower) ? 1.0 : 0.0;
    } else {
      p_bind = normal_cdf((p.i_lower - mu_rule) / sd_rule);
    }
    p_bind = std::min(1.0, std::max(0.0, p_bind));

    const double w_bind = w_node * p_bind;
    const double w_slack = w_node * (1.0 - p_bind);

    if (w_slack > 0.0) {
      mix.components.push_back(statespace::LinearGaussianTransition{lt_slack.A, lt_slack.a, lt_slack.Q});
      mix.weights.push_back(w_slack);
    }
    if (w_bind > 0.0) {
      mix.components.push_back(statespace::LinearGaussianTransition{lt_bind.A, lt_bind.a, lt_bind.Q});
      mix.weights.push_back(w_bind);
    }
  }

  // Normalize weights (guard against tiny drift).
  double sw = 0.0;
  for (double w : mix.weights) sw += w;
  for (double& w : mix.weights) w /= sw;
  return mix;
}

}  // namespace models
