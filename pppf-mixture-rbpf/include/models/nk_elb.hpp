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

enum class PppfOmegaMode {
  ROnly = 0,
  RAndNu = 1,
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
                                                        const Eigen::VectorXd& omega,
                                                        PppfOmegaMode omega_mode =
                                                            PppfOmegaMode::RAndNu) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (omega.size() > 0) {
    int L = 0;
    if (omega_mode == PppfOmegaMode::RAndNu) {
      if (omega.size() % 2 != 0) throw std::invalid_argument("nk_transition_markov_anticipated: omega odd");
      L = static_cast<int>(omega.size() / 2);
    } else {
      L = static_cast<int>(omega.size());
    }
    if (L > p.horizon) throw std::invalid_argument("nk_transition_markov_anticipated: omega L>horizon");
    if (omega_mode == PppfOmegaMode::RAndNu) {
      for (int i = 0; i < L; ++i) {
        eps_r_path[i] = omega(i);
        eps_nu_path[i] = omega(L + i);
      }
    } else {
      for (int i = 0; i < L; ++i) eps_r_path[i] = omega(i);
    }
  }

  const NkPath path = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);

  Eigen::Vector4d z_next;
  z_next << path.x[0], path.pi[0], r_next, nu_next;
  return z_next;
}

inline Eigen::Vector4d nk_transition_markov_anticipated_conditional_bind0(
    const NkParams& p, const Eigen::Vector4d& z_prev, double eps_r, double eps_nu,
    const Eigen::VectorXd& omega, int bind0,
    PppfOmegaMode omega_mode = PppfOmegaMode::RAndNu) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (omega.size() > 0) {
    int L = 0;
    if (omega_mode == PppfOmegaMode::RAndNu) {
      if (omega.size() % 2 != 0) throw std::invalid_argument("nk_transition_bind0: omega odd");
      L = static_cast<int>(omega.size() / 2);
    } else {
      L = static_cast<int>(omega.size());
    }
    if (L > p.horizon) throw std::invalid_argument("nk_transition_bind0: omega L>horizon");
    if (omega_mode == PppfOmegaMode::RAndNu) {
      for (int i = 0; i < L; ++i) {
        eps_r_path[i] = omega(i);
        eps_nu_path[i] = omega(L + i);
      }
    } else {
      for (int i = 0; i < L; ++i) eps_r_path[i] = omega(i);
    }
  }

  const NkPath path =
      solve_nk_occbin_path_conditional_bind0(p, r_next, nu_next, eps_r_path, eps_nu_path, bind0);

  Eigen::Vector4d z_next;
  z_next << path.x[0], path.pi[0], r_next, nu_next;
  return z_next;
}

inline Eigen::Vector4d nk_transition_markov_anticipated_given_bind(const NkParams& p,
                                                                   const Eigen::Vector4d& z_prev,
                                                                   double eps_r, double eps_nu,
                                                                   const Eigen::VectorXd& omega,
                                                                   const std::vector<int>& bind,
                                                                   PppfOmegaMode omega_mode =
                                                                       PppfOmegaMode::RAndNu) {
  const double r_next = p.rho_r * z_prev(2) + p.sigma_r * eps_r;
  const double nu_next = p.rho_nu * z_prev(3) + p.sigma_nu * eps_nu;

  std::vector<double> eps_r_path(p.horizon, 0.0);
  std::vector<double> eps_nu_path(p.horizon, 0.0);
  if (omega.size() > 0) {
    int L = 0;
    if (omega_mode == PppfOmegaMode::RAndNu) {
      if (omega.size() % 2 != 0) throw std::invalid_argument("nk_transition_bind: omega odd");
      L = static_cast<int>(omega.size() / 2);
    } else {
      L = static_cast<int>(omega.size());
    }
    if (L > p.horizon) throw std::invalid_argument("nk_transition_bind: omega L>horizon");
    if (omega_mode == PppfOmegaMode::RAndNu) {
      for (int i = 0; i < L; ++i) {
        eps_r_path[i] = omega(i);
        eps_nu_path[i] = omega(L + i);
      }
    } else {
      for (int i = 0; i < L; ++i) eps_r_path[i] = omega(i);
    }
  }

  const NkPath path = solve_nk_occbin_path_given_bind(p, r_next, nu_next, eps_r_path, eps_nu_path, bind);
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
                                                                PppfOmegaMode omega_mode =
                                                                    PppfOmegaMode::ROnly,
                                                                PppfRegimeMode regime_mode =
                                                                    PppfRegimeMode::EndogenousByNode,
                                                                double fd_eps = 1e-6) {
  if (omega_horizon < 0) throw std::invalid_argument("build_pppf_mixture: omega_horizon<0");
  if (!(omega_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: omega_var<0");
  if (!(unexpected_eps_r_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: eps_r_var<0");
  if (!(unexpected_eps_nu_var >= 0.0)) throw std::invalid_argument("build_pppf_mixture: eps_nu_var<0");

  const int L = std::min(omega_horizon, p.horizon);
  const int d = (omega_mode == PppfOmegaMode::RAndNu) ? (2 * L) : L;

  auto unpack_omega_paths = [&](const Eigen::VectorXd& omega_node, std::vector<double>& eps_r_path,
                                std::vector<double>& eps_nu_path) {
    if (omega_node.size() == 0) return;

    int L0 = 0;
    if (omega_mode == PppfOmegaMode::RAndNu) {
      if (omega_node.size() % 2 != 0) {
        throw std::invalid_argument("build_pppf_mixture: omega node has odd length");
      }
      L0 = static_cast<int>(omega_node.size() / 2);
    } else {
      L0 = static_cast<int>(omega_node.size());
    }
    if (L0 > p.horizon) {
      throw std::invalid_argument("build_pppf_mixture: omega node exceeds horizon");
    }

    for (int i = 0; i < L0; ++i) eps_r_path[i] = omega_node(i);
    if (omega_mode == PppfOmegaMode::RAndNu) {
      for (int i = 0; i < L0; ++i) eps_nu_path[i] = omega_node(L0 + i);
    }
  };
  auto shocked_anchor_state = [&](double eps_r, double eps_nu) {
    return std::pair<double, double>{p.rho_r * z_ref(2) + p.sigma_r * eps_r,
                                     p.rho_nu * z_ref(3) + p.sigma_nu * eps_nu};
  };
  auto shadow_gap = [&](const Eigen::Vector4d& z) {
    return p.i_ss + p.phi_pi * z(1) + p.phi_x * z(0) + z(3) - p.i_lower;
  };
  auto elb_risk_score = [&]() {
    const Eigen::VectorXd omega0 = Eigen::VectorXd::Zero(0);
    const Eigen::Vector4d z_det =
        nk_transition_markov_anticipated(p, z_ref, unexpected_eps_r_mean, 0.0, omega0, omega_mode);
    return std::min(shadow_gap(z_ref), shadow_gap(z_det));
  };
  const bool use_discrete_shock_closure =
      (unexpected_eps_r_var > 0.0 || unexpected_eps_nu_var > 0.0) && (elb_risk_score() <= 0.02);

  struct ShockNode {
    double eps_r = 0.0;
    double eps_nu = 0.0;
    double weight = 1.0;
  };
  std::vector<ShockNode> shock_nodes;
  if (use_discrete_shock_closure) {
    Eigen::Vector2d eps_mean = Eigen::Vector2d::Zero();
    eps_mean(0) = unexpected_eps_r_mean;
    Eigen::Matrix2d eps_cov = Eigen::Matrix2d::Zero();
    eps_cov(0, 0) = unexpected_eps_r_var;
    eps_cov(1, 1) = unexpected_eps_nu_var;
    const auto eps_sp = quadrature::unscented_sigma_points(eps_mean, eps_cov, /*alpha=*/1.0,
                                                           /*kappa=*/2.0, /*beta=*/2.0);
    shock_nodes.reserve(eps_sp.points.rows());
    for (int q = 0; q < eps_sp.points.rows(); ++q) {
      shock_nodes.push_back(ShockNode{eps_sp.points(q, 0), eps_sp.points(q, 1), eps_sp.w_mean(q)});
    }
  } else {
    shock_nodes.push_back(ShockNode{unexpected_eps_r_mean, 0.0, 1.0});
  }

  // Special case: omega is degenerate (or absent) => a single component (exactly), which is
  // important for the "all methods coincide" no-ELB sanity check.
  if ((d == 0 || omega_var == 0.0) && !use_discrete_shock_closure) {
    const Eigen::VectorXd omega0 = Eigen::VectorXd::Zero(d);
    const auto [r_next0, nu_next0] = shocked_anchor_state(unexpected_eps_r_mean, 0.0);
    std::vector<double> eps_r_path(p.horizon, 0.0);
    std::vector<double> eps_nu_path(p.horizon, 0.0);
    const NkPath base_path =
        solve_nk_occbin_path(p, r_next0, nu_next0, eps_r_path, eps_nu_path);
    const std::vector<int> bind = base_path.bind;

    statespace::GaussianMixtureTransition mix;
    mix.components.resize(1);
    mix.weights = {1.0};

    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      return nk_transition_markov_anticipated_given_bind(p, z4, 0.0, 0.0, omega0, bind, omega_mode);
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    Eigen::Vector4d z_mean =
        nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, 0.0, omega0, bind, omega_mode);

    const double h = 1e-6;
    const Eigen::Vector4d zrp =
        nk_transition_markov_anticipated_given_bind(p, z_ref, +h, 0.0, omega0, bind, omega_mode);
    const Eigen::Vector4d zrm =
        nk_transition_markov_anticipated_given_bind(p, z_ref, -h, 0.0, omega0, bind, omega_mode);
    const Eigen::Vector4d Br = (zrp - zrm) / (2.0 * h);
    z_mean += Br * unexpected_eps_r_mean;

    const Eigen::Vector4d a = z_mean - A * z_ref;

    const Eigen::Vector4d znp =
        nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, +h, omega0, bind, omega_mode);
    const Eigen::Vector4d znm =
        nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, -h, omega0, bind, omega_mode);
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
  std::vector<Eigen::VectorXd> omega_nodes;
  std::vector<double> omega_weights;
  if (d == 0 || omega_var == 0.0) {
    omega_nodes.push_back(Eigen::VectorXd::Zero(d));
    omega_weights.push_back(1.0);
  } else {
    const Eigen::VectorXd omega_mean = Eigen::VectorXd::Zero(d);
    const Eigen::MatrixXd omega_cov = omega_var * Eigen::MatrixXd::Identity(d, d);
    const double alpha_ut = 1.0;
    const double kappa_ut = 2.0;
    const double beta_ut = 2.0;
    const auto sp =
        quadrature::unscented_sigma_points(omega_mean, omega_cov, alpha_ut, kappa_ut, beta_ut);
    omega_nodes.reserve(sp.points.rows());
    omega_weights.reserve(sp.points.rows());
    for (int k = 0; k < sp.points.rows(); ++k) {
      omega_nodes.push_back(sp.points.row(k).transpose());
      omega_weights.push_back(sp.w_mean(k));
    }
  }

  statespace::GaussianMixtureTransition mix;
  mix.components.clear();
  mix.weights.clear();

  int bind0_fixed = 0;
  if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
    std::vector<double> eps_r_path(p.horizon, 0.0);
    std::vector<double> eps_nu_path(p.horizon, 0.0);
    const auto [r_next, nu_next] = shocked_anchor_state(unexpected_eps_r_mean, 0.0);
    const NkPath path0 = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);
    bind0_fixed = path0.bind[0];
  }

  struct LinTrans {
    Eigen::MatrixXd A;
    Eigen::Vector4d a;
    Eigen::Matrix4d Q;
    Eigen::Vector4d z_mean;
  };

  auto regime_path_for_anchor = [&](const Eigen::VectorXd& omega_node, double eps_r_anchor,
                                    double eps_nu_anchor,
                                    std::optional<int> bind0_override) -> std::vector<int> {
    std::vector<double> eps_r0(p.horizon, 0.0);
    std::vector<double> eps_nu0(p.horizon, 0.0);
    unpack_omega_paths(omega_node, eps_r0, eps_nu0);
    const auto [r_next0, nu_next0] = shocked_anchor_state(eps_r_anchor, eps_nu_anchor);
    NkPath base_path;
    if (bind0_override.has_value()) {
      base_path =
          solve_nk_occbin_path_conditional_bind0(p, r_next0, nu_next0, eps_r0, eps_nu0, *bind0_override);
    } else if (regime_mode == PppfRegimeMode::FixedBind0FromMean) {
      base_path = solve_nk_occbin_path_conditional_bind0(p, r_next0, nu_next0, eps_r0, eps_nu0, bind0_fixed);
    } else {
      base_path = solve_nk_occbin_path(p, r_next0, nu_next0, eps_r0, eps_nu0);
    }
    return base_path.bind;
  };

  auto linearize_gaussianized = [&](const Eigen::VectorXd& omega_node,
                                    std::optional<int> bind0_override) -> LinTrans {
    const std::vector<int> bind_path =
        regime_path_for_anchor(omega_node, unexpected_eps_r_mean, 0.0, bind0_override);

    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      return nk_transition_markov_anticipated_given_bind(p, z4, 0.0, 0.0, omega_node, bind_path, omega_mode);
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    Eigen::Vector4d z_mean;
    z_mean =
        nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, 0.0, omega_node, bind_path, omega_mode);
    const double h = 1e-6;
    Eigen::Vector4d zrp, zrm, znp, znm;
    zrp = nk_transition_markov_anticipated_given_bind(p, z_ref, +h, 0.0, omega_node, bind_path, omega_mode);
    zrm = nk_transition_markov_anticipated_given_bind(p, z_ref, -h, 0.0, omega_node, bind_path, omega_mode);
    znp = nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, +h, omega_node, bind_path, omega_mode);
    znm = nk_transition_markov_anticipated_given_bind(p, z_ref, 0.0, -h, omega_node, bind_path, omega_mode);

    const Eigen::Vector4d Br = (zrp - zrm) / (2.0 * h);
    z_mean += Br * unexpected_eps_r_mean;
    const Eigen::Vector4d a = z_mean - A * z_ref;
    const Eigen::Vector4d Bn = (znp - znm) / (2.0 * h);

    Eigen::Matrix4d Q =
        unexpected_eps_r_var * (Br * Br.transpose()) + unexpected_eps_nu_var * (Bn * Bn.transpose());
    Q += 1e-10 * Eigen::Matrix4d::Identity();
    return LinTrans{A, a, Q, z_mean};
  };

  auto linearize_fixed_shock = [&](const Eigen::VectorXd& omega_node, double eps_r_anchor,
                                   double eps_nu_anchor,
                                   std::optional<int> bind0_override) -> LinTrans {
    const std::vector<int> bind_path =
        regime_path_for_anchor(omega_node, eps_r_anchor, eps_nu_anchor, bind0_override);

    auto g = [&](const Eigen::VectorXd& z_in) -> Eigen::VectorXd {
      const Eigen::Vector4d z4 = z_in;
      return nk_transition_markov_anticipated_given_bind(p, z4, eps_r_anchor, eps_nu_anchor, omega_node,
                                                         bind_path, omega_mode);
    };

    const Eigen::MatrixXd A = solvers::finite_diff_jacobian(g, z_ref, fd_eps);
    const Eigen::Vector4d z_mean =
        nk_transition_markov_anticipated_given_bind(p, z_ref, eps_r_anchor, eps_nu_anchor, omega_node,
                                                    bind_path, omega_mode);
    const Eigen::Vector4d a = z_mean - A * z_ref;
    const Eigen::Matrix4d Q = 1e-10 * Eigen::Matrix4d::Identity();
    return LinTrans{A, a, Q, z_mean};
  };

  auto bind0_probability_cubature = [&](const Eigen::VectorXd& omega_node) -> double {
    Eigen::Vector2d eps_mean = Eigen::Vector2d::Zero();
    eps_mean(0) = unexpected_eps_r_mean;
    Eigen::Matrix2d eps_cov = Eigen::Matrix2d::Zero();
    eps_cov(0, 0) = unexpected_eps_r_var;
    eps_cov(1, 1) = unexpected_eps_nu_var;

    if (unexpected_eps_r_var == 0.0 && unexpected_eps_nu_var == 0.0) {
      const Eigen::Vector4d z_next =
          nk_transition_markov_anticipated(p, z_ref, unexpected_eps_r_mean, 0.0, omega_node, omega_mode);
      return nk_policy_rate(p, z_next) <= p.i_lower + p.regime_tol ? 1.0 : 0.0;
    }

    const auto eps_sp = quadrature::unscented_sigma_points(eps_mean, eps_cov, /*alpha=*/1.0,
                                                           /*kappa=*/2.0, /*beta=*/2.0);
    double p_bind = 0.0;
    std::vector<double> eps_r_path(p.horizon, 0.0);
    std::vector<double> eps_nu_path(p.horizon, 0.0);
    unpack_omega_paths(omega_node, eps_r_path, eps_nu_path);

    for (int q = 0; q < eps_sp.points.rows(); ++q) {
      const double eps_r = eps_sp.points(q, 0);
      const double eps_nu = eps_sp.points(q, 1);
      const double r_next = p.rho_r * z_ref(2) + p.sigma_r * eps_r;
      const double nu_next = p.rho_nu * z_ref(3) + p.sigma_nu * eps_nu;
      const NkPath path = solve_nk_occbin_path(p, r_next, nu_next, eps_r_path, eps_nu_path);
      p_bind += eps_sp.w_mean(q) * static_cast<double>(path.bind[0] != 0);
    }
    return std::min(1.0, std::max(0.0, p_bind));
  };

  for (int k = 0; k < static_cast<int>(omega_nodes.size()); ++k) {
    const Eigen::VectorXd& omega_node = omega_nodes[k];
    const double w_node = omega_weights[k];
    if (w_node < 0.0) {
      throw std::runtime_error("build_pppf_mixture: negative quadrature weight (not a mixture)");
    }

    if (regime_mode != PppfRegimeMode::MixtureBind0) {
      for (const auto& shock_node : shock_nodes) {
        if (shock_node.weight <= 0.0) continue;
        const LinTrans lt = use_discrete_shock_closure
                                ? linearize_fixed_shock(omega_node, shock_node.eps_r, shock_node.eps_nu,
                                                        std::nullopt)
                                : linearize_gaussianized(omega_node, std::nullopt);
        mix.components.push_back(statespace::LinearGaussianTransition{lt.A, lt.a, lt.Q});
        mix.weights.push_back(w_node * shock_node.weight);
        if (!use_discrete_shock_closure) break;
      }
      continue;
    }

    if (use_discrete_shock_closure) {
      for (const auto& shock_node : shock_nodes) {
        if (shock_node.weight <= 0.0) continue;
        const LinTrans lt =
            linearize_fixed_shock(omega_node, shock_node.eps_r, shock_node.eps_nu, std::nullopt);
        mix.components.push_back(statespace::LinearGaussianTransition{lt.A, lt.a, lt.Q});
        mix.weights.push_back(w_node * shock_node.weight);
      }
      continue;
    }

    const LinTrans lt_slack = linearize_gaussianized(omega_node, /*bind0_override=*/0);
    const LinTrans lt_bind = linearize_gaussianized(omega_node, /*bind0_override=*/1);

    const double p_bind = bind0_probability_cubature(omega_node);

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
