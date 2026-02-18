#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace models {

struct NkParams {
  // Structural.
  double beta = 0.99;
  double sigma = 1.0;   // IS uses 1/sigma
  double kappa = 0.1;
  double phi_pi = 1.5;
  double phi_x = 0.25;
  double i_ss = 0.01;   // steady-state policy intercept
  double i_lower = 0.0;
  double r_ss = 0.01;   // steady-state natural rate (so i_ss=r_ss is a consistent steady state)

  // Exogenous AR(1).
  double rho_r = 0.9;
  double sigma_r = 0.01;
  double rho_nu = 0.8;
  double sigma_nu = 0.01;

  int horizon = 20;
  int max_regime_iter = 50;
  double regime_tol = 1e-12;
};

struct NkPath {
  std::vector<double> x;
  std::vector<double> pi;
  std::vector<double> i;
  std::vector<double> r;
  std::vector<double> nu;
  std::vector<int> bind;  // 1 if ELB binds at t, for t=0..H-1

  int H() const { return static_cast<int>(x.size()) - 1; }
};

inline NkPath solve_nk_occbin_path(const NkParams& p, double r0, double nu0,
                                   const std::vector<double>& eps_r,
                                   const std::vector<double>& eps_nu) {
  const int H = p.horizon;
  if (static_cast<int>(eps_r.size()) != H) throw std::invalid_argument("nk_occbin: eps_r size");
  if (static_cast<int>(eps_nu.size()) != H) throw std::invalid_argument("nk_occbin: eps_nu size");
  if (!(p.sigma > 0.0)) throw std::invalid_argument("nk_occbin: sigma<=0");

  NkPath path;
  path.x.assign(H + 1, 0.0);
  path.pi.assign(H + 1, 0.0);
  path.i.assign(H + 1, 0.0);
  path.r.assign(H + 1, 0.0);
  path.nu.assign(H + 1, 0.0);
  path.bind.assign(H, 0);

  path.r[0] = r0;
  path.nu[0] = nu0;
  for (int t = 0; t < H; ++t) {
    path.r[t + 1] = p.rho_r * path.r[t] + p.sigma_r * eps_r[t];
    path.nu[t + 1] = p.rho_nu * path.nu[t] + p.sigma_nu * eps_nu[t];
  }

  const double inv_sigma = 1.0 / p.sigma;

  auto solve_given_regime = [&](const std::vector<int>& bind, std::vector<double>* x,
                                std::vector<double>* pi) {
    x->assign(H + 1, 0.0);
    pi->assign(H + 1, 0.0);
    (*x)[H] = 0.0;
    (*pi)[H] = 0.0;

    for (int t = H - 1; t >= 0; --t) {
      Eigen::Matrix2d A;
      Eigen::Matrix2d B;
      Eigen::Vector2d c;
      B << -1.0, -inv_sigma, 0.0, -p.beta;
      c << 0.0, 0.0;
      const double r_n = p.r_ss + path.r[t];
      if (bind[t]) {
        // x_t - x_{t+1} + inv_sigma (i_lower - pi_{t+1} - r_t) = 0
        A << 1.0, 0.0, -p.kappa, 1.0;
        c(0) = inv_sigma * (p.i_lower - r_n);
      } else {
        // x_t - x_{t+1} + inv_sigma (i_ss + phi_pi pi_t + phi_x x_t + nu_t - pi_{t+1} - r_t) = 0
        A << 1.0 + inv_sigma * p.phi_x, inv_sigma * p.phi_pi, -p.kappa, 1.0;
        c(0) = inv_sigma * (p.i_ss + path.nu[t] - r_n);
      }
      // pi_t - beta pi_{t+1} - kappa x_t = 0
      c(1) = 0.0;

      const Eigen::Vector2d u_next((*x)[t + 1], (*pi)[t + 1]);
      const Eigen::Vector2d rhs = -(B * u_next + c);
      const Eigen::Vector2d u = A.fullPivLu().solve(rhs);
      (*x)[t] = u(0);
      (*pi)[t] = u(1);
    }
  };

  std::vector<int> bind = path.bind;
  std::vector<double> x, pi;
  for (int it = 0; it < p.max_regime_iter; ++it) {
    solve_given_regime(bind, &x, &pi);

    std::vector<int> bind_new = bind;
    for (int t = 0; t < H; ++t) {
      const double i_rule = p.i_ss + p.phi_pi * pi[t] + p.phi_x * x[t] + path.nu[t];
      bind_new[t] = (i_rule <= p.i_lower + p.regime_tol) ? 1 : 0;
    }

    if (bind_new == bind) {
      bind = bind_new;
      break;
    }
    bind = bind_new;
    if (it == p.max_regime_iter - 1) throw std::runtime_error("nk_occbin: regime did not converge");
  }

  // Final solve with converged regime.
  solve_given_regime(bind, &path.x, &path.pi);
  path.bind = bind;

  for (int t = 0; t < H; ++t) {
    const double i_rule = p.i_ss + p.phi_pi * path.pi[t] + p.phi_x * path.x[t] + path.nu[t];
    path.i[t] = bind[t] ? p.i_lower : std::max(p.i_lower, i_rule);
  }
  // Terminal i_H (not used).
  path.i[H] = p.i_ss + p.phi_pi * path.pi[H] + p.phi_x * path.x[H] + path.nu[H];

  return path;
}

// Variant of OccBin that conditions on the time-0 binding decision (i.e., whether the ELB binds
// at the first period of the solved path). This is useful for regime-mixture experiments near the
// kink, where one wants to compare or mix transitions conditional on bind[0]=0 versus bind[0]=1.
inline NkPath solve_nk_occbin_path_conditional_bind0(const NkParams& p, double r0, double nu0,
                                                     const std::vector<double>& eps_r,
                                                     const std::vector<double>& eps_nu,
                                                     int bind0) {
  const int H = p.horizon;
  if (static_cast<int>(eps_r.size()) != H) throw std::invalid_argument("nk_occbin_cond: eps_r size");
  if (static_cast<int>(eps_nu.size()) != H) throw std::invalid_argument("nk_occbin_cond: eps_nu size");
  if (!(p.sigma > 0.0)) throw std::invalid_argument("nk_occbin_cond: sigma<=0");
  if (!(bind0 == 0 || bind0 == 1)) throw std::invalid_argument("nk_occbin_cond: bind0 not in {0,1}");

  NkPath path;
  path.x.assign(H + 1, 0.0);
  path.pi.assign(H + 1, 0.0);
  path.i.assign(H + 1, 0.0);
  path.r.assign(H + 1, 0.0);
  path.nu.assign(H + 1, 0.0);
  path.bind.assign(H, 0);

  path.r[0] = r0;
  path.nu[0] = nu0;
  for (int t = 0; t < H; ++t) {
    path.r[t + 1] = p.rho_r * path.r[t] + p.sigma_r * eps_r[t];
    path.nu[t + 1] = p.rho_nu * path.nu[t] + p.sigma_nu * eps_nu[t];
  }

  const double inv_sigma = 1.0 / p.sigma;

  auto solve_given_regime = [&](const std::vector<int>& bind, std::vector<double>* x,
                                std::vector<double>* pi) {
    x->assign(H + 1, 0.0);
    pi->assign(H + 1, 0.0);
    (*x)[H] = 0.0;
    (*pi)[H] = 0.0;

    for (int t = H - 1; t >= 0; --t) {
      Eigen::Matrix2d A;
      Eigen::Matrix2d B;
      Eigen::Vector2d c;
      B << -1.0, -inv_sigma, 0.0, -p.beta;
      c << 0.0, 0.0;
      const double r_n = p.r_ss + path.r[t];
      if (bind[t]) {
        A << 1.0, 0.0, -p.kappa, 1.0;
        c(0) = inv_sigma * (p.i_lower - r_n);
      } else {
        A << 1.0 + inv_sigma * p.phi_x, inv_sigma * p.phi_pi, -p.kappa, 1.0;
        c(0) = inv_sigma * (p.i_ss + path.nu[t] - r_n);
      }
      c(1) = 0.0;

      const Eigen::Vector2d u_next((*x)[t + 1], (*pi)[t + 1]);
      const Eigen::Vector2d rhs = -(B * u_next + c);
      const Eigen::Vector2d u = A.fullPivLu().solve(rhs);
      (*x)[t] = u(0);
      (*pi)[t] = u(1);
    }
  };

  std::vector<int> bind = path.bind;
  bind[0] = bind0;
  std::vector<double> x, pi;
  for (int it = 0; it < p.max_regime_iter; ++it) {
    solve_given_regime(bind, &x, &pi);

    std::vector<int> bind_new = bind;
    bind_new[0] = bind0;
    for (int t = 1; t < H; ++t) {
      const double i_rule = p.i_ss + p.phi_pi * pi[t] + p.phi_x * x[t] + path.nu[t];
      bind_new[t] = (i_rule <= p.i_lower + p.regime_tol) ? 1 : 0;
    }

    if (bind_new == bind) {
      bind = bind_new;
      break;
    }
    bind = bind_new;
    if (it == p.max_regime_iter - 1) {
      throw std::runtime_error("nk_occbin_cond: regime did not converge");
    }
  }

  solve_given_regime(bind, &path.x, &path.pi);
  path.bind = bind;

  for (int t = 0; t < H; ++t) {
    const double i_rule = p.i_ss + p.phi_pi * path.pi[t] + p.phi_x * path.x[t] + path.nu[t];
    path.i[t] = bind[t] ? p.i_lower : std::max(p.i_lower, i_rule);
  }
  path.i[H] = p.i_ss + p.phi_pi * path.pi[H] + p.phi_x * path.x[H] + path.nu[H];
  return path;
}

}  // namespace models
