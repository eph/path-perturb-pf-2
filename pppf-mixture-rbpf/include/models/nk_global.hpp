#pragma once

#include "models/nk_occbin.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace models {

// Global (state-space) NK-ELB solution on a discrete Markov chain for (r,nu).
//
// We discretize the exogenous AR(1) states using Tauchen and solve the stochastic equilibrium
// on that discrete state space as a mixed complementarity problem (ELB kink), using a
// Fischer–Burmeister semi-smooth Newton method.

struct NkGlobalGrid {
  std::vector<double> r_grid;   // deviation, size nr
  std::vector<double> nu_grid;  // deviation, size nn
  Eigen::MatrixXd P;            // joint Markov transition (row-stochastic), size SxS, S=nr*nn

  Eigen::MatrixXd x;   // nr x nn
  Eigen::MatrixXd pi;  // nr x nn
  Eigen::MatrixXd i;   // nr x nn (policy rate levels)
};

inline double normal_cdf(double z) { return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))); }

struct Tauchen1D {
  std::vector<double> grid;
  Eigen::MatrixXd P;
};

inline Tauchen1D tauchen_1d(double rho, double sigma, int n, double m = 3.0) {
  if (n < 2) throw std::invalid_argument("tauchen_1d: n<2");
  if (!(sigma > 0.0)) throw std::invalid_argument("tauchen_1d: sigma<=0");
  if (!(std::abs(rho) < 1.0)) throw std::invalid_argument("tauchen_1d: |rho|>=1");
  if (!(m > 0.0)) throw std::invalid_argument("tauchen_1d: m<=0");

  const double sd = sigma / std::sqrt(1.0 - rho * rho);
  const double zmax = m * sd;
  const double zmin = -zmax;
  const double step = (zmax - zmin) / static_cast<double>(n - 1);

  Tauchen1D out;
  out.grid.resize(n);
  for (int i = 0; i < n; ++i) out.grid[i] = zmin + step * static_cast<double>(i);

  out.P = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    const double mu = rho * out.grid[i];
    for (int j = 0; j < n; ++j) {
      double p = 0.0;
      if (j == 0) {
        p = normal_cdf((out.grid[0] + 0.5 * step - mu) / sigma);
      } else if (j == n - 1) {
        p = 1.0 - normal_cdf((out.grid[n - 1] - 0.5 * step - mu) / sigma);
      } else {
        const double z_hi = (out.grid[j] + 0.5 * step - mu) / sigma;
        const double z_lo = (out.grid[j] - 0.5 * step - mu) / sigma;
        p = normal_cdf(z_hi) - normal_cdf(z_lo);
      }
      out.P(i, j) = std::max(0.0, p);
    }
    out.P.row(i) /= out.P.row(i).sum();
  }
  return out;
}

inline int idx2(int i, int j, int nn) { return i * nn + j; }

inline NkGlobalGrid make_default_global_grid(const NkParams& p) {
  const int nr = 17;
  const int nn = 13;
  const double m = 3.5;

  const Tauchen1D tr = tauchen_1d(p.rho_r, p.sigma_r, nr, m);
  const Tauchen1D tnu = tauchen_1d(p.rho_nu, p.sigma_nu, nn, m);

  NkGlobalGrid g;
  g.r_grid = tr.grid;
  g.nu_grid = tnu.grid;
  g.x = Eigen::MatrixXd::Zero(nr, nn);
  g.pi = Eigen::MatrixXd::Zero(nr, nn);
  g.i = Eigen::MatrixXd::Zero(nr, nn);

  const int S = nr * nn;
  g.P = Eigen::MatrixXd::Zero(S, S);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nn; ++j) {
      const int s = idx2(i, j, nn);
      for (int ip = 0; ip < nr; ++ip) {
        const double pr = tr.P(i, ip);
        for (int jp = 0; jp < nn; ++jp) {
          const double pnu = tnu.P(j, jp);
          const int sp = idx2(ip, jp, nn);
          g.P(s, sp) = pr * pnu;
        }
      }
      g.P.row(s) /= g.P.row(s).sum();
    }
  }
  return g;
}

struct NkGlobalSolveOptions {
  int max_iter = 80;
  double tol = 1e-10;
  double fb_eps = 1e-12;  // regularization inside sqrt
};

inline void solve_nk_global_policy(const NkParams& p, NkGlobalGrid* g,
                                  const NkGlobalSolveOptions& opt = {}) {
  if (!g) throw std::invalid_argument("nk_global: grid null");
  const int nr = static_cast<int>(g->r_grid.size());
  const int nn = static_cast<int>(g->nu_grid.size());
  const int S = nr * nn;
  if (g->P.rows() != S || g->P.cols() != S) throw std::invalid_argument("nk_global: P dim");
  if (!(p.sigma > 0.0)) throw std::invalid_argument("nk_global: sigma<=0");

  const double inv_sigma = 1.0 / p.sigma;

  Eigen::VectorXd r_state(S), nu_state(S);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nn; ++j) {
      const int s = idx2(i, j, nn);
      r_state(s) = g->r_grid[i];
      nu_state(s) = g->nu_grid[j];
    }
  }
  // Unknown vector u = [x; pi; i] across all discrete states.
  const int U = 3 * S;
  Eigen::VectorXd u = Eigen::VectorXd::Zero(U);

  // Initialize with slack-only linear solution, then set i=max(i_lower, rule).
  {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * S, 2 * S);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * S);

    for (int s = 0; s < S; ++s) {
      for (int sp = 0; sp < S; ++sp) {
        const double pssp = g->P(s, sp);
        A(s, sp) += -pssp;
        A(s, S + sp) += -inv_sigma * pssp;
      }
      A(s, s) += 1.0 + inv_sigma * p.phi_x;
      A(s, S + s) += inv_sigma * p.phi_pi;
      const double r_n = p.r_ss + r_state(s);
      b(s) = -inv_sigma * (p.i_ss + nu_state(s) - r_n);
    }

    for (int s = 0; s < S; ++s) {
      const int row = S + s;
      A(row, s) += -p.kappa;
      A(row, S + s) += 1.0;
      for (int sp = 0; sp < S; ++sp) A(row, S + sp) += -p.beta * g->P(s, sp);
      b(row) = 0.0;
    }

    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    if (lu.isInvertible()) {
      const Eigen::VectorXd sol = lu.solve(b);
      if (sol.allFinite()) {
        u.head(S) = sol.head(S);
        u.segment(S, S) = sol.tail(S);
      }
    }

    for (int s = 0; s < S; ++s) {
      const double x = u(s);
      const double pi = u(S + s);
      const double i_rule = p.i_ss + p.phi_pi * pi + p.phi_x * x + nu_state(s);
      u(2 * S + s) = std::max(p.i_lower, i_rule);
    }
  }

  auto eval_F = [&](const Eigen::VectorXd& u_in, Eigen::VectorXd* dphi_da,
                    Eigen::VectorXd* dphi_db) {
    const Eigen::VectorXd x = u_in.head(S);
    const Eigen::VectorXd pi = u_in.segment(S, S);
    const Eigen::VectorXd i_vec = u_in.tail(S);

    const Eigen::VectorXd Ex1 = g->P * x;
    const Eigen::VectorXd Epi1 = g->P * pi;

    Eigen::VectorXd F(U);
    F.head(S) = x - Ex1 + inv_sigma * (i_vec - Epi1 - (p.r_ss + r_state.array()).matrix());
    F.segment(S, S) = pi - p.beta * Epi1 - p.kappa * x;

    if (dphi_da) dphi_da->resize(S);
    if (dphi_db) dphi_db->resize(S);

    for (int s = 0; s < S; ++s) {
      const double a = i_vec(s) - p.i_lower;
      const double rule = p.i_ss + p.phi_pi * pi(s) + p.phi_x * x(s) + nu_state(s);
      const double bcomp = i_vec(s) - rule;
      const double t = std::sqrt(a * a + bcomp * bcomp + opt.fb_eps);
      F(2 * S + s) = t - a - bcomp;
      if (dphi_da) (*dphi_da)(s) = (a / t) - 1.0;
      if (dphi_db) (*dphi_db)(s) = (bcomp / t) - 1.0;
    }
    return F;
  };

  for (int it = 0; it < opt.max_iter; ++it) {
    Eigen::VectorXd dphi_da, dphi_db;
    const Eigen::VectorXd F = eval_F(u, &dphi_da, &dphi_db);
    const double fnorm = F.template lpNorm<Eigen::Infinity>();
    if (!(std::isfinite(fnorm))) break;
    if (fnorm < opt.tol) break;

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(U, U);
    J.topLeftCorner(S, S) = Eigen::MatrixXd::Identity(S, S) - g->P;
    J.block(0, S, S, S) = -inv_sigma * g->P;
    for (int s = 0; s < S; ++s) J(s, 2 * S + s) = inv_sigma;

    for (int s = 0; s < S; ++s) J(S + s, s) = -p.kappa;
    J.block(S, S, S, S) = Eigen::MatrixXd::Identity(S, S) - p.beta * g->P;

    for (int s = 0; s < S; ++s) {
      const double dA = dphi_da(s);
      const double dB = dphi_db(s);
      J(2 * S + s, 2 * S + s) = dA + dB;
      J(2 * S + s, s) = dB * (-p.phi_x);
      J(2 * S + s, S + s) = dB * (-p.phi_pi);
    }

    const Eigen::VectorXd du = J.fullPivLu().solve(-F);
    if (!du.allFinite()) break;

    double step = 1.0;
    double best = fnorm;
    Eigen::VectorXd u_best = u;
    for (int bt = 0; bt < 25; ++bt) {
      const Eigen::VectorXd u_try = u + step * du;
      const Eigen::VectorXd Ft = eval_F(u_try, nullptr, nullptr);
      const double fn = Ft.template lpNorm<Eigen::Infinity>();
      if (std::isfinite(fn) && fn < best) {
        best = fn;
        u_best = u_try;
      }
      if (std::isfinite(fn) && fn <= (1.0 - 1e-4 * step) * fnorm) {
        u = u_try;
        break;
      }
      step *= 0.5;
      if (bt == 24) u = u_best;
    }
    if (!u.allFinite()) break;
  }

  // Unpack to matrices (use last iterate even if not fully converged).
  const Eigen::VectorXd x_out = u.head(S);
  const Eigen::VectorXd pi_out = u.segment(S, S);
  const Eigen::VectorXd i_out = u.tail(S);

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nn; ++j) {
      const int s = idx2(i, j, nn);
      g->x(i, j) = x_out(s);
      g->pi(i, j) = pi_out(s);
      g->i(i, j) = std::max(p.i_lower, i_out(s));  // enforce nonneg for output
    }
  }
}

inline Eigen::VectorXd initial_distribution_bilinear(const NkGlobalGrid& g, double r0, double nu0) {
  const int nr = static_cast<int>(g.r_grid.size());
  const int nn = static_cast<int>(g.nu_grid.size());
  const int S = nr * nn;
  Eigen::VectorXd d = Eigen::VectorXd::Zero(S);

  auto clamp = [](double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); };
  const double r = clamp(r0, g.r_grid.front(), g.r_grid.back());
  const double nu = clamp(nu0, g.nu_grid.front(), g.nu_grid.back());

  auto lower = [](const std::vector<double>& grid, double v) {
    auto it = std::lower_bound(grid.begin(), grid.end(), v);
    if (it == grid.begin()) return 0;
    if (it == grid.end()) return static_cast<int>(grid.size()) - 2;
    return static_cast<int>(std::distance(grid.begin(), it)) - 1;
  };

  const int i = lower(g.r_grid, r);
  const int j = lower(g.nu_grid, nu);
  const double r0g = g.r_grid[i];
  const double r1g = g.r_grid[i + 1];
  const double nu0g = g.nu_grid[j];
  const double nu1g = g.nu_grid[j + 1];
  const double tr = (r1g == r0g) ? 0.0 : (r - r0g) / (r1g - r0g);
  const double tnu = (nu1g == nu0g) ? 0.0 : (nu - nu0g) / (nu1g - nu0g);

  const double w00 = (1.0 - tr) * (1.0 - tnu);
  const double w10 = tr * (1.0 - tnu);
  const double w01 = (1.0 - tr) * tnu;
  const double w11 = tr * tnu;

  d(idx2(i, j, nn)) += w00;
  d(idx2(i + 1, j, nn)) += w10;
  d(idx2(i, j + 1, nn)) += w01;
  d(idx2(i + 1, j + 1, nn)) += w11;
  d /= d.sum();
  return d;
}

inline void expected_irf_global(const NkParams& p, const NkGlobalGrid& g, double shock_eps_r, int T,
                                std::vector<double>* x_out, std::vector<double>* pi_out,
                                std::vector<double>* i_out) {
  if (!x_out || !pi_out || !i_out) throw std::invalid_argument("expected_irf_global: null out");
  const int nr = static_cast<int>(g.r_grid.size());
  const int nn = static_cast<int>(g.nu_grid.size());
  const int S = nr * nn;
  if (g.P.rows() != S || g.P.cols() != S) throw std::invalid_argument("expected_irf_global: P dim");

  Eigen::VectorXd d = initial_distribution_bilinear(g, p.sigma_r * shock_eps_r, 0.0);

  x_out->assign(T, 0.0);
  pi_out->assign(T, 0.0);
  i_out->assign(T, 0.0);

  Eigen::VectorXd x_state(S), pi_state(S), i_state(S);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nn; ++j) {
      const int s = idx2(i, j, nn);
      x_state(s) = g.x(i, j);
      pi_state(s) = g.pi(i, j);
      i_state(s) = g.i(i, j);
    }
  }

  for (int t = 0; t < T; ++t) {
    (*x_out)[t] = d.dot(x_state);
    (*pi_out)[t] = d.dot(pi_state);
    (*i_out)[t] = d.dot(i_state);
    d = g.P.transpose() * d;
    d /= d.sum();
  }
}

}  // namespace models
