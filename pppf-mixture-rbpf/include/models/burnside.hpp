#pragma once

#include "quadrature/sigma_points.hpp"

#include <cmath>
#include <stdexcept>

namespace models {

struct BurnsideParams {
  double beta = 0.99;
  double gamma = 2.0;
  double rho = 0.95;
  double mu = 0.0;
  double sigma = 0.01;
};

// Exact v(x) via the infinite series:
//   v(x) = Σ_{j>=1} β^j E[ exp((1-γ) Σ_{s=1}^j x_{t+s}) | x_t=x ]
// For Gaussian AR(1), the conditional sum is Gaussian, so the expectation is analytic.
inline double burnside_v_exact(const BurnsideParams& p, double x, double tol = 1e-12,
                               int max_terms = 100000) {
  const double a = 1.0 - p.gamma;
  const double one_m_rho = 1.0 - p.rho;
  if (std::abs(one_m_rho) < 1e-14) throw std::invalid_argument("burnside_v_exact: rho too close to 1");

  double v = 0.0;
  double beta_pow = p.beta;
  double sum_rho_pows = 0.0;  // Σ_{s=1}^j ρ^s
  double var_sum = 0.0;       // Var(Σ x_{t+s}) conditional on x

  for (int j = 1; j <= max_terms; ++j) {
    sum_rho_pows += std::pow(p.rho, j);
    const double mean_sum = static_cast<double>(j) * p.mu + (x - p.mu) * sum_rho_pows;

    // Update variance term incrementally:
    // Var(S_j) = σ^2 Σ_{m=1}^j ((1-ρ^m)/(1-ρ))^2
    const double c = (1.0 - std::pow(p.rho, j)) / one_m_rho;
    var_sum += p.sigma * p.sigma * c * c;

    const double term = beta_pow * std::exp(a * mean_sum + 0.5 * a * a * var_sum);
    v += term;

    if (std::abs(term) < tol * (1.0 + std::abs(v))) break;
    beta_pow *= p.beta;
    if (!(beta_pow > 0.0)) break;
  }
  return v;
}

// Certainty-equivalence: drop the variance term (set future shocks to zero in expectation).
inline double burnside_v_ce(const BurnsideParams& p, double x, double tol = 1e-12,
                            int max_terms = 100000) {
  const double a = 1.0 - p.gamma;
  double v = 0.0;
  double beta_pow = p.beta;
  double sum_rho_pows = 0.0;

  for (int j = 1; j <= max_terms; ++j) {
    sum_rho_pows += std::pow(p.rho, j);
    const double mean_sum = static_cast<double>(j) * p.mu + (x - p.mu) * sum_rho_pows;
    const double term = beta_pow * std::exp(a * mean_sum);
    v += term;
    if (std::abs(term) < tol * (1.0 + std::abs(v))) break;
    beta_pow *= p.beta;
    if (!(beta_pow > 0.0)) break;
  }
  return v;
}

// One-step UT quadrature over ε_{t+1} with continuation approximated by CE and no further shocks:
//   v_ut(x) = β E[ exp(a x_{t+1}) (1 + v_ce(x_{t+1})) | x_t=x ]
inline double burnside_v_ut_one_step(const BurnsideParams& p, double x, double alpha = 1e-2,
                                     double kappa = 0.0, double beta_ut = 2.0) {
  const double a = 1.0 - p.gamma;
  const double x_mean = (1.0 - p.rho) * p.mu + p.rho * x;
  const auto sp = quadrature::unscented_sigma_points_1d(0.0, p.sigma * p.sigma, alpha, kappa, beta_ut);

  double acc = 0.0;
  for (int k = 0; k < sp.points.rows(); ++k) {
    const double eps = sp.points(k, 0);
    const double x1 = x_mean + eps;
    const double cont = 1.0 + burnside_v_ce(p, x1);
    acc += sp.w_mean(k) * std::exp(a * x1) * cont;
  }
  return p.beta * acc;
}

}  // namespace models

