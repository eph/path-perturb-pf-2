#include "filters/resampling.hpp"
#include "util/io.hpp"
#include "util/rng.hpp"
#include "util/stats.hpp"
#include "util/timer.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RbcParams {
  double alpha = 0.36;
  double beta = 0.96;
  double delta = 0.08;
  double rho = 0.95;
  double sigma = 0.02;
};

struct CliOptions {
  int T = 150;
  int N = 512;
  int R = 20;
  int Nk = 121;
  int Nz = 9;
  double k_width = 0.45;
  double z_width_sd = 3.0;
  double meas_sd = 0.02;
  std::uint64_t seed_data = 20260307ULL;
  std::filesystem::path out_dir = "output/rbc";
};

struct TauchenDiscretization {
  std::vector<double> z_grid;
  std::vector<double> z_edges;
  Eigen::MatrixXd P;
  Eigen::VectorXd stationary;
};

struct RbcSolution {
  RbcParams p;
  std::vector<double> k_grid;
  TauchenDiscretization z_disc;
  Eigen::MatrixXd policy_kp;  // Nk x Nz
};

struct RbcObs {
  Eigen::Vector3d mean;
};

struct RbcState {
  double k = 0.0;
  double z = 0.0;
};

struct ProposalMixture {
  std::vector<double> means;
  std::vector<double> probs;
  double comp_sd = 1e-3;
};

struct PfDiagnostics {
  double loglik = 0.0;
  std::vector<double> ess;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (a == "--T") {
      opt.T = std::stoi(need(a));
    } else if (a == "--N") {
      opt.N = std::stoi(need(a));
    } else if (a == "--R") {
      opt.R = std::stoi(need(a));
    } else if (a == "--Nk") {
      opt.Nk = std::stoi(need(a));
    } else if (a == "--Nz") {
      opt.Nz = std::stoi(need(a));
    } else if (a == "--k_width") {
      opt.k_width = std::stod(need(a));
    } else if (a == "--z_width_sd") {
      opt.z_width_sd = std::stod(need(a));
    } else if (a == "--meas_sd") {
      opt.meas_sd = std::stod(need(a));
    } else if (a == "--seed_data") {
      opt.seed_data = static_cast<std::uint64_t>(std::stoull(need(a)));
    } else if (a == "--out_dir") {
      opt.out_dir = need(a);
    } else if (a == "-h" || a == "--help") {
      std::cout << "Usage: rbc_compare [--T int] [--N int] [--R int] [--Nk int] [--Nz int] "
                   "[--k_width double] [--z_width_sd double] [--meas_sd double] "
                   "[--seed_data uint64] [--out_dir path]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown flag: " + a);
    }
  }
  if (opt.T <= 0 || opt.N <= 0 || opt.R <= 0) throw std::invalid_argument("T,N,R must be positive");
  if (opt.Nk < 31 || opt.Nk % 2 == 0) throw std::invalid_argument("Nk must be odd and >= 31");
  if (opt.Nz < 5) throw std::invalid_argument("Nz must be >= 5");
  if (!(opt.k_width > 0.0) || !(opt.z_width_sd > 0.0) || !(opt.meas_sd > 0.0)) {
    throw std::invalid_argument("widths and meas_sd must be positive");
  }
  return opt;
}

double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double normal_pdf(double x, double mean, double sd) {
  const double z = (x - mean) / sd;
  return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * M_PI));
}

double log_norm_pdf_scalar(double y, double mean, double sd) {
  const double z = (y - mean) / sd;
  return -0.5 * std::log(2.0 * M_PI) - std::log(sd) - 0.5 * z * z;
}

double normal_bin_prob(double mean, double sd, double lo, double hi) {
  if (std::isinf(lo) && lo < 0.0) return normal_cdf((hi - mean) / sd);
  if (std::isinf(hi) && hi > 0.0) return 1.0 - normal_cdf((lo - mean) / sd);
  return normal_cdf((hi - mean) / sd) - normal_cdf((lo - mean) / sd);
}

double rbc_steady_state_k(const RbcParams& p) {
  const double num = (1.0 / p.beta) - 1.0 + p.delta;
  return std::pow(num / p.alpha, 1.0 / (p.alpha - 1.0));
}

TauchenDiscretization tauchen_ar1(double rho, double sigma, int Nz, double width_sd) {
  const double z_sd = sigma / std::sqrt(1.0 - rho * rho);
  const double lo = -width_sd * z_sd;
  const double hi = +width_sd * z_sd;

  TauchenDiscretization out;
  out.z_grid.resize(Nz);
  for (int i = 0; i < Nz; ++i) {
    const double frac = static_cast<double>(i) / static_cast<double>(Nz - 1);
    out.z_grid[i] = lo + frac * (hi - lo);
  }
  out.z_edges.assign(Nz + 1, 0.0);
  out.z_edges.front() = -std::numeric_limits<double>::infinity();
  out.z_edges.back() = std::numeric_limits<double>::infinity();
  for (int i = 1; i < Nz; ++i) out.z_edges[i] = 0.5 * (out.z_grid[i - 1] + out.z_grid[i]);

  out.P = Eigen::MatrixXd::Zero(Nz, Nz);
  for (int i = 0; i < Nz; ++i) {
    const double mean = rho * out.z_grid[i];
    for (int j = 0; j < Nz; ++j) {
      out.P(i, j) = normal_bin_prob(mean, sigma, out.z_edges[j], out.z_edges[j + 1]);
    }
    out.P.row(i) /= out.P.row(i).sum();
  }

  Eigen::VectorXd w = Eigen::VectorXd::Constant(Nz, 1.0 / static_cast<double>(Nz));
  for (int it = 0; it < 10000; ++it) {
    const Eigen::VectorXd w_new = out.P.transpose() * w;
    if ((w_new - w).cwiseAbs().maxCoeff() < 1e-14) {
      w = w_new;
      break;
    }
    w = w_new;
  }
  out.stationary = w / w.sum();
  return out;
}

std::vector<double> make_k_grid(double k_ss, int Nk, double width) {
  std::vector<double> grid(Nk);
  const double lo = std::max(1e-4, k_ss * (1.0 - width));
  const double hi = k_ss * (1.0 + width);
  for (int i = 0; i < Nk; ++i) {
    const double frac = static_cast<double>(i) / static_cast<double>(Nk - 1);
    grid[i] = lo + frac * (hi - lo);
  }
  return grid;
}

int lower_bracket(const std::vector<double>& grid, double x) {
  if (x <= grid.front()) return 0;
  if (x >= grid.back()) return static_cast<int>(grid.size()) - 2;
  const auto it = std::upper_bound(grid.begin(), grid.end(), x);
  return static_cast<int>(it - grid.begin()) - 1;
}

double interp1(const std::vector<double>& grid, const Eigen::VectorXd& vals, double x) {
  if (x <= grid.front()) return vals(0);
  if (x >= grid.back()) return vals(vals.size() - 1);
  const int lo = lower_bracket(grid, x);
  const int hi = lo + 1;
  const double w = (x - grid[lo]) / (grid[hi] - grid[lo]);
  return (1.0 - w) * vals(lo) + w * vals(hi);
}

double interp_policy(const RbcSolution& sol, double k, double z) {
  const auto& zg = sol.z_disc.z_grid;
  const int Nz = static_cast<int>(zg.size());
  if (z <= zg.front()) {
    return interp1(sol.k_grid, sol.policy_kp.col(0), k);
  }
  if (z >= zg.back()) {
    return interp1(sol.k_grid, sol.policy_kp.col(Nz - 1), k);
  }
  const auto it = std::upper_bound(zg.begin(), zg.end(), z);
  const int j0 = static_cast<int>(it - zg.begin()) - 1;
  const int j1 = j0 + 1;
  const double wz = (z - zg[j0]) / (zg[j1] - zg[j0]);
  const double kp0 = interp1(sol.k_grid, sol.policy_kp.col(j0), k);
  const double kp1 = interp1(sol.k_grid, sol.policy_kp.col(j1), k);
  return (1.0 - wz) * kp0 + wz * kp1;
}

RbcObs rbc_observables(const RbcParams& p, const RbcSolution& sol, double k, double z) {
  const double y = std::exp(z) * std::pow(k, p.alpha);
  const double kp = interp_policy(sol, k, z);
  const double invest = kp - (1.0 - p.delta) * k;
  const double c = y - invest;
  if (!(c > 0.0) || !(invest > 0.0) || !(y > 0.0)) throw std::runtime_error("rbc_observables: nonpositive");
  RbcObs out;
  out.mean << std::log(y), std::log(c), std::log(invest);
  return out;
}

RbcSolution solve_rbc_policy(const RbcParams& p, int Nk, int Nz, double k_width, double z_width_sd) {
  RbcSolution sol;
  sol.p = p;
  const double k_ss = rbc_steady_state_k(p);
  sol.k_grid = make_k_grid(k_ss, Nk, k_width);
  sol.z_disc = tauchen_ar1(p.rho, p.sigma, Nz, z_width_sd);
  sol.policy_kp = Eigen::MatrixXd::Constant(Nk, Nz, k_ss);

  auto euler_residual = [&](double kp, double k, int iz, const Eigen::MatrixXd& policy) {
    const double z = sol.z_disc.z_grid[iz];
    const double output = std::exp(z) * std::pow(k, p.alpha);
    const double invest = kp - (1.0 - p.delta) * k;
    const double c = output - invest;
    if (!(c > 1e-12)) return std::numeric_limits<double>::infinity();
    double exp_term = 0.0;
    for (int jz = 0; jz < Nz; ++jz) {
      const double zp = sol.z_disc.z_grid[jz];
      const double kp2 = interp1(sol.k_grid, policy.col(jz), kp);
      const double yp = std::exp(zp) * std::pow(kp, p.alpha);
      const double invest_p = kp2 - (1.0 - p.delta) * kp;
      const double cp = yp - invest_p;
      if (!(cp > 1e-12)) return -std::numeric_limits<double>::infinity();
      const double mpk = p.alpha * std::exp(zp) * std::pow(kp, p.alpha - 1.0) + (1.0 - p.delta);
      exp_term += sol.z_disc.P(iz, jz) * mpk / cp;
    }
    return (1.0 / c) - p.beta * exp_term;
  };

  for (int it = 0; it < 1000; ++it) {
    Eigen::MatrixXd updated = sol.policy_kp;
    double maxdiff = 0.0;
    for (int iz = 0; iz < Nz; ++iz) {
      const double z = sol.z_disc.z_grid[iz];
      for (int ik = 0; ik < Nk; ++ik) {
        const double k = sol.k_grid[ik];
        const double resource = std::exp(z) * std::pow(k, p.alpha) + (1.0 - p.delta) * k;
        const double k_lo = std::max(sol.k_grid.front(), 1e-6);
        const double k_hi = std::min(sol.k_grid.back(), resource - 1e-8);
        if (!(k_hi > k_lo)) throw std::runtime_error("solve_rbc_policy: infeasible state");

        double a = k_lo;
        double b = k_hi;
        double fa = euler_residual(a, k, iz, sol.policy_kp);
        double fb = euler_residual(b, k, iz, sol.policy_kp);
        if (!std::isfinite(fa)) fa = 1e12;
        if (!std::isfinite(fb)) fb = -1e12;

        if (fa * fb > 0.0) {
          double best_kp = std::min(std::max(sol.policy_kp(ik, iz), a), b);
          double best_val = std::abs(euler_residual(best_kp, k, iz, sol.policy_kp));
          for (int m = 0; m < 60; ++m) {
            const double km = a + (b - a) * static_cast<double>(m) / 59.0;
            const double fm = std::abs(euler_residual(km, k, iz, sol.policy_kp));
            if (fm < best_val) {
              best_val = fm;
              best_kp = km;
            }
          }
          updated(ik, iz) = best_kp;
        } else {
          for (int bis = 0; bis < 80; ++bis) {
            const double m = 0.5 * (a + b);
            const double fm = euler_residual(m, k, iz, sol.policy_kp);
            if (!std::isfinite(fm)) {
              b = m;
              continue;
            }
            if (fa * fm <= 0.0) {
              b = m;
              fb = fm;
            } else {
              a = m;
              fa = fm;
            }
          }
          updated(ik, iz) = 0.5 * (a + b);
        }
        maxdiff = std::max(maxdiff, std::abs(updated(ik, iz) - sol.policy_kp(ik, iz)));
      }
    }
    sol.policy_kp = updated;
    if (maxdiff < 1e-8) break;
    if (it == 999) throw std::runtime_error("solve_rbc_policy: no convergence");
  }

  return sol;
}

Eigen::VectorXd stationary_state_weights(const RbcSolution& sol) {
  const int Nk = static_cast<int>(sol.k_grid.size());
  const int Nz = static_cast<int>(sol.z_disc.z_grid.size());
  Eigen::VectorXd w = Eigen::VectorXd::Zero(Nk * Nz);
  const double k_ss = rbc_steady_state_k(sol.p);
  const int lo = lower_bracket(sol.k_grid, k_ss);
  const int hi = std::min(lo + 1, Nk - 1);
  const double wk_hi = (k_ss - sol.k_grid[lo]) / (sol.k_grid[hi] - sol.k_grid[lo] + 1e-16);
  const double wk_lo = 1.0 - wk_hi;
  for (int iz = 0; iz < Nz; ++iz) {
    w(lo * Nz + iz) += wk_lo * sol.z_disc.stationary(iz);
    w(hi * Nz + iz) += wk_hi * sol.z_disc.stationary(iz);
  }
  return w / w.sum();
}

Eigen::MatrixXd build_discrete_transition(const RbcSolution& sol) {
  const int Nk = static_cast<int>(sol.k_grid.size());
  const int Nz = static_cast<int>(sol.z_disc.z_grid.size());
  const int S = Nk * Nz;
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(S, S);
  for (int ik = 0; ik < Nk; ++ik) {
    for (int iz = 0; iz < Nz; ++iz) {
      const int row = ik * Nz + iz;
      const double kp = sol.policy_kp(ik, iz);
      const int klo = lower_bracket(sol.k_grid, kp);
      const int khi = std::min(klo + 1, Nk - 1);
      const double wk_hi = (kp - sol.k_grid[klo]) / (sol.k_grid[khi] - sol.k_grid[klo] + 1e-16);
      const double wk_lo = 1.0 - wk_hi;
      for (int jz = 0; jz < Nz; ++jz) {
        const double pz = sol.z_disc.P(iz, jz);
        P(row, klo * Nz + jz) += wk_lo * pz;
        P(row, khi * Nz + jz) += wk_hi * pz;
      }
      P.row(row) /= P.row(row).sum();
    }
  }
  return P;
}

std::vector<Eigen::VectorXd> simulate_data(const RbcSolution& sol, int T, double meas_sd, std::uint64_t seed_data,
                                           std::vector<RbcState>* states_out) {
  util::Rng rng(seed_data);
  const double k_ss = rbc_steady_state_k(sol.p);
  double k = k_ss;
  double z = (sol.p.sigma / std::sqrt(1.0 - sol.p.rho * sol.p.rho)) * rng.normal();
  for (int b = 0; b < 250; ++b) {
    const double kp = interp_policy(sol, k, z);
    z = sol.p.rho * z + sol.p.sigma * rng.normal();
    k = kp;
  }

  std::vector<Eigen::VectorXd> y(T);
  states_out->resize(T);
  for (int t = 0; t < T; ++t) {
    const RbcObs obs = rbc_observables(sol.p, sol, k, z);
    Eigen::Vector3d yt = obs.mean;
    for (int j = 0; j < 3; ++j) yt(j) += meas_sd * rng.normal();
    y[t] = yt;
    (*states_out)[t] = RbcState{k, z};

    const double kp = interp_policy(sol, k, z);
    z = sol.p.rho * z + sol.p.sigma * rng.normal();
    k = kp;
  }
  return y;
}

double grid_loglik(const RbcSolution& sol, const Eigen::MatrixXd& P, const Eigen::VectorXd& alpha0,
                   const std::vector<Eigen::VectorXd>& y, double meas_sd) {
  const int Nk = static_cast<int>(sol.k_grid.size());
  const int Nz = static_cast<int>(sol.z_disc.z_grid.size());
  const int S = Nk * Nz;
  Eigen::VectorXd alpha = alpha0;
  double ll = 0.0;
  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd pred = P.transpose() * alpha;
    Eigen::VectorXd obs_lik(S);
    for (int ik = 0; ik < Nk; ++ik) {
      for (int iz = 0; iz < Nz; ++iz) {
        const int s = ik * Nz + iz;
        const RbcObs obs = rbc_observables(sol.p, sol, sol.k_grid[ik], sol.z_disc.z_grid[iz]);
        obs_lik(s) = std::exp(util::log_mvnorm_pdf(y[t], obs.mean, meas_sd * meas_sd * Eigen::Matrix3d::Identity()));
      }
    }
    const double zt = pred.dot(obs_lik);
    ll += std::log(zt);
    alpha = pred.cwiseProduct(obs_lik) / zt;
  }
  return ll;
}

double log_mixture_normal_pdf(double x, const ProposalMixture& proposal) {
  Eigen::VectorXd log_terms(static_cast<int>(proposal.means.size()));
  for (int i = 0; i < static_cast<int>(proposal.means.size()); ++i) {
    log_terms(i) = std::log(proposal.probs[i]) + log_norm_pdf_scalar(x, proposal.means[i], proposal.comp_sd);
  }
  return util::log_sum_exp(log_terms);
}

ProposalMixture build_proposal_exact(const RbcSolution& sol, double k_t, double prior_mean_z, double y_t_sd,
                                     const Eigen::VectorXd& y_t, int nodes = 31, double width_sd = 5.5,
                                     double jitter_frac = 0.6) {
  Eigen::VectorXd log_terms(nodes);
  std::vector<double> means(nodes, 0.0);
  const double dz = (2.0 * width_sd) / static_cast<double>(nodes - 1);
  for (int i = 0; i < nodes; ++i) {
    const double zi = prior_mean_z + sol.p.sigma * (-width_sd + dz * static_cast<double>(i));
    const double trap_w = (i == 0 || i == nodes - 1) ? 0.5 * dz : dz;
    means[i] = zi;
    const RbcObs obs = rbc_observables(sol.p, sol, k_t, zi);
    log_terms(i) = std::log(trap_w) + std::log(normal_pdf(zi, prior_mean_z, sol.p.sigma)) +
                   util::log_mvnorm_pdf(y_t, obs.mean, y_t_sd * y_t_sd * Eigen::Matrix3d::Identity());
  }
  const double lse = util::log_sum_exp(log_terms);
  ProposalMixture out;
  out.means = means;
  out.probs.resize(nodes);
  for (int i = 0; i < nodes; ++i) out.probs[i] = std::exp(log_terms(i) - lse);
  out.comp_sd = std::max(1e-6, jitter_frac * sol.p.sigma * dz);
  return out;
}

ProposalMixture build_proposal_ut(const RbcSolution& sol, double k_t, double prior_mean_z, double y_t_sd,
                                  const Eigen::VectorXd& y_t) {
  // Five-node positive-weight Gaussian quadrature for z_t | z_{t-1}.
  // Nodes/weights correspond to the N(0,1) measure.
  const std::vector<double> x_std = {
      -2.8569700138728056, -1.3556261799742659, 0.0, 1.3556261799742659, 2.8569700138728056};
  const std::vector<double> w0 = {
      0.011257411327720689, 0.22207592200561266, 0.5333333333333333, 0.22207592200561266,
      0.011257411327720689};
  const int K = static_cast<int>(x_std.size());
  std::vector<double> nodes(K, 0.0);
  for (int i = 0; i < K; ++i) nodes[i] = prior_mean_z + sol.p.sigma * x_std[i];

  Eigen::VectorXd log_terms(K);
  for (int i = 0; i < K; ++i) {
    const RbcObs obs = rbc_observables(sol.p, sol, k_t, nodes[i]);
    log_terms(i) = std::log(w0[i]) +
                   util::log_mvnorm_pdf(y_t, obs.mean, y_t_sd * y_t_sd * Eigen::Matrix3d::Identity());
  }
  const double lse = util::log_sum_exp(log_terms);
  ProposalMixture out;
  out.means = nodes;
  out.probs.resize(K);
  for (int i = 0; i < K; ++i) out.probs[i] = std::exp(log_terms(i) - lse);
  out.comp_sd = std::max(1e-6, 0.20 * sol.p.sigma);
  return out;
}

ProposalMixture build_proposal_ce(const RbcSolution& sol, double prior_mean_z) {
  ProposalMixture out;
  out.means = {prior_mean_z};
  out.probs = {1.0};
  out.comp_sd = std::max(1e-6, 0.15 * sol.p.sigma);
  return out;
}

PfDiagnostics bootstrap_pf(const RbcSolution& sol, int N, const std::vector<Eigen::VectorXd>& y, double meas_sd,
                           std::uint64_t seed, double ess_frac = 0.5) {
  util::Rng rng(seed);
  const double k_ss = rbc_steady_state_k(sol.p);
  const double z_sd = sol.p.sigma / std::sqrt(1.0 - sol.p.rho * sol.p.rho);
  std::vector<RbcState> particles(N, RbcState{k_ss, 0.0});
  for (int i = 0; i < N; ++i) particles[i].z = z_sd * rng.normal();
  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
  PfDiagnostics diag;
  diag.ess.reserve(y.size());

  const Eigen::Matrix3d R = meas_sd * meas_sd * Eigen::Matrix3d::Identity();
  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd logw_new(N);
    for (int i = 0; i < N; ++i) {
      particles[i].k = interp_policy(sol, particles[i].k, particles[i].z);
      particles[i].z = sol.p.rho * particles[i].z + sol.p.sigma * rng.normal();
      const RbcObs obs = rbc_observables(sol.p, sol, particles[i].k, particles[i].z);
      logw_new(i) = logw(i) + util::log_mvnorm_pdf(y[t], obs.mean, R);
    }
    const double logZ = util::log_sum_exp(logw_new);
    diag.loglik += logZ;
    logw_new.array() -= logZ;
    const double ess_t = util::ess_from_logw(logw_new);
    diag.ess.push_back(ess_t);
    if (ess_t < ess_frac * static_cast<double>(N)) {
      const Eigen::VectorXd w_norm = logw_new.array().exp().matrix();
      const std::vector<int> idx = filters::systematic_resample(w_norm, rng);
      std::vector<RbcState> repl(N);
      for (int i = 0; i < N; ++i) repl[i] = particles[idx[i]];
      particles.swap(repl);
      logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
    } else {
      logw = logw_new;
    }
  }
  return diag;
}

enum class ProposalMode { Exact, UT, CE };

PfDiagnostics copf_pf(const RbcSolution& sol, ProposalMode mode, int N, const std::vector<Eigen::VectorXd>& y,
                      double meas_sd, std::uint64_t seed, double ess_frac = 0.5) {
  util::Rng rng(seed);
  const double k_ss = rbc_steady_state_k(sol.p);
  const double z_sd = sol.p.sigma / std::sqrt(1.0 - sol.p.rho * sol.p.rho);
  std::vector<RbcState> particles(N, RbcState{k_ss, 0.0});
  for (int i = 0; i < N; ++i) particles[i].z = z_sd * rng.normal();
  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
  PfDiagnostics diag;
  diag.ess.reserve(y.size());
  const Eigen::Matrix3d R = meas_sd * meas_sd * Eigen::Matrix3d::Identity();

  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd logw_new(N);
    std::vector<RbcState> new_particles(N);
    for (int i = 0; i < N; ++i) {
      const double k_t = interp_policy(sol, particles[i].k, particles[i].z);
      const double prior_mean_z = sol.p.rho * particles[i].z;
      ProposalMixture proposal;
      if (mode == ProposalMode::Exact) {
        proposal = build_proposal_exact(sol, k_t, prior_mean_z, meas_sd, y[t]);
      } else if (mode == ProposalMode::UT) {
        proposal = build_proposal_ut(sol, k_t, prior_mean_z, meas_sd, y[t]);
      } else {
        proposal = build_proposal_ce(sol, prior_mean_z);
      }
      const int draw_idx = rng.categorical(proposal.probs);
      const double z_t = proposal.means[draw_idx] + proposal.comp_sd * rng.normal();
      const RbcObs obs = rbc_observables(sol.p, sol, k_t, z_t);
      const double log_obs = util::log_mvnorm_pdf(y[t], obs.mean, R);
      const double log_prior = log_norm_pdf_scalar(z_t, prior_mean_z, sol.p.sigma);
      const double log_prop = log_mixture_normal_pdf(z_t, proposal);
      logw_new(i) = logw(i) + log_obs + log_prior - log_prop;
      new_particles[i] = RbcState{k_t, z_t};
    }
    const double logZ = util::log_sum_exp(logw_new);
    diag.loglik += logZ;
    logw_new.array() -= logZ;
    const double ess_t = util::ess_from_logw(logw_new);
    diag.ess.push_back(ess_t);
    particles.swap(new_particles);
    if (ess_t < ess_frac * static_cast<double>(N)) {
      const Eigen::VectorXd w_norm = logw_new.array().exp().matrix();
      const std::vector<int> idx = filters::systematic_resample(w_norm, rng);
      std::vector<RbcState> repl(N);
      for (int i = 0; i < N; ++i) repl[i] = particles[idx[i]];
      particles.swap(repl);
      logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
    } else {
      logw = logw_new;
    }
  }
  return diag;
}

void write_summary_json(const std::filesystem::path& path,
                        const std::map<std::string, std::vector<double>>& loglik_by_method,
                        const std::map<std::string, std::vector<double>>& rt_by_method,
                        const std::map<std::string, double>& mean_ess_by_method, const CliOptions& cli) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"config\": {\n";
  oss << "    \"T\": " << cli.T << ",\n";
  oss << "    \"N\": " << cli.N << ",\n";
  oss << "    \"R\": " << cli.R << ",\n";
  oss << "    \"Nk\": " << cli.Nk << ",\n";
  oss << "    \"Nz\": " << cli.Nz << ",\n";
  oss << "    \"meas_sd\": " << cli.meas_sd << "\n";
  oss << "  },\n";
  oss << "  \"methods\": {\n";
  bool first = true;
  for (const auto& [name, vals] : loglik_by_method) {
    if (!first) oss << ",\n";
    first = false;
    oss << "    \"" << util::json_escape(name) << "\": {\n";
    oss << "      \"mean_loglik\": " << util::mean(vals) << ",\n";
    oss << "      \"sd_loglik\": " << util::stdev(vals) << ",\n";
    oss << "      \"mean_runtime_ms\": " << util::mean(rt_by_method.at(name)) << ",\n";
    oss << "      \"sd_runtime_ms\": " << util::stdev(rt_by_method.at(name)) << ",\n";
    oss << "      \"mean_ess\": " << mean_ess_by_method.at(name) << "\n";
    oss << "    }";
  }
  oss << "\n  }\n";
  oss << "}\n";
  util::write_text_file(path, oss.str());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    const RbcParams p;
    util::ensure_dir(cli.out_dir);

    util::Timer solve_timer;
    const RbcSolution sol = solve_rbc_policy(p, cli.Nk, cli.Nz, cli.k_width, cli.z_width_sd);
    const double solve_ms = solve_timer.elapsed_ms();

    std::vector<RbcState> states;
    const std::vector<Eigen::VectorXd> y = simulate_data(sol, cli.T, cli.meas_sd, cli.seed_data, &states);

    std::ofstream sim_out(cli.out_dir / "sim_data.csv");
    if (!sim_out) throw std::runtime_error("failed to open sim_data.csv");
    util::write_csv_header(sim_out, {"t", "k", "z", "log_y", "log_c", "log_i"});
    for (int t = 0; t < cli.T; ++t) {
      util::write_csv_row(sim_out, t, states[t].k, states[t].z, y[t](0), y[t](1), y[t](2));
    }

    const Eigen::MatrixXd P = build_discrete_transition(sol);
    const Eigen::VectorXd alpha0 = stationary_state_weights(sol);

    std::map<std::string, std::vector<double>> loglik_by_method;
    std::map<std::string, std::vector<double>> rt_by_method;
    std::map<std::string, double> mean_ess_by_method;

    std::ofstream ll_out(cli.out_dir / "loglik_repeats.csv");
    if (!ll_out) throw std::runtime_error("failed to open loglik_repeats.csv");
    util::write_csv_header(ll_out, {"method", "rep", "loglik", "runtime_ms"});
    std::ofstream ess_out(cli.out_dir / "ess.csv");
    if (!ess_out) throw std::runtime_error("failed to open ess.csv");
    util::write_csv_header(ess_out, {"method", "rep", "t", "ess"});

    {
      util::Timer timer;
      const double ll = grid_loglik(sol, P, alpha0, y, cli.meas_sd);
      const double rt = timer.elapsed_ms();
      loglik_by_method["grid_exact"].push_back(ll);
      rt_by_method["grid_exact"].push_back(rt);
      mean_ess_by_method["grid_exact"] = -1.0;
      util::write_csv_row(ll_out, "grid_exact", 0, ll, rt);
    }

    const std::vector<std::pair<std::string, ProposalMode>> methods = {
        {"copf_exact", ProposalMode::Exact},
        {"copf_ut", ProposalMode::UT},
        {"copf_ce", ProposalMode::CE},
    };

    {
      std::vector<double> ess_means;
      for (int rep = 0; rep < cli.R; ++rep) {
        util::Timer timer;
        const PfDiagnostics diag =
            bootstrap_pf(sol, cli.N, y, cli.meas_sd, 880000ULL + static_cast<std::uint64_t>(rep) * 17ULL);
        const double rt = timer.elapsed_ms();
        loglik_by_method["bootstrap_pf"].push_back(diag.loglik);
        rt_by_method["bootstrap_pf"].push_back(rt);
        util::write_csv_row(ll_out, "bootstrap_pf", rep, diag.loglik, rt);
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(diag.ess.size()); ++t) {
          util::write_csv_row(ess_out, "bootstrap_pf", rep, t, diag.ess[t]);
          ess_mean += diag.ess[t];
        }
        ess_mean /= static_cast<double>(diag.ess.size());
        ess_means.push_back(ess_mean);
      }
      mean_ess_by_method["bootstrap_pf"] = util::mean(ess_means);
    }

    for (const auto& [name, mode] : methods) {
      std::vector<double> ess_means;
      for (int rep = 0; rep < cli.R; ++rep) {
        util::Timer timer;
        const PfDiagnostics diag =
            copf_pf(sol, mode, cli.N, y, cli.meas_sd, 990000ULL + static_cast<std::uint64_t>(rep) * 31ULL);
        const double rt = timer.elapsed_ms();
        loglik_by_method[name].push_back(diag.loglik);
        rt_by_method[name].push_back(rt);
        util::write_csv_row(ll_out, name, rep, diag.loglik, rt);
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(diag.ess.size()); ++t) {
          util::write_csv_row(ess_out, name, rep, t, diag.ess[t]);
          ess_mean += diag.ess[t];
        }
        ess_mean /= static_cast<double>(diag.ess.size());
        ess_means.push_back(ess_mean);
      }
      mean_ess_by_method[name] = util::mean(ess_means);
    }

    write_summary_json(cli.out_dir / "summary.json", loglik_by_method, rt_by_method, mean_ess_by_method, cli);

    std::cout << "Solved RBC policy in " << solve_ms << " ms\n";
    std::cout << "Wrote " << (cli.out_dir / "sim_data.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "loglik_repeats.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "ess.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "summary.json").string() << "\n";
    const double ll_exact = util::mean(loglik_by_method.at("grid_exact"));
    std::cout << "Mean loglik gaps vs grid_exact:\n";
    for (const auto& [name, vals] : loglik_by_method) {
      std::cout << "  " << name << ": " << util::mean(vals) - ll_exact << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rbc_compare error: " << e.what() << "\n";
    return 1;
  }
}
