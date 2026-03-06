#include "filters/bootstrap_pf.hpp"
#include "filters/rbpf.hpp"
#include "models/nk_elb.hpp"
#include "models/nk_global.hpp"
#include "statespace/linear_gaussian.hpp"
#include "util/io.hpp"
#include "util/rng.hpp"
#include "util/stats.hpp"
#include "util/timer.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct RunResult {
  double loglik = 0.0;
  double runtime_ms = 0.0;
  std::vector<double> ess;
};

enum class AnchorMode { Shared = 0, Clustered = 1, PerParticle = 2 };

struct NkExperimentConfig {
  models::NkParams p;
  std::filesystem::path out_dir;
  std::uint64_t seed_data = 0;
  double shock_eps_r_irf = -2.0;
  int omega_horizon = 1;  // number of anticipated innovations at t+2.. included in omega (per shock)
  AnchorMode anchor_mode = AnchorMode::Shared;
};

enum class PppfIMeasMode { Censored = 0, Linear = 1 };

struct PppfOptions {
  double omega_var = 1.0;
  int omega_horizon = 1;
  models::PppfOmegaMode omega_mode = models::PppfOmegaMode::ROnly;
  bool optimal_index_proposal = true;
  AnchorMode anchor_mode = AnchorMode::Shared;
  PppfIMeasMode i_meas_mode = PppfIMeasMode::Censored;
  models::PppfRegimeMode regime_mode = models::PppfRegimeMode::EndogenousByNode;
};

double nk_shadow_rate_gap(const models::NkParams& p, const Eigen::Vector4d& z) {
  const double i_shadow = p.i_ss + p.phi_pi * z(1) + p.phi_x * z(0) + z(3);
  return i_shadow - p.i_lower;
}

double nk_elb_risk_score(const models::NkParams& p, const Eigen::Vector4d& z,
                         models::PppfOmegaMode omega_mode) {
  const double gap_now = nk_shadow_rate_gap(p, z);
  const Eigen::VectorXd omega0 = Eigen::VectorXd::Zero(0);
  const Eigen::Vector4d z_det =
      models::nk_transition_markov_anticipated(p, z, 0.0, 0.0, omega0, omega_mode);
  const double gap_det = nk_shadow_rate_gap(p, z_det);
  return std::min(gap_now, gap_det);
}

int nk_elb_risk_group(const models::NkParams& p, const Eigen::Vector4d& z,
                      models::PppfOmegaMode omega_mode) {
  const double risk = nk_elb_risk_score(p, z, omega_mode);
  if (risk <= -0.02) return 0;
  if (risk <= -0.01) return 1;
  if (risk <= 0.0) return 2;
  if (risk <= 0.005) return 3;
  if (risk <= 0.01) return 4;
  if (risk <= 0.02) return 5;
  return 6;
}

std::vector<int> nk_anchor_groups_elb_risk(const models::NkParams& p,
                                           const std::vector<Eigen::VectorXd>& means,
                                           const Eigen::VectorXd& weights,
                                           models::PppfOmegaMode omega_mode) {
  const int N = static_cast<int>(means.size());
  if (weights.size() != N) throw std::invalid_argument("nk_anchor_groups_elb_risk: size mismatch");
  std::vector<int> groups(N, 6);
  std::vector<std::pair<double, int>> risky_order;
  risky_order.reserve(N);
  double risky_mass = 0.0;
  for (int i = 0; i < N; ++i) {
    const double risk = nk_elb_risk_score(p, static_cast<const Eigen::Vector4d&>(means[i]), omega_mode);
    if (risk > 0.02) continue;
    risky_order.emplace_back(risk, i);
    risky_mass += weights(i);
  }
  if (risky_order.empty() || risky_mass <= 0.0) return groups;

  std::sort(risky_order.begin(), risky_order.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  double cum = 0.0;
  int g = 0;
  for (const auto& [risk, idx] : risky_order) {
    (void)risk;
    const double mass_mid = (cum + 0.5 * weights(idx)) / risky_mass;
    while (g < 5 && mass_mid > static_cast<double>(g + 1) / 6.0) ++g;
    groups[idx] = g;
    cum += weights(idx);
  }
  return groups;
}

struct NkLinearCoeffs {
  double a_r = 0.0, a_nu = 0.0;  // x = a_r * r + a_nu * nu
  double b_r = 0.0, b_nu = 0.0;  // pi = b_r * r + b_nu * nu
  double i_r = 0.0, i_nu = 0.0;  // i = i_ss + i_r * r + i_nu * nu   (no ELB)
};

NkLinearCoeffs solve_nk_linear_no_elb(const models::NkParams& p) {
  // Closed-form undetermined-coefficients solution for the NK model with no ELB:
  //   x_t = E x_{t+1} - (1/sigma)(i_t - E pi_{t+1} - r_t)
  //   pi_t = beta E pi_{t+1} + kappa x_t
  //   i_t = i_ss + phi_pi pi_t + phi_x x_t + nu_t
  // with r_t, nu_t AR(1).
  auto C = [&](double rho) {
    const double denom = 1.0 - p.beta * rho;
    return p.phi_x + p.kappa * (p.phi_pi - rho) / denom;
  };

  const double Cr = C(p.rho_r);
  const double Cn = C(p.rho_nu);

  NkLinearCoeffs co;
  co.a_r = 1.0 / (p.sigma * (1.0 - p.rho_r) + Cr);
  co.a_nu = -1.0 / (p.sigma * (1.0 - p.rho_nu) + Cn);

  co.b_r = (p.kappa * co.a_r) / (1.0 - p.beta * p.rho_r);
  co.b_nu = (p.kappa * co.a_nu) / (1.0 - p.beta * p.rho_nu);

  co.i_r = p.phi_pi * co.b_r + p.phi_x * co.a_r;
  co.i_nu = p.phi_pi * co.b_nu + p.phi_x * co.a_nu + 1.0;
  return co;
}

NkLinearCoeffs linearize_occbin_policy_no_elb(const models::NkParams& p, double fd_h = 1e-4) {
  // Numerical policy linearization implied by the *finite-horizon OccBin/EP* mapping in this repo.
  // This is used to construct an "exact" Kalman likelihood for the same approximate model that the
  // OccBin and PPPF filters are actually evaluating.
  std::vector<double> zeros_r(p.horizon, 0.0);
  std::vector<double> zeros_nu(p.horizon, 0.0);

  auto eval = [&](double r, double nu) {
    const auto path = solve_nk_occbin_path(p, r, nu, zeros_r, zeros_nu);
    return std::pair<double, double>(path.x[0], path.pi[0]);
  };

  const auto [xrp, pirp] = eval(+fd_h, 0.0);
  const auto [xrm, pirm] = eval(-fd_h, 0.0);
  const auto [xnp, pinp] = eval(0.0, +fd_h);
  const auto [xnm, pinm] = eval(0.0, -fd_h);

  NkLinearCoeffs co;
  co.a_r = (xrp - xrm) / (2.0 * fd_h);
  co.a_nu = (xnp - xnm) / (2.0 * fd_h);
  co.b_r = (pirp - pirm) / (2.0 * fd_h);
  co.b_nu = (pinp - pinm) / (2.0 * fd_h);
  co.i_r = p.phi_pi * co.b_r + p.phi_x * co.a_r;
  co.i_nu = p.phi_pi * co.b_nu + p.phi_x * co.a_nu + 1.0;
  return co;
}

RunResult run_plc_copf_pf(const models::NkParams& p, const models::NkGlobalGrid& grid, int N,
                          const Eigen::Vector2d& mean0, const Eigen::Matrix2d& cov0,
                          const std::vector<Eigen::VectorXd>& y, const Eigen::Matrix3d& R,
                          std::uint64_t seed) {
  util::Rng rng(seed);
  util::Timer timer;

  struct Particle {
    Eigen::Vector2d s;  // [r, nu]
  };

  const Eigen::Matrix2d A = (Eigen::Matrix2d() << p.rho_r, 0.0, 0.0, p.rho_nu).finished();
  const Eigen::Matrix2d Q = (Eigen::Matrix2d() << p.sigma_r * p.sigma_r, 0.0, 0.0,
                         p.sigma_nu * p.sigma_nu)
                                .finished();

  const Eigen::Matrix2d P0_sqrt = cov0.llt().matrixL();

  std::vector<Particle> particles(N);
  for (int i = 0; i < N; ++i) {
    particles[i].s = mean0 + P0_sqrt * rng.normal_vec(2);
  }
  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));

  RunResult res;
  res.ess.reserve(static_cast<int>(y.size()));

  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd logw_new(N);
    for (int i = 0; i < N; ++i) {
      const Eigen::Vector2d s_prev = particles[i].s;
      const Eigen::Vector2d m = A * s_prev;
      const Eigen::Matrix2d P = Q;

      // PLC-COPF proposal: treat the PLC interpolation as locally affine at the prior mean m.
      Eigen::Matrix<double, 3, 2> H;
      const Eigen::Vector3d y_m = models::nk_observables_plc_jacobian(grid, m(0), m(1), &H);
      Eigen::Matrix3d S = H * P * H.transpose() + R;

      Eigen::LLT<Eigen::Matrix3d> llt(S);
      if (llt.info() != Eigen::Success) {
        S += 1e-12 * Eigen::Matrix3d::Identity();
        llt.compute(S);
        if (llt.info() != Eigen::Success) throw std::runtime_error("plc_copf_pf: S not PD");
      }
      const Eigen::Matrix<double, 2, 3> K =
          P * H.transpose() * llt.solve(Eigen::Matrix3d::Identity());
      const Eigen::Vector2d m_post = m + K * (y[t] - y_m);
      const Eigen::Matrix2d P_post =
          (Eigen::Matrix2d::Identity() - K * H) * P * (Eigen::Matrix2d::Identity() - K * H).transpose() +
          K * R * K.transpose();

      Eigen::LLT<Eigen::Matrix2d> lltP(P_post);
      Eigen::Matrix2d L = Eigen::Matrix2d::Zero();
      if (lltP.info() == Eigen::Success) {
        L = lltP.matrixL();
      } else {
        const Eigen::Matrix2d Pj = P_post + 1e-12 * Eigen::Matrix2d::Identity();
        lltP.compute(Pj);
        if (lltP.info() != Eigen::Success) throw std::runtime_error("plc_copf_pf: P_post not PD");
        L = lltP.matrixL();
      }

      const Eigen::Vector2d s_t = m_post + L * rng.normal_vec(2);
      particles[i].s = s_t;

      // Under the locally affine PLC measurement, this is the conditionally optimal proposal,
      // so the incremental weight is the predictive likelihood p(y_t | s_{t-1}).
      const double ll_pred = util::log_mvnorm_pdf(y[t], y_m, S);
      logw_new(i) = logw(i) + ll_pred;
    }

    const double logZ = util::log_sum_exp(logw_new);
    res.loglik += logZ;
    logw_new.array() -= logZ;
    const double ess_t = util::ess_from_logw(logw_new);
    res.ess.push_back(ess_t);

    if (ess_t < 0.5 * static_cast<double>(N)) {
      const Eigen::VectorXd w_norm = logw_new.array().exp().matrix();
      const std::vector<int> idx = filters::systematic_resample(w_norm, rng);
      std::vector<Particle> new_particles(N);
      for (int i = 0; i < N; ++i) new_particles[i] = particles[idx[i]];
      particles.swap(new_particles);
      logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
    } else {
      logw = logw_new;
    }
  }

  res.runtime_ms = timer.elapsed_ms();
  return res;
}

RunResult run_occbin_bootstrap_pf(const models::NkParams& p, int N, const Eigen::Vector4d& mean0,
                                  const Eigen::Matrix4d& cov0,
                                  const std::vector<Eigen::VectorXd>& y,
                                  const Eigen::Matrix3d& R, std::uint64_t seed) {
  util::Rng rng(seed);
  util::Timer timer;

  const Eigen::Matrix4d P0_sqrt = cov0.llt().matrixL();

  const auto transition = [&](Eigen::VectorXd* state, util::Rng& rng_in, int /*t*/) {
    const double eps_r = rng_in.normal();
    const double eps_nu = rng_in.normal();
    const Eigen::Vector4d z = (*state);
    const Eigen::Vector4d z_next = models::nk_transition_markov(p, z, eps_r, eps_nu);
    (*state) = z_next;
  };

  const auto log_obs = [&](const Eigen::VectorXd& state, const Eigen::VectorXd& y_t, int /*t*/) {
    const Eigen::Vector4d z = state;
    const Eigen::Vector3d yhat = models::nk_observables(p, z);
    return util::log_mvnorm_pdf(y_t, yhat, R);
  };

  const filters::PfDiagnostics diag =
      filters::bootstrap_pf(N, mean0, P0_sqrt, y, transition, log_obs, rng, 0.5);

  RunResult res;
  res.loglik = diag.loglik;
  res.runtime_ms = timer.elapsed_ms();
  res.ess = diag.ess;
  return res;
}

RunResult run_pppf_mixture_rbpf(const models::NkParams& p, int N, const Eigen::Vector4d& mean0,
                               const Eigen::Matrix4d& cov0, const std::vector<Eigen::VectorXd>& y,
                               const Eigen::Matrix3d& R, std::uint64_t seed, const PppfOptions& opt) {
  util::Rng rng(seed);
  util::Timer timer;
  const filters::AnchorGroupsFn anchor_groups_fn =
      (opt.anchor_mode == AnchorMode::Clustered)
          ? filters::AnchorGroupsFn([&](int /*t*/, const std::vector<Eigen::VectorXd>& z_prev,
                                        const Eigen::VectorXd& w_norm,
                                        const Eigen::VectorXd& /*ref_shared*/) {
              return nk_anchor_groups_elb_risk(p, z_prev, w_norm, opt.omega_mode);
            })
          : filters::AnchorGroupsFn();
  const int num_anchor_groups = (opt.anchor_mode == AnchorMode::Clustered) ? 7 : 1;

  const auto mixture_builder = [&](int /*t*/, const Eigen::VectorXd& ref_prev) {
    const Eigen::Vector4d z_ref = ref_prev;
    return models::build_pppf_mixture(p, z_ref, opt.omega_horizon, opt.omega_var, /*eps_r_mean=*/0.0,
                                      /*eps_r_var=*/1.0, /*eps_nu_var=*/1.0, opt.omega_mode,
                                      opt.regime_mode);
  };

  const auto obs_update = [&](int /*t*/, const Eigen::VectorXd& y_t, int /*k*/,
                              statespace::KalmanState* st) -> double {
    // Linear measurements: y = [pi, x, i]
    Eigen::Matrix<double, 2, 4> Hlin = Eigen::Matrix<double, 2, 4>::Zero();
    Hlin(0, 1) = 1.0;  // pi
    Hlin(1, 0) = 1.0;  // x
    Eigen::Vector2d dlin = Eigen::Vector2d::Zero();
    Eigen::Matrix2d Rlin = Eigen::Matrix2d::Zero();
    Rlin(0, 0) = R(0, 0);
    Rlin(1, 1) = R(1, 1);
    const Eigen::Vector2d ylin = y_t.head(2);

    double ll = statespace::kalman_update(Hlin, dlin, Rlin, ylin, st);

    const Eigen::RowVector4d h_rule =
        (Eigen::RowVector4d() << p.phi_x, p.phi_pi, 0.0, 1.0).finished();
    if (opt.i_meas_mode == PppfIMeasMode::Censored) {
      ll += statespace::kalman_update_censored_lower(h_rule, p.i_ss, p.i_lower, R(2, 2), y_t(2), st);
    } else {
      Eigen::Matrix<double, 1, 4> Hi;
      Hi = h_rule;
      Eigen::VectorXd di(1);
      di(0) = p.i_ss;
      Eigen::MatrixXd Ri(1, 1);
      Ri(0, 0) = R(2, 2);
      Eigen::VectorXd yi(1);
      yi(0) = y_t(2);
      ll += statespace::kalman_update(Hi, di, Ri, yi, st);
    }
    return ll;
  };

  const filters::RbpfDiagnostics diag =
      filters::rbpf(N, mean0, cov0, y, mixture_builder, obs_update, rng, 0.5, opt.optimal_index_proposal,
                    opt.anchor_mode == AnchorMode::PerParticle, num_anchor_groups, anchor_groups_fn);

  RunResult res;
  res.loglik = diag.loglik;
  res.runtime_ms = timer.elapsed_ms();
  res.ess = diag.ess;
  return res;
}

void expected_irf_pppf_anchor_mc(const models::NkParams& p, int T, double shock_eps_r,
                                 int omega_horizon, AnchorMode anchor_mode, std::vector<double>* x_out,
                                 std::vector<double>* pi_out, std::vector<double>* i_out,
                                 int num_particles = 20000, std::uint64_t seed = 20260306ULL) {
  if (!x_out || !pi_out || !i_out) throw std::invalid_argument("expected_irf_pppf_anchor_mc: null out");
  if (num_particles <= 0) throw std::invalid_argument("expected_irf_pppf_anchor_mc: num_particles<=0");

  util::Rng rng(seed);
  std::vector<Eigen::Vector4d> particles(num_particles, Eigen::Vector4d::Zero());

  x_out->assign(T, 0.0);
  pi_out->assign(T, 0.0);
  i_out->assign(T, 0.0);

  auto sample_gaussian = [&](const Eigen::Vector4d& mean, const Eigen::Matrix4d& cov) {
    Eigen::Matrix4d Q = cov;
    Eigen::LLT<Eigen::Matrix4d> llt(Q);
    if (llt.info() != Eigen::Success) {
      Q += 1e-10 * Eigen::Matrix4d::Identity();
      llt.compute(Q);
      if (llt.info() != Eigen::Success) {
        throw std::runtime_error("expected_irf_pppf_anchor_mc: Q not PD");
      }
    }
    return mean + llt.matrixL() * rng.normal_vec(4);
  };

  for (int t = 0; t < T; ++t) {
    const double eps_mean = (t == 0) ? shock_eps_r : 0.0;
    // For an IRF, the period-0 natural-rate impulse is realized rather than integrated over.
    const double eps_var_r = (t == 0) ? 0.0 : 1.0;
    const double eps_var_nu = (t == 0) ? 0.0 : 1.0;
    Eigen::Vector4d z_ref = Eigen::Vector4d::Zero();
    for (const auto& z : particles) z_ref += z;
    z_ref /= static_cast<double>(num_particles);

    statespace::GaussianMixtureTransition mix_shared;
    std::vector<statespace::GaussianMixtureTransition> mix_groups;
    std::vector<char> has_group;
    std::vector<int> particle_group(num_particles, 0);

    mix_shared = models::build_pppf_mixture(p, z_ref, /*omega_horizon=*/omega_horizon, /*omega_var=*/1.0,
                                            eps_mean, eps_var_r, eps_var_nu);
    mix_shared.check();

    if (anchor_mode == AnchorMode::Clustered) {
      mix_groups.resize(7);
      has_group.assign(7, 0);
      std::vector<int> group_count(7, 0);
      std::vector<Eigen::Vector4d> group_ref(7, Eigen::Vector4d::Zero());
      std::vector<Eigen::VectorXd> particle_means(num_particles);
      Eigen::VectorXd weights =
          Eigen::VectorXd::Constant(num_particles, 1.0 / static_cast<double>(num_particles));
      for (int n = 0; n < num_particles; ++n) particle_means[n] = particles[n];
      particle_group = nk_anchor_groups_elb_risk(p, particle_means, weights, models::PppfOmegaMode::ROnly);
      for (int n = 0; n < num_particles; ++n) {
        const int g = particle_group[n];
        particle_group[n] = g;
        ++group_count[g];
        group_ref[g] += particles[n];
      }
      for (int g = 0; g < 7; ++g) {
        if (group_count[g] == 0) continue;
        mix_groups[g] = models::build_pppf_mixture(p, group_ref[g] / static_cast<double>(group_count[g]),
                                                   /*omega_horizon=*/omega_horizon, /*omega_var=*/1.0,
                                                   eps_mean, eps_var_r, eps_var_nu);
        mix_groups[g].check();
        has_group[g] = 1;
      }
    }

    std::vector<Eigen::Vector4d> next_particles(num_particles);
    double mean_x = 0.0;
    double mean_pi = 0.0;
    double mean_i = 0.0;

    for (int n = 0; n < num_particles; ++n) {
      const statespace::GaussianMixtureTransition* mix = &mix_shared;
      if (anchor_mode == AnchorMode::PerParticle) {
        mix_shared = models::build_pppf_mixture(
            p, particles[n], /*omega_horizon=*/omega_horizon, /*omega_var=*/1.0, eps_mean, eps_var_r,
            eps_var_nu);
        mix_shared.check();
        mix = &mix_shared;
      }
      if (anchor_mode == AnchorMode::Clustered && has_group[particle_group[n]]) {
        mix = &mix_groups[particle_group[n]];
      }
      const int k = rng.categorical(mix->weights);
      const auto& tr = mix->components[k];
      const Eigen::Vector4d mean = tr.A * particles[n] + tr.a;
      next_particles[n] = sample_gaussian(mean, tr.Q);
      mean_x += next_particles[n](0);
      mean_pi += next_particles[n](1);
      mean_i += models::nk_policy_rate(p, next_particles[n]);
    }

    particles.swap(next_particles);
    (*x_out)[t] = mean_x / static_cast<double>(num_particles);
    (*pi_out)[t] = mean_pi / static_cast<double>(num_particles);
    (*i_out)[t] = mean_i / static_cast<double>(num_particles);
  }
}

void write_summary_json(const std::filesystem::path& path,
                        const std::map<std::string, std::vector<double>>& loglik_by_method,
                        const std::map<std::string, std::vector<double>>& rt_by_method,
                        const std::map<std::string, double>& mean_ess_by_method, int T, int N,
                        int R) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"config\": {\n";
  oss << "    \"T\": " << T << ",\n";
  oss << "    \"N\": " << N << ",\n";
  oss << "    \"R\": " << R << "\n";
  oss << "  },\n";
  oss << "  \"methods\": {\n";

  bool first = true;
  for (const auto& kv : loglik_by_method) {
    const std::string& name = kv.first;
    const auto& ll = kv.second;
    const auto& rt = rt_by_method.at(name);
    const double mll = util::mean(ll);
    const double sll = util::stdev(ll);
    const double mrt = util::mean(rt);
    const double srt = util::stdev(rt);
    const double mess = mean_ess_by_method.at(name);

    if (!first) oss << ",\n";
    first = false;
    oss << "    \"" << util::json_escape(name) << "\": {\n";
    oss << "      \"mean_loglik\": " << mll << ",\n";
    oss << "      \"sd_loglik\": " << sll << ",\n";
    oss << "      \"mean_runtime_ms\": " << mrt << ",\n";
    oss << "      \"sd_runtime_ms\": " << srt << ",\n";
    oss << "      \"mean_ess\": " << mess << "\n";
    oss << "    }";
  }
  oss << "\n  }\n";
  oss << "}\n";
  util::write_text_file(path, oss.str());
}

void run_nk_experiment(const NkExperimentConfig& cfg, int T, int N, int R, const Eigen::Matrix3d& Rm) {
  const models::NkParams& p = cfg.p;
  const std::filesystem::path& out_dir = cfg.out_dir;

  util::ensure_dir(out_dir);

  // Solve the global discrete-state benchmark (Tauchen grid) for this configuration.
  models::NkGlobalGrid grid = models::make_default_global_grid(p);
  models::NkGlobalSolveOptions gopt;
  gopt.max_iter = 80;
  gopt.tol = 1e-10;
  gopt.fb_eps = 1e-12;
  util::Timer tglob;
  models::solve_nk_global_policy(p, &grid, gopt);
  std::cout << "Solved global NK policy (Tauchen grid) in " << tglob.elapsed_ms() << " ms\n";

  const int nr = static_cast<int>(grid.r_grid.size());
  const int nn = static_cast<int>(grid.nu_grid.size());
  const int S = nr * nn;

  // Simulate a dataset from the global discrete-state benchmark.
  util::Rng rng_data(cfg.seed_data);

  std::vector<Eigen::Vector4d> z_true(T);
  std::vector<Eigen::VectorXd> y_obs(T);

  Eigen::VectorXd d0 = models::initial_distribution_bilinear(grid, 0.0, 0.0);
  std::vector<double> d0v(S);
  for (int s = 0; s < S; ++s) d0v[s] = d0(s);
  int s_prev = rng_data.categorical(d0v);

  int bind_count = 0;
  for (int t = 0; t < T; ++t) {
    const int i = s_prev / nn;
    const int j = s_prev % nn;
    const double r = grid.r_grid[i];
    const double nu = grid.nu_grid[j];
    const double x = grid.x(i, j);
    const double pi = grid.pi(i, j);
    const double irate = grid.i(i, j);

    Eigen::Vector4d z_t;
    z_t << x, pi, r, nu;

    Eigen::Vector3d y_t;
    y_t << pi, x, irate;
    Eigen::Vector3d y_noisy = y_t;
    y_noisy += Eigen::Vector3d(std::sqrt(Rm(0, 0)) * rng_data.normal(),
                               std::sqrt(Rm(1, 1)) * rng_data.normal(),
                               std::sqrt(Rm(2, 2)) * rng_data.normal());

    y_obs[t] = y_noisy;
    z_true[t] = z_t;

    if (std::abs(irate - p.i_lower) < 1e-12) ++bind_count;

    // Sample next discrete state.
    std::vector<double> row(S);
    for (int sp = 0; sp < S; ++sp) row[sp] = grid.P(s_prev, sp);
    s_prev = rng_data.categorical(row);
  }

  std::cout << "Simulated NK dataset: T=" << T << ", ELB binds in " << bind_count << " periods\n";

  const Eigen::Vector4d mean0 = Eigen::Vector4d::Zero();
  Eigen::Matrix4d cov0 = Eigen::Matrix4d::Zero();
  cov0(0, 0) = 0.02 * 0.02;
  cov0(1, 1) = 0.02 * 0.02;
  cov0(2, 2) = 0.05 * 0.05;
  cov0(3, 3) = 0.05 * 0.05;

  // Repeated likelihood runs.
  const std::filesystem::path ll_csv = out_dir / "loglik_repeats.csv";
  std::ofstream ll_out(ll_csv);
  if (!ll_out) throw std::runtime_error("failed to open: " + ll_csv.string());
  util::write_csv_header(ll_out, {"method", "rep", "loglik", "runtime_ms"});

  const std::filesystem::path ess_csv = out_dir / "ess.csv";
  std::ofstream ess_out(ess_csv);
  if (!ess_out) throw std::runtime_error("failed to open: " + ess_csv.string());
  util::write_csv_header(ess_out, {"method", "t", "ess"});

  std::map<std::string, std::vector<double>> loglik_by_method;
  std::map<std::string, std::vector<double>> rt_by_method;
  std::map<std::string, double> mean_ess_by_method;

  for (const std::string& method :
       {"occbin_bootstrap_pf", "pppf_mixture_rbpf", "plc_copf_pf", "global_discrete_exact"}) {
    loglik_by_method[method] = {};
    rt_by_method[method] = {};
  }

  // Exact likelihood for the discretized global benchmark (forward algorithm on the HMM).
  {
    util::Timer t_exact;
    Eigen::VectorXd alpha = d0;
    double ll = 0.0;
    for (int t = 0; t < T; ++t) {
      // Predict: alpha_pred(sp) = sum_s alpha(s) P(s, sp)
      Eigen::VectorXd alpha_pred = grid.P.transpose() * alpha;

      // Weight by observation likelihood at each discrete state.
      Eigen::VectorXd logg(S);
      for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nn; ++j) {
          const int s = i * nn + j;
          Eigen::Vector3d yhat;
          yhat << grid.pi(i, j), grid.x(i, j), grid.i(i, j);
          logg(s) = util::log_mvnorm_pdf(y_obs[t], yhat, Rm);
        }
      }
      Eigen::VectorXd g = logg.array().exp().matrix();
      Eigen::VectorXd alpha_unnorm = alpha_pred.array() * g.array();
      const double z = alpha_unnorm.sum();
      if (!(z > 0.0) || !std::isfinite(z)) throw std::runtime_error("global_exact: alpha sum");
      ll += std::log(z);
      alpha = alpha_unnorm / z;
    }

    loglik_by_method["global_discrete_exact"].push_back(ll);
    rt_by_method["global_discrete_exact"].push_back(t_exact.elapsed_ms());
    mean_ess_by_method["global_discrete_exact"] = -1.0;
    util::write_csv_row(ll_out, "global_discrete_exact", 0, ll, t_exact.elapsed_ms());
  }

  for (int rep = 0; rep < R; ++rep) {
    const std::uint64_t seed_base = 900000ULL + static_cast<std::uint64_t>(rep) * 1000ULL;

    {
      const RunResult rr = run_occbin_bootstrap_pf(p, N, mean0, cov0, y_obs, Rm, seed_base + 1);
      loglik_by_method["occbin_bootstrap_pf"].push_back(rr.loglik);
      rt_by_method["occbin_bootstrap_pf"].push_back(rr.runtime_ms);
      util::write_csv_row(ll_out, "occbin_bootstrap_pf", rep, rr.loglik, rr.runtime_ms);

      if (rep == 0) {
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(rr.ess.size()); ++t) {
          util::write_csv_row(ess_out, "occbin_bootstrap_pf", t, rr.ess[t]);
          ess_mean += rr.ess[t];
        }
        ess_mean /= static_cast<double>(rr.ess.size());
        mean_ess_by_method["occbin_bootstrap_pf"] = ess_mean;
      }
    }

    {
      PppfOptions opt;
      opt.omega_var = 1.0;
      opt.omega_horizon = cfg.omega_horizon;
      opt.optimal_index_proposal = true;
      opt.anchor_mode = cfg.anchor_mode;
      opt.i_meas_mode = PppfIMeasMode::Censored;
      opt.regime_mode = models::PppfRegimeMode::EndogenousByNode;
      const RunResult rr = run_pppf_mixture_rbpf(p, N, mean0, cov0, y_obs, Rm, seed_base + 2, opt);
      loglik_by_method["pppf_mixture_rbpf"].push_back(rr.loglik);
      rt_by_method["pppf_mixture_rbpf"].push_back(rr.runtime_ms);
      util::write_csv_row(ll_out, "pppf_mixture_rbpf", rep, rr.loglik, rr.runtime_ms);

      if (rep == 0) {
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(rr.ess.size()); ++t) {
          util::write_csv_row(ess_out, "pppf_mixture_rbpf", t, rr.ess[t]);
          ess_mean += rr.ess[t];
        }
        ess_mean /= static_cast<double>(rr.ess.size());
        mean_ess_by_method["pppf_mixture_rbpf"] = ess_mean;
      }
    }

    {
      const Eigen::Vector2d s_mean0 = Eigen::Vector2d::Zero();
      const Eigen::Matrix2d s_cov0 =
          (Eigen::Matrix2d() << 0.05 * 0.05, 0.0, 0.0, 0.05 * 0.05).finished();
      const RunResult rr = run_plc_copf_pf(p, grid, N, s_mean0, s_cov0, y_obs, Rm, seed_base + 3);
      loglik_by_method["plc_copf_pf"].push_back(rr.loglik);
      rt_by_method["plc_copf_pf"].push_back(rr.runtime_ms);
      util::write_csv_row(ll_out, "plc_copf_pf", rep, rr.loglik, rr.runtime_ms);

      if (rep == 0) {
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(rr.ess.size()); ++t) {
          util::write_csv_row(ess_out, "plc_copf_pf", t, rr.ess[t]);
          ess_mean += rr.ess[t];
        }
        ess_mean /= static_cast<double>(rr.ess.size());
        mean_ess_by_method["plc_copf_pf"] = ess_mean;
      }
    }
  }

  write_summary_json(out_dir / "summary.json", loglik_by_method, rt_by_method, mean_ess_by_method,
                     T, N, R);
  std::cout << "Wrote " << (out_dir / "summary.json").string() << "\n";

  const double ll_exact = loglik_by_method["global_discrete_exact"][0];
  std::cout << "Mean loglik gaps vs global_discrete_exact: "
            << "OccBin=" << util::mean(loglik_by_method["occbin_bootstrap_pf"]) - ll_exact << ", "
            << "PPPF=" << util::mean(loglik_by_method["pppf_mixture_rbpf"]) - ll_exact << ", "
            << "PLC=" << util::mean(loglik_by_method["plc_copf_pf"]) - ll_exact << "\n";

  // IRFs: a negative natural-rate impulse at t=0.
  const int Tirf = 40;
  const double shock_eps_r = cfg.shock_eps_r_irf;

  std::vector<double> irf_x_occ(Tirf), irf_pi_occ(Tirf), irf_i_occ(Tirf);
  std::vector<double> irf_x_pppf(Tirf), irf_pi_pppf(Tirf), irf_i_pppf(Tirf);
  std::vector<double> irf_x_glob(Tirf), irf_pi_glob(Tirf), irf_i_glob(Tirf);
  std::vector<double> irf_x_plc(Tirf), irf_pi_plc(Tirf), irf_i_plc(Tirf);

  double r_prev = 0.0;
  double nu_prev = 0.0;
  for (int t = 0; t < Tirf; ++t) {
    const double eps_r = (t == 0) ? shock_eps_r : 0.0;
    const double eps_nu = 0.0;
    const double r_t = p.rho_r * r_prev + p.sigma_r * eps_r;
    const double nu_t = p.rho_nu * nu_prev + p.sigma_nu * eps_nu;

    std::vector<double> zeros_r(p.horizon, 0.0);
    std::vector<double> zeros_nu(p.horizon, 0.0);
    const models::NkPath path = models::solve_nk_occbin_path(p, r_t, nu_t, zeros_r, zeros_nu);

    irf_x_occ[t] = path.x[0];
    irf_pi_occ[t] = path.pi[0];
    irf_i_occ[t] = path.i[0];

    r_prev = r_t;
    nu_prev = nu_t;
  }

  // PPPF expected IRF: simulate the approximate PPPF transition kernel directly.
  // This is more comparable to the stochastic global benchmark than collapsing the mixture
  // recursion to a single Gaussian mean/covariance process at each step.
  expected_irf_pppf_anchor_mc(p, Tirf, shock_eps_r, cfg.omega_horizon, cfg.anchor_mode, &irf_x_pppf,
                              &irf_pi_pppf, &irf_i_pppf);

  // Global stochastic solution on a discrete Markov chain (state-space, expectation-consistent).
  models::expected_irf_global(p, grid, shock_eps_r, Tirf, &irf_x_glob, &irf_pi_glob, &irf_i_glob);

  // PLC interpolation IRF: propagate mean exogenous state under the impulse and map to observables.
  double r_m = 0.0;
  double nu_m = 0.0;
  for (int t = 0; t < Tirf; ++t) {
    const double eps_r = (t == 0) ? shock_eps_r : 0.0;
    const double eps_nu = 0.0;
    r_m = p.rho_r * r_m + p.sigma_r * eps_r;
    nu_m = p.rho_nu * nu_m + p.sigma_nu * eps_nu;
    const Eigen::Vector3d yplc = models::nk_observables_plc(p, grid, r_m, nu_m);
    irf_pi_plc[t] = yplc(0);
    irf_x_plc[t] = yplc(1);
    irf_i_plc[t] = yplc(2);
  }

  const std::filesystem::path irf_csv = out_dir / "irf.csv";
  std::ofstream irf_out(irf_csv);
  if (!irf_out) throw std::runtime_error("failed to open: " + irf_csv.string());
  util::write_csv_header(
      irf_out, {"t", "x_global", "pi_global", "i_global", "x_occbin", "pi_occbin", "i_occbin",
                "x_pppf", "pi_pppf", "i_pppf", "x_plc", "pi_plc", "i_plc"});
  for (int t = 0; t < Tirf; ++t) {
    util::write_csv_row(irf_out, t, irf_x_glob[t], irf_pi_glob[t], irf_i_glob[t], irf_x_occ[t],
                        irf_pi_occ[t], irf_i_occ[t], irf_x_pppf[t], irf_pi_pppf[t], irf_i_pppf[t],
                        irf_x_plc[t], irf_pi_plc[t], irf_i_plc[t]);
  }
  std::cout << "Wrote " << irf_csv.string() << "\n";

  auto rmse = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (int t = 0; t < Tirf; ++t) {
      const double d = a[t] - b[t];
      s += d * d;
    }
    return std::sqrt(s / static_cast<double>(Tirf));
  };
  std::cout << "IRF RMSE vs global: OccBin (x,pi,i)=(" << rmse(irf_x_occ, irf_x_glob) << ", "
            << rmse(irf_pi_occ, irf_pi_glob) << ", " << rmse(irf_i_occ, irf_i_glob)
            << "), PPPF=(" << rmse(irf_x_pppf, irf_x_glob) << ", " << rmse(irf_pi_pppf, irf_pi_glob)
            << ", " << rmse(irf_i_pppf, irf_i_glob) << ")\n";

  std::cout << "Wrote " << ll_csv.string() << "\n";
  std::cout << "Wrote " << ess_csv.string() << "\n";
}

void run_nk_no_elb_continuous_sanity(models::NkParams p, const std::filesystem::path& out_dir, int T,
                                    int N, int R, const Eigen::Matrix3d& Rm,
                                    std::uint64_t seed_data, int omega_horizon = 1) {
  // This sanity check removes the ELB kink and uses a continuous (Gaussian) AR(1) DGP for (r,nu),
  // so that the entire model is linear-Gaussian and the likelihood is available exactly via Kalman.
  p.i_lower = -1000.0;

  util::ensure_dir(out_dir);

  // Use the policy coefficients implied by the repo's own finite-horizon OccBin mapping.
  // With no ELB and a linear NK system, this mapping should be exactly linear, so the resulting
  // Kalman likelihood is an "exact" likelihood for the same approximate model evaluated by OccBin/PPPF.
  const NkLinearCoeffs co = linearize_occbin_policy_no_elb(p);

  // Simulate (r, nu) from the continuous AR(1), then generate observables from the linear policy.
  util::Rng rng(seed_data);
  std::vector<Eigen::VectorXd> y_obs(T);

  double r = 0.0;
  double nu = 0.0;
  int bind_count = 0;
  for (int t = 0; t < T; ++t) {
    r = p.rho_r * r + p.sigma_r * rng.normal();
    nu = p.rho_nu * nu + p.sigma_nu * rng.normal();

    const double x = co.a_r * r + co.a_nu * nu;
    const double pi = co.b_r * r + co.b_nu * nu;
    const double i_rule = p.i_ss + p.phi_pi * pi + p.phi_x * x + nu;
    const double irate = std::max(p.i_lower, i_rule);
    if (std::abs(irate - p.i_lower) < 1e-12) ++bind_count;

    Eigen::Vector3d y;
    y << pi, x, irate;
    y += Eigen::Vector3d(std::sqrt(Rm(0, 0)) * rng.normal(),
                         std::sqrt(Rm(1, 1)) * rng.normal(),
                         std::sqrt(Rm(2, 2)) * rng.normal());
    y_obs[t] = y;
  }
  std::cout << "Simulated NK no-ELB continuous dataset: T=" << T << ", ELB binds in " << bind_count
            << " periods\n";

  // Exact Kalman likelihood on state s_t = [r_t, nu_t] with y_t = H s_t + d + eps.
  const Eigen::Matrix2d A = (Eigen::Matrix2d() << p.rho_r, 0.0, 0.0, p.rho_nu).finished();
  const Eigen::Matrix2d Q = (Eigen::Matrix2d() << p.sigma_r * p.sigma_r, 0.0, 0.0,
                         p.sigma_nu * p.sigma_nu)
                                .finished();

  Eigen::Matrix<double, 3, 2> H = Eigen::Matrix<double, 3, 2>::Zero();
  H(0, 0) = co.b_r;
  H(0, 1) = co.b_nu;
  H(1, 0) = co.a_r;
  H(1, 1) = co.a_nu;
  H(2, 0) = co.i_r;
  H(2, 1) = co.i_nu;

  Eigen::Vector3d d = Eigen::Vector3d::Zero();
  d(2) = p.i_ss;

  statespace::KalmanState kst;
  kst.mean = Eigen::Vector2d::Zero();
  kst.cov = (Eigen::Matrix2d() << 0.05 * 0.05, 0.0, 0.0, 0.05 * 0.05).finished();

  double ll_kalman = 0.0;
  for (int t = 0; t < T; ++t) {
    statespace::kalman_predict(A, Eigen::Vector2d::Zero(), Q, &kst);
    ll_kalman += statespace::kalman_update(H, d, Rm, y_obs[t], &kst);
  }

  // PF-based comparators (OccBin PF and PPPF RBPF), on the same continuous DGP.
  const Eigen::Vector4d mean0 = Eigen::Vector4d::Zero();
  Eigen::Matrix4d cov0 = Eigen::Matrix4d::Zero();
  cov0(0, 0) = 0.02 * 0.02;
  cov0(1, 1) = 0.02 * 0.02;
  cov0(2, 2) = 0.05 * 0.05;
  cov0(3, 3) = 0.05 * 0.05;

  const std::filesystem::path ll_csv = out_dir / "loglik_repeats.csv";
  std::ofstream ll_out(ll_csv);
  if (!ll_out) throw std::runtime_error("failed to open: " + ll_csv.string());
  util::write_csv_header(ll_out, {"method", "rep", "loglik", "runtime_ms"});
  util::write_csv_row(ll_out, "kalman_exact", 0, ll_kalman, 0.0);

  const std::filesystem::path ess_csv = out_dir / "ess.csv";
  std::ofstream ess_out(ess_csv);
  if (!ess_out) throw std::runtime_error("failed to open: " + ess_csv.string());
  util::write_csv_header(ess_out, {"method", "t", "ess"});

  std::map<std::string, std::vector<double>> loglik_by_method;
  std::map<std::string, std::vector<double>> rt_by_method;
  std::map<std::string, double> mean_ess_by_method;
  for (const std::string& method : {"occbin_bootstrap_pf", "pppf_mixture_rbpf", "kalman_exact"}) {
    loglik_by_method[method] = {};
    rt_by_method[method] = {};
  }
  loglik_by_method["kalman_exact"].push_back(ll_kalman);
  rt_by_method["kalman_exact"].push_back(0.0);
  mean_ess_by_method["kalman_exact"] = -1.0;

  for (int rep = 0; rep < R; ++rep) {
    const std::uint64_t seed_base = 910000ULL + static_cast<std::uint64_t>(rep) * 1000ULL;

    {
      const RunResult rr = run_occbin_bootstrap_pf(p, N, mean0, cov0, y_obs, Rm, seed_base + 1);
      loglik_by_method["occbin_bootstrap_pf"].push_back(rr.loglik);
      rt_by_method["occbin_bootstrap_pf"].push_back(rr.runtime_ms);
      util::write_csv_row(ll_out, "occbin_bootstrap_pf", rep, rr.loglik, rr.runtime_ms);
      if (rep == 0) {
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(rr.ess.size()); ++t) {
          util::write_csv_row(ess_out, "occbin_bootstrap_pf", t, rr.ess[t]);
          ess_mean += rr.ess[t];
        }
        ess_mean /= static_cast<double>(rr.ess.size());
        mean_ess_by_method["occbin_bootstrap_pf"] = ess_mean;
      }
    }

	    {
	      // In a linear-Gaussian no-ELB model, expectation integration is exact under certainty equivalence.
	      // To obtain a clean "all methods coincide" sanity check, we therefore set omega_var=0 so that
	      // PPPF degenerates to a single linear-Gaussian kernel (K=1) and the RBPF coincides with Kalman.
	      PppfOptions opt;
	      opt.omega_var = 0.0;
	      opt.omega_horizon = omega_horizon;
	      opt.optimal_index_proposal = true;
	      opt.anchor_mode = AnchorMode::Shared;
	      opt.i_meas_mode = PppfIMeasMode::Censored;
	      opt.regime_mode = models::PppfRegimeMode::EndogenousByNode;
	      const RunResult rr = run_pppf_mixture_rbpf(p, N, mean0, cov0, y_obs, Rm, seed_base + 2, opt);
	      loglik_by_method["pppf_mixture_rbpf"].push_back(rr.loglik);
	      rt_by_method["pppf_mixture_rbpf"].push_back(rr.runtime_ms);
	      util::write_csv_row(ll_out, "pppf_mixture_rbpf", rep, rr.loglik, rr.runtime_ms);
      if (rep == 0) {
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(rr.ess.size()); ++t) {
          util::write_csv_row(ess_out, "pppf_mixture_rbpf", t, rr.ess[t]);
          ess_mean += rr.ess[t];
        }
        ess_mean /= static_cast<double>(rr.ess.size());
        mean_ess_by_method["pppf_mixture_rbpf"] = ess_mean;
      }
    }
  }

  const std::filesystem::path summary_json = out_dir / "summary.json";
  write_summary_json(summary_json, loglik_by_method, rt_by_method, mean_ess_by_method, T, N, R);
  std::cout << "Wrote " << summary_json.string() << "\n";

  std::cout << "No-ELB sanity mean loglik gaps vs kalman_exact: "
            << "OccBin=" << util::mean(loglik_by_method["occbin_bootstrap_pf"]) - ll_kalman << ", "
            << "PPPF=" << util::mean(loglik_by_method["pppf_mixture_rbpf"]) - ll_kalman << "\n";

  std::cout << "Wrote " << ll_csv.string() << "\n";
  std::cout << "Wrote " << ess_csv.string() << "\n";
}

struct NkBenchmarkData {
  models::NkGlobalGrid grid;
  std::vector<Eigen::VectorXd> y_obs;
  Eigen::VectorXd alpha0;
  double loglik_exact = 0.0;
  int bind_count = 0;
};

double global_discrete_loglik_prefix(const models::NkGlobalGrid& grid, const Eigen::VectorXd& alpha0,
                                     const std::vector<Eigen::VectorXd>& y_obs,
                                     const Eigen::Matrix3d& Rm, int T_prefix) {
  const int nr = static_cast<int>(grid.r_grid.size());
  const int nn = static_cast<int>(grid.nu_grid.size());
  const int S = nr * nn;
  Eigen::VectorXd alpha = alpha0;
  double ll = 0.0;
  for (int t = 0; t < T_prefix; ++t) {
    Eigen::VectorXd alpha_pred = grid.P.transpose() * alpha;

    Eigen::VectorXd logg(S);
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < nn; ++j) {
        const int s = i * nn + j;
        Eigen::Vector3d yhat;
        yhat << grid.pi(i, j), grid.x(i, j), grid.i(i, j);
        logg(s) = util::log_mvnorm_pdf(y_obs[t], yhat, Rm);
      }
    }
    Eigen::VectorXd g = logg.array().exp().matrix();
    Eigen::VectorXd alpha_unnorm = alpha_pred.array() * g.array();
    const double z = alpha_unnorm.sum();
    if (!(z > 0.0) || !std::isfinite(z)) throw std::runtime_error("global_exact: alpha sum");
    ll += std::log(z);
    alpha = alpha_unnorm / z;
  }
  return ll;
}

NkBenchmarkData make_nk_elb_benchmark_data(const models::NkParams& p, int T, const Eigen::Matrix3d& Rm,
                                          std::uint64_t seed_data) {
  // Solve global policy (discretized Markov chain).
  models::NkGlobalGrid grid = models::make_default_global_grid(p);
  models::NkGlobalSolveOptions gopt;
  gopt.max_iter = 80;
  gopt.tol = 1e-10;
  gopt.fb_eps = 1e-12;
  util::Timer tglob;
  models::solve_nk_global_policy(p, &grid, gopt);
  std::cout << "Solved global NK policy (Tauchen grid) in " << tglob.elapsed_ms() << " ms\n";

  const int nr = static_cast<int>(grid.r_grid.size());
  const int nn = static_cast<int>(grid.nu_grid.size());
  const int S = nr * nn;

  // Simulate data from the discrete benchmark.
  util::Rng rng_data(seed_data);
  std::vector<Eigen::VectorXd> y_obs(T);

  Eigen::VectorXd d0 = models::initial_distribution_bilinear(grid, 0.0, 0.0);
  std::vector<double> d0v(S);
  for (int s = 0; s < S; ++s) d0v[s] = d0(s);
  int s_prev = rng_data.categorical(d0v);

  int bind_count = 0;
  for (int t = 0; t < T; ++t) {
    const int i = s_prev / nn;
    const int j = s_prev % nn;
    const double x = grid.x(i, j);
    const double pi = grid.pi(i, j);
    const double irate = grid.i(i, j);

    Eigen::Vector3d y_t;
    y_t << pi, x, irate;
    Eigen::Vector3d y_noisy = y_t;
    y_noisy += Eigen::Vector3d(std::sqrt(Rm(0, 0)) * rng_data.normal(),
                               std::sqrt(Rm(1, 1)) * rng_data.normal(),
                               std::sqrt(Rm(2, 2)) * rng_data.normal());
    y_obs[t] = y_noisy;

    if (std::abs(irate - p.i_lower) < 1e-12) ++bind_count;

    std::vector<double> row(S);
    for (int sp = 0; sp < S; ++sp) row[sp] = grid.P(s_prev, sp);
    s_prev = rng_data.categorical(row);
  }

  const double ll_exact = global_discrete_loglik_prefix(grid, d0, y_obs, Rm, T);
  std::cout << "Simulated NK dataset: T=" << T << ", ELB binds in " << bind_count << " periods\n";

  NkBenchmarkData out;
  out.grid = std::move(grid);
  out.y_obs = std::move(y_obs);
  out.alpha0 = std::move(d0);
  out.loglik_exact = ll_exact;
  out.bind_count = bind_count;
  return out;
}

struct AblationRow {
  std::string name;
  int horizon = 0;
  int omega_horizon = 0;
  int K = 0;
  AnchorMode anchor_mode = AnchorMode::Shared;
  models::PppfRegimeMode regime_mode = models::PppfRegimeMode::EndogenousByNode;
  PppfIMeasMode i_meas_mode = PppfIMeasMode::Censored;
  bool optimal_index_proposal = true;
  int T = 0;
  int N = 0;
  int R = 0;
  double mean_gap = 0.0;
  double sd_loglik = 0.0;
  double mean_ess = 0.0;
  double mean_runtime_ms = 0.0;
};

std::string to_string(AnchorMode m) {
  switch (m) {
    case AnchorMode::Shared:
      return "shared";
    case AnchorMode::Clustered:
      return "clustered";
    case AnchorMode::PerParticle:
      return "per_particle";
  }
  return "unknown";
}
std::string to_string(PppfIMeasMode m) { return (m == PppfIMeasMode::Censored) ? "censored" : "linear"; }
std::string to_string(models::PppfRegimeMode m) {
  switch (m) {
    case models::PppfRegimeMode::EndogenousByNode:
      return "endogenous";
    case models::PppfRegimeMode::FixedBind0FromMean:
      return "fixed_bind0";
    case models::PppfRegimeMode::MixtureBind0:
      return "mixture_bind0";
  }
  return "unknown";
}

int implied_K(int omega_horizon, int horizon, double omega_var, models::PppfOmegaMode omega_mode,
              models::PppfRegimeMode regime_mode) {
  const int L = std::min(omega_horizon, horizon);
  const int d = (omega_mode == models::PppfOmegaMode::RAndNu) ? (2 * L) : L;
  const int Kut = (d == 0 || omega_var == 0.0) ? 1 : (2 * d + 1);
  if (regime_mode == models::PppfRegimeMode::MixtureBind0 && Kut > 1) return 2 * Kut;
  return Kut;
}

void write_nk_ablation_csv(const std::filesystem::path& path, const std::vector<AblationRow>& rows) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to open: " + path.string());
  util::write_csv_header(out, {"case", "T", "N", "R", "horizon", "omega_horizon", "K", "anchor", "regime",
                               "i_meas", "optimal_index_proposal", "mean_gap_vs_exact", "sd_loglik",
                               "mean_ess", "mean_runtime_ms"});
  for (const auto& r : rows) {
    util::write_csv_row(out, r.name, r.T, r.N, r.R, r.horizon, r.omega_horizon, r.K, to_string(r.anchor_mode),
                        to_string(r.regime_mode), to_string(r.i_meas_mode), r.optimal_index_proposal ? 1 : 0,
                        r.mean_gap, r.sd_loglik, r.mean_ess, r.mean_runtime_ms);
  }
}

void write_nk_ablation_table_tex(const std::filesystem::path& path, const std::vector<AblationRow>& rows) {
  std::ostringstream oss;
  const int T0 = rows.empty() ? 0 : rows.front().T;
  const int N0 = rows.empty() ? 0 : rows.front().N;
  const int R0 = rows.empty() ? 0 : rows.front().R;
  oss << "\\begin{tabular}{lrrrrrr}\n";
  oss << "\\toprule\n";
  oss << "Case & $H$ & $\\omega$ horizon & $K$ & Gap & SD & Mean ESS \\\\\n";
  oss << "\\midrule\n";
  oss.setf(std::ios::fixed);
  for (const auto& r : rows) {
    std::string name = r.name;
    if (r.T != T0 || r.N != N0 || r.R != R0) name += "$^{\\dagger}$";
    oss << name << " & " << r.horizon << " & " << r.omega_horizon << " & " << r.K << " & "
        << std::setprecision(1) << r.mean_gap << " & " << std::setprecision(1) << r.sd_loglik << " & "
        << std::setprecision(1) << r.mean_ess << " \\\\\n";
  }
  oss << "\\bottomrule\n";
  oss << "\\end{tabular}\n";
  util::write_text_file(path, oss.str());
}

void run_nk_elb_ablations(const models::NkParams& p_base, int T, int N, int R, const Eigen::Matrix3d& Rm,
                          std::uint64_t seed_data, const std::filesystem::path& out_dir,
                          const std::filesystem::path& paper_table_path) {
  util::ensure_dir(out_dir);

  const NkBenchmarkData bench = make_nk_elb_benchmark_data(p_base, T, Rm, seed_data);
  std::map<int, double> ll_exact_cache;
  ll_exact_cache[T] = bench.loglik_exact;

  const Eigen::Vector4d mean0 = Eigen::Vector4d::Zero();
  Eigen::Matrix4d cov0 = Eigen::Matrix4d::Zero();
  cov0(0, 0) = 0.02 * 0.02;
  cov0(1, 1) = 0.02 * 0.02;
  cov0(2, 2) = 0.05 * 0.05;
  cov0(3, 3) = 0.05 * 0.05;

  struct CaseSpec {
    std::string name;
    int horizon;
    PppfOptions opt;
    int T_override = -1;
    int N_override = -1;
    int R_override = -1;
  };

  std::vector<CaseSpec> cases;
  CaseSpec base;
  base.name = "Baseline";
  base.horizon = 20;
  base.opt.omega_horizon = 1;
  base.opt.omega_var = 1.0;
  base.opt.anchor_mode = AnchorMode::Clustered;
  base.opt.regime_mode = models::PppfRegimeMode::EndogenousByNode;
  base.opt.i_meas_mode = PppfIMeasMode::Censored;
  base.opt.optimal_index_proposal = true;
  cases.push_back(base);

  {
    CaseSpec c = base;
    c.name = "Shared anchor";
    c.opt.anchor_mode = AnchorMode::Shared;
    cases.push_back(c);
  }

  {
    CaseSpec c = base;
    c.name = "CE ($K{=}1$)";
    c.opt.omega_var = 0.0;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "$\\omega$ horizon 2";
    c.opt.omega_horizon = 2;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "$\\omega$ horizon 3";
    c.opt.omega_horizon = 3;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "$H{=}10$";
    c.horizon = 10;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "$H{=}40$";
    c.horizon = 40;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "Fixed bind0";
    c.opt.regime_mode = models::PppfRegimeMode::FixedBind0FromMean;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "Bind0 mixture";
    c.opt.regime_mode = models::PppfRegimeMode::MixtureBind0;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "Linear $i$";
    c.opt.i_meas_mode = PppfIMeasMode::Linear;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "No aux proposal";
    c.opt.optimal_index_proposal = false;
    cases.push_back(c);
  }
  {
    CaseSpec c = base;
    c.name = "Per-particle anchor";
    c.opt.anchor_mode = AnchorMode::PerParticle;
    c.T_override = 50;
    c.N_override = 4;
    c.R_override = 2;
    cases.push_back(c);
  }

  std::vector<AblationRow> rows;
  rows.reserve(cases.size());

  for (const auto& cs : cases) {
    models::NkParams p = p_base;
    p.horizon = cs.horizon;

    const int Tc = (cs.T_override > 0) ? cs.T_override : T;
    const int Nc = (cs.N_override > 0) ? cs.N_override : N;
    const int Rc = (cs.R_override > 0) ? cs.R_override : R;

    if (ll_exact_cache.find(Tc) == ll_exact_cache.end()) {
      ll_exact_cache[Tc] = global_discrete_loglik_prefix(bench.grid, bench.alpha0, bench.y_obs, Rm, Tc);
    }
    const double ll_exact = ll_exact_cache.at(Tc);
    std::vector<Eigen::VectorXd> y_prefix(bench.y_obs.begin(), bench.y_obs.begin() + Tc);

    std::vector<double> ll;
    std::vector<double> rt;
    ll.reserve(Rc);
    rt.reserve(Rc);
    double mean_ess = 0.0;

    for (int rep = 0; rep < Rc; ++rep) {
      const std::uint64_t seed_base = 930000ULL + static_cast<std::uint64_t>(rep) * 1000ULL;
      const RunResult rr = run_pppf_mixture_rbpf(p, Nc, mean0, cov0, y_prefix, Rm, seed_base + 17, cs.opt);
      ll.push_back(rr.loglik);
      rt.push_back(rr.runtime_ms);
      if (rep == 0) {
        for (double e : rr.ess) mean_ess += e;
        mean_ess /= static_cast<double>(rr.ess.size());
      }
    }

    AblationRow row;
    row.name = cs.name;
    row.horizon = cs.horizon;
    row.omega_horizon = cs.opt.omega_horizon;
    row.anchor_mode = cs.opt.anchor_mode;
    row.regime_mode = cs.opt.regime_mode;
    row.i_meas_mode = cs.opt.i_meas_mode;
    row.optimal_index_proposal = cs.opt.optimal_index_proposal;
    row.T = Tc;
    row.N = Nc;
    row.R = Rc;
    row.K = implied_K(cs.opt.omega_horizon, cs.horizon, cs.opt.omega_var, cs.opt.omega_mode,
                      cs.opt.regime_mode);
    row.mean_gap = util::mean(ll) - ll_exact;
    row.sd_loglik = util::stdev(ll);
    row.mean_ess = mean_ess;
    row.mean_runtime_ms = util::mean(rt);
    rows.push_back(row);

    std::cout << "Ablation " << cs.name << ": gap=" << row.mean_gap << ", sd=" << row.sd_loglik
              << ", ess=" << row.mean_ess << ", time(ms)=" << row.mean_runtime_ms << std::endl;
  }

  write_nk_ablation_csv(out_dir / "nk_ablation.csv", rows);
  if (!paper_table_path.empty()) write_nk_ablation_table_tex(paper_table_path, rows);
}

}  // namespace

struct CliOptions {
  std::string mode = "baseline";
  int T = 250;
  int N = 256;
  int R = 30;
  int horizon = 20;
  int omega_horizon = 1;
  AnchorMode anchor_mode = AnchorMode::Clustered;
  std::uint64_t seed_data = 20260113ULL;
  std::uint64_t seed_sanity = 20260218ULL;
  std::filesystem::path out_dir = "output/nk";
  std::filesystem::path out_dir_abl = "output/nk_ablation";
  std::filesystem::path out_dir_sanity = "output/nk_sanity_no_elb";
  std::filesystem::path paper_table = {};
  bool run_sanity = true;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (a == "--mode") {
      opt.mode = need(a);
    } else if (a == "--T") {
      opt.T = std::stoi(need(a));
    } else if (a == "--N") {
      opt.N = std::stoi(need(a));
    } else if (a == "--R") {
      opt.R = std::stoi(need(a));
    } else if (a == "--horizon") {
      opt.horizon = std::stoi(need(a));
    } else if (a == "--omega_horizon") {
      opt.omega_horizon = std::stoi(need(a));
    } else if (a == "--anchor") {
      const std::string v = need(a);
      if (v == "shared") {
        opt.anchor_mode = AnchorMode::Shared;
      } else if (v == "clustered") {
        opt.anchor_mode = AnchorMode::Clustered;
      } else if (v == "per_particle") {
        opt.anchor_mode = AnchorMode::PerParticle;
      } else {
        throw std::invalid_argument("unknown anchor mode: " + v);
      }
    } else if (a == "--seed_data") {
      opt.seed_data = static_cast<std::uint64_t>(std::stoull(need(a)));
    } else if (a == "--seed_sanity") {
      opt.seed_sanity = static_cast<std::uint64_t>(std::stoull(need(a)));
    } else if (a == "--out_dir") {
      opt.out_dir = need(a);
    } else if (a == "--out_dir_abl") {
      opt.out_dir_abl = need(a);
    } else if (a == "--out_dir_sanity") {
      opt.out_dir_sanity = need(a);
    } else if (a == "--paper_table") {
      opt.paper_table = need(a);
    } else if (a == "--no_sanity") {
      opt.run_sanity = false;
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: nk_compare [--mode baseline|ablations] [--T int] [--N int] [--R int]\\n";
      std::cout << "                 [--horizon int] [--omega_horizon int] [--anchor shared|clustered|per_particle] [--out_dir path]\\n";
      std::cout << "                 [--out_dir_abl path] [--paper_table path]\\n";
      std::cout << "                 [--seed_data uint64] [--seed_sanity uint64] [--out_dir_sanity path] [--no_sanity]\\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown flag: " + a);
    }
  }
  return opt;
}

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);

    models::NkParams p;
    p.beta = 0.99;
    p.sigma = 1.0;
    p.kappa = 0.1;
    p.phi_pi = 1.5;
    p.phi_x = 0.25;
    p.i_ss = 0.04;
    p.i_lower = 0.0;
    p.r_ss = p.i_ss;
    p.rho_r = 0.95;
    p.sigma_r = 0.01;
    p.rho_nu = 0.8;
    p.sigma_nu = 0.005;
    p.horizon = cli.horizon;

    const double meas_sd_pi = 0.005;
    const double meas_sd_x = 0.005;
    const double meas_sd_i = 0.005;

    Eigen::Matrix3d Rm = Eigen::Matrix3d::Zero();
    Rm(0, 0) = meas_sd_pi * meas_sd_pi;
    Rm(1, 1) = meas_sd_x * meas_sd_x;
    Rm(2, 2) = meas_sd_i * meas_sd_i;

    if (cli.mode == "baseline") {
      std::cout << "\n=== NK-ELB benchmark (ELB may bind) ===\n";
      NkExperimentConfig cfg1;
      cfg1.p = p;
      cfg1.out_dir = cli.out_dir;
      cfg1.seed_data = cli.seed_data;
      cfg1.shock_eps_r_irf = -2.0;
      cfg1.omega_horizon = cli.omega_horizon;
      cfg1.anchor_mode = cli.anchor_mode;
      run_nk_experiment(cfg1, cli.T, cli.N, cli.R, Rm);

      if (cli.run_sanity) {
        std::cout << "\n=== NK sanity check (no ELB; linear-Gaussian; all methods should coincide) ===\n";
        run_nk_no_elb_continuous_sanity(p, cli.out_dir_sanity, cli.T, 2048, 10, Rm, cli.seed_sanity,
                                        /*omega_horizon=*/cfg1.omega_horizon);
      }
    } else if (cli.mode == "ablations") {
      std::cout << "\n=== NK-ELB ablation suite ===\n";
      run_nk_elb_ablations(p, cli.T, cli.N, cli.R, Rm, cli.seed_data, cli.out_dir_abl, cli.paper_table);
    } else {
      throw std::invalid_argument("unknown mode: " + cli.mode);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "nk_compare error: " << e.what() << "\n";
    return 1;
  }
}
