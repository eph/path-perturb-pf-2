#include "filters/bootstrap_pf.hpp"
#include "filters/rbpf.hpp"
#include "models/nk_elb.hpp"
#include "models/nk_global.hpp"
#include "util/io.hpp"
#include "util/rng.hpp"
#include "util/stats.hpp"
#include "util/timer.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
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
                               const Eigen::Matrix3d& R, std::uint64_t seed) {
  util::Rng rng(seed);
  util::Timer timer;

  const auto mixture_builder = [&](int /*t*/, const Eigen::VectorXd& ref_prev) {
    const Eigen::Vector4d z_ref = ref_prev;
    return models::build_pppf_mixture(p, z_ref);
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

    // Censored measurement for i: i = max(i_lower, i_ss + phi_pi*pi + phi_x*x + nu) + noise.
    const Eigen::RowVector4d h_rule =
        (Eigen::RowVector4d() << p.phi_x, p.phi_pi, 0.0, 1.0).finished();
    ll += statespace::kalman_update_censored_lower(h_rule, p.i_ss, p.i_lower, R(2, 2), y_t(2), st);
    return ll;
  };

  const filters::RbpfDiagnostics diag =
      filters::rbpf(N, mean0, cov0, y, mixture_builder, obs_update, rng, 0.5, true);

  RunResult res;
  res.loglik = diag.loglik;
  res.runtime_ms = timer.elapsed_ms();
  res.ess = diag.ess;
  return res;
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

}  // namespace

int main() {
  try {
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
    p.horizon = 20;

    const int T = 250;
    const int N = 256;
    const int R = 30;

    const double meas_sd_pi = 0.005;
    const double meas_sd_x = 0.005;
    const double meas_sd_i = 0.005;

    Eigen::Matrix3d Rm = Eigen::Matrix3d::Zero();
    Rm(0, 0) = meas_sd_pi * meas_sd_pi;
    Rm(1, 1) = meas_sd_x * meas_sd_x;
    Rm(2, 2) = meas_sd_i * meas_sd_i;

    const std::filesystem::path out_dir = "output/nk";
    util::ensure_dir(out_dir);

    // Solve the global discrete-state benchmark (Tauchen grid) once; used for data + IRFs.
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
    const std::uint64_t seed_data = 20260113ULL;
    util::Rng rng_data(seed_data);

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
      y_noisy(0) += meas_sd_pi * rng_data.normal();
      y_noisy(1) += meas_sd_x * rng_data.normal();
      y_noisy(2) += meas_sd_i * rng_data.normal();
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

    for (const std::string& method : {"occbin_bootstrap_pf", "pppf_mixture_rbpf", "global_discrete_exact"}) {
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
        const RunResult rr =
            run_occbin_bootstrap_pf(p, N, mean0, cov0, y_obs, Rm, seed_base + 1);
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
        const RunResult rr =
            run_pppf_mixture_rbpf(p, N, mean0, cov0, y_obs, Rm, seed_base + 2);
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

    write_summary_json(out_dir / "summary.json", loglik_by_method, rt_by_method, mean_ess_by_method,
                       T, N, R);
    std::cout << "Wrote " << (out_dir / "summary.json").string() << "\n";

    // IRFs near the ELB: a negative natural-rate impulse at t=0.
    // We report:
    // - OccBin deterministic (perfect-foresight) IRF (future shocks = 0)
    // - PPPF-mixture mean IRF (integrates future shocks each step via sigma points)
    // - Global policy-function IRF: E[y_t] under the global stochastic solution (integrating future shocks)
    const int Tirf = 40;
    const double shock_eps_r = -2.0;

    std::vector<double> irf_x_occ(Tirf), irf_pi_occ(Tirf), irf_i_occ(Tirf);
    std::vector<double> irf_x_pppf(Tirf), irf_pi_pppf(Tirf), irf_i_pppf(Tirf);
    std::vector<double> irf_x_glob(Tirf), irf_pi_glob(Tirf), irf_i_glob(Tirf);

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

    // PPPF-mixture mean IRF: propagate mean+cov under the time-varying mixture kernel built at the current mean.
    Eigen::Vector4d z_mean = Eigen::Vector4d::Zero();
    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    for (int t = 0; t < Tirf; ++t) {
      const double eps_mean = (t == 0) ? shock_eps_r : 0.0;
      const double eps_var = 1.0;
      const auto mix = models::build_pppf_mixture(p, z_mean, 0.0, 1.0, eps_mean, eps_var);

      Eigen::Vector4d z_next = Eigen::Vector4d::Zero();
      Eigen::Matrix4d P_next = Eigen::Matrix4d::Zero();
      std::vector<Eigen::Vector4d> mks(mix.size());
      std::vector<Eigen::Matrix4d> Pks(mix.size());
      for (int k = 0; k < mix.size(); ++k) {
        const auto& tr = mix.components[k];
        mks[k] = tr.A * z_mean + tr.a;
        Pks[k] = tr.A * P * tr.A.transpose() + tr.Q;
        z_next += mix.weights[k] * mks[k];
      }
      for (int k = 0; k < mix.size(); ++k) {
        const Eigen::Vector4d dm = mks[k] - z_next;
        P_next += mix.weights[k] * (Pks[k] + dm * dm.transpose());
      }
      z_mean = z_next;
      P = P_next;

      irf_x_pppf[t] = z_mean(0);
      irf_pi_pppf[t] = z_mean(1);
      // Approximate E[i_t] using a Gaussian approximation for i_rule.
      const Eigen::RowVector4d h_rule =
          (Eigen::RowVector4d() << p.phi_x, p.phi_pi, 0.0, 1.0).finished();
      const double mu_i = p.i_ss + (h_rule * z_mean)(0);
      const double var_i = (h_rule * P * h_rule.transpose())(0, 0);
      const double sd_i = std::sqrt(std::max(0.0, var_i));
      if (sd_i < 1e-12) {
        irf_i_pppf[t] = std::max(p.i_lower, mu_i);
      } else {
        const double z = (p.i_lower - mu_i) / sd_i;
        const double Phi = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
        const double phi = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
        irf_i_pppf[t] = p.i_lower * Phi + mu_i * (1.0 - Phi) + sd_i * phi;
      }
    }

    // Global stochastic solution on a discrete Markov chain (state-space, expectation-consistent).
    models::expected_irf_global(p, grid, shock_eps_r, Tirf, &irf_x_glob, &irf_pi_glob, &irf_i_glob);

    const std::filesystem::path irf_csv = out_dir / "irf.csv";
    std::ofstream irf_out(irf_csv);
    if (!irf_out) throw std::runtime_error("failed to open: " + irf_csv.string());
    util::write_csv_header(
        irf_out, {"t", "x_global", "pi_global", "i_global", "x_occbin", "pi_occbin", "i_occbin",
                  "x_pppf", "pi_pppf", "i_pppf"});
    for (int t = 0; t < Tirf; ++t) {
      util::write_csv_row(irf_out, t, irf_x_glob[t], irf_pi_glob[t], irf_i_glob[t], irf_x_occ[t],
                          irf_pi_occ[t], irf_i_occ[t], irf_x_pppf[t], irf_pi_pppf[t],
                          irf_i_pppf[t]);
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
              << "), PPPF=(" << rmse(irf_x_pppf, irf_x_glob) << ", "
              << rmse(irf_pi_pppf, irf_pi_glob) << ", " << rmse(irf_i_pppf, irf_i_glob) << ")\n";

    std::cout << "Wrote " << ll_csv.string() << "\n";
    std::cout << "Wrote " << ess_csv.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "nk_compare error: " << e.what() << "\n";
    return 1;
  }
}
