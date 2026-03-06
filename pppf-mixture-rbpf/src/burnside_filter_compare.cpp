#include "filters/bootstrap_pf.hpp"
#include "models/burnside.hpp"
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
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  int T = 150;
  int N = 512;
  int R = 20;
  int grid_size = 401;
  double grid_width_sd = 6.0;
  double meas_sd = 10.0;
  std::uint64_t seed_data = 20260306ULL;
  std::filesystem::path out_dir = "output/burnside_filter";
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
    } else if (a == "--grid_size") {
      opt.grid_size = std::stoi(need(a));
    } else if (a == "--grid_width_sd") {
      opt.grid_width_sd = std::stod(need(a));
    } else if (a == "--meas_sd") {
      opt.meas_sd = std::stod(need(a));
    } else if (a == "--seed_data") {
      opt.seed_data = static_cast<std::uint64_t>(std::stoull(need(a)));
    } else if (a == "--out_dir") {
      opt.out_dir = need(a);
    } else if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: burnside_filter_compare [--T int] [--N int] [--R int] [--grid_size int] "
             "[--grid_width_sd double] [--meas_sd double] [--seed_data uint64] [--out_dir path]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown flag: " + a);
    }
  }
  if (opt.T <= 0) throw std::invalid_argument("T must be positive");
  if (opt.N <= 0) throw std::invalid_argument("N must be positive");
  if (opt.R <= 0) throw std::invalid_argument("R must be positive");
  if (opt.grid_size < 21 || opt.grid_size % 2 == 0) {
    throw std::invalid_argument("grid_size must be an odd integer >= 21");
  }
  if (!(opt.grid_width_sd > 0.0)) throw std::invalid_argument("grid_width_sd must be positive");
  if (!(opt.meas_sd > 0.0)) throw std::invalid_argument("meas_sd must be positive");
  return opt;
}

double normal_cdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_bin_prob(double mean, double sd, double lo, double hi) {
  if (!(sd > 0.0)) throw std::invalid_argument("normal_bin_prob: sd<=0");
  if (std::isinf(lo) && lo < 0.0) {
    return normal_cdf((hi - mean) / sd);
  }
  if (std::isinf(hi) && hi > 0.0) {
    return 1.0 - normal_cdf((lo - mean) / sd);
  }
  return normal_cdf((hi - mean) / sd) - normal_cdf((lo - mean) / sd);
}

double stationary_sd(const models::BurnsideParams& p) {
  return p.sigma / std::sqrt(1.0 - p.rho * p.rho);
}

std::vector<double> make_grid(const models::BurnsideParams& p, int grid_size, double grid_width_sd) {
  const double sd0 = stationary_sd(p);
  const double lo = p.mu - grid_width_sd * sd0;
  const double hi = p.mu + grid_width_sd * sd0;
  std::vector<double> grid(grid_size);
  for (int i = 0; i < grid_size; ++i) {
    const double frac = static_cast<double>(i) / static_cast<double>(grid_size - 1);
    grid[i] = lo + frac * (hi - lo);
  }
  return grid;
}

std::vector<double> make_edges(const std::vector<double>& grid) {
  const int M = static_cast<int>(grid.size());
  std::vector<double> edges(M + 1, 0.0);
  edges.front() = -std::numeric_limits<double>::infinity();
  edges.back() = std::numeric_limits<double>::infinity();
  for (int j = 1; j < M; ++j) edges[j] = 0.5 * (grid[j - 1] + grid[j]);
  return edges;
}

Eigen::MatrixXd transition_matrix(const models::BurnsideParams& p, const std::vector<double>& grid,
                                  const std::vector<double>& edges) {
  const int M = static_cast<int>(grid.size());
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(M, M);
  for (int i = 0; i < M; ++i) {
    const double mean = (1.0 - p.rho) * p.mu + p.rho * grid[i];
    for (int j = 0; j < M; ++j) {
      P(i, j) = normal_bin_prob(mean, p.sigma, edges[j], edges[j + 1]);
    }
    const double rowsum = P.row(i).sum();
    if (!(rowsum > 0.0)) throw std::runtime_error("transition_matrix: zero row mass");
    P.row(i) /= rowsum;
  }
  return P;
}

Eigen::VectorXd stationary_weights(const models::BurnsideParams& p, const std::vector<double>& edges) {
  const int M = static_cast<int>(edges.size()) - 1;
  Eigen::VectorXd w(M);
  const double sd0 = stationary_sd(p);
  for (int j = 0; j < M; ++j) {
    w(j) = normal_bin_prob(p.mu, sd0, edges[j], edges[j + 1]);
  }
  w /= w.sum();
  return w;
}

double log_norm_pdf_scalar(double y, double mean, double sd) {
  const double z = (y - mean) / sd;
  return -0.5 * std::log(2.0 * M_PI) - std::log(sd) - 0.5 * z * z;
}

double interp_value(const std::vector<double>& grid, const std::vector<double>& vals, double x);

double sample_normal(double mean, double sd, util::Rng& rng) { return mean + sd * rng.normal(); }

struct ScalarMixtureProposal {
  std::vector<double> means;
  std::vector<double> probs;
  double comp_sd = 1.0;
};

double log_mixture_normal_pdf(double x, const ScalarMixtureProposal& proposal) {
  Eigen::VectorXd log_terms(static_cast<int>(proposal.means.size()));
  for (int j = 0; j < static_cast<int>(proposal.means.size()); ++j) {
    log_terms(j) = std::log(proposal.probs[j]) + log_norm_pdf_scalar(x, proposal.means[j], proposal.comp_sd);
  }
  return util::log_sum_exp(log_terms);
}

ScalarMixtureProposal build_scalar_posterior_mixture_proposal(
    double prior_mean, double prior_sd, double y_t, double meas_sd, const std::vector<double>& value_grid,
    const std::vector<double>& grid, int num_nodes = 21, double width_sd = 5.5, double jitter_frac = 0.75) {
  if (num_nodes < 5 || num_nodes % 2 == 0) throw std::invalid_argument("proposal nodes must be odd and >= 5");
  const double z_lo = -width_sd;
  const double z_hi = width_sd;
  const double dz = (z_hi - z_lo) / static_cast<double>(num_nodes - 1);
  Eigen::VectorXd log_terms(num_nodes);
  std::vector<double> means(num_nodes, 0.0);
  for (int j = 0; j < num_nodes; ++j) {
    const double z = z_lo + static_cast<double>(j) * dz;
    const double xj = prior_mean + prior_sd * z;
    const double trap_w = (j == 0 || j == num_nodes - 1) ? 0.5 * dz : dz;
    const double mean_y = interp_value(grid, value_grid, xj);
    means[j] = xj;
    log_terms(j) = std::log(trap_w) - 0.5 * z * z - 0.5 * std::log(2.0 * M_PI) +
                   log_norm_pdf_scalar(y_t, mean_y, meas_sd);
  }
  const double log_norm = util::log_sum_exp(log_terms);
  ScalarMixtureProposal proposal;
  proposal.means = means;
  proposal.probs.resize(num_nodes);
  for (int j = 0; j < num_nodes; ++j) proposal.probs[j] = std::exp(log_terms(j) - log_norm);
  proposal.comp_sd = std::max(1e-6, jitter_frac * prior_sd * dz);
  return proposal;
}

filters::PfDiagnostics copf_scalar_burnside(
    int N, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0_sqrt, const std::vector<Eigen::VectorXd>& y,
    const models::BurnsideParams& p, double meas_sd, const std::vector<double>& value_grid,
    const std::vector<double>& grid, util::Rng& rng, double ess_frac = 0.5) {
  if (N <= 0) throw std::invalid_argument("copf_scalar_burnside: N<=0");
  if (x0.size() != 1 || P0_sqrt.rows() != 1 || P0_sqrt.cols() != 1) {
    throw std::invalid_argument("copf_scalar_burnside: scalar state required");
  }

  std::vector<double> particles(N, x0(0));
  for (int i = 0; i < N; ++i) particles[i] = x0(0) + P0_sqrt(0, 0) * rng.normal();

  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
  filters::PfDiagnostics diag;
  diag.ess.reserve(y.size());

  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd logw_new(N);
    std::vector<double> new_particles(N, 0.0);
    for (int i = 0; i < N; ++i) {
      const double prior_mean = (1.0 - p.rho) * p.mu + p.rho * particles[i];
      const ScalarMixtureProposal proposal = build_scalar_posterior_mixture_proposal(
          prior_mean, p.sigma, y[t](0), meas_sd, value_grid, grid);
      const int draw_idx = rng.categorical(proposal.probs);
      const double x_draw = sample_normal(proposal.means[draw_idx], proposal.comp_sd, rng);
      const double mean_y = interp_value(grid, value_grid, x_draw);
      const double log_obs = log_norm_pdf_scalar(y[t](0), mean_y, meas_sd);
      const double log_prior = log_norm_pdf_scalar(x_draw, prior_mean, p.sigma);
      const double log_prop = log_mixture_normal_pdf(x_draw, proposal);
      logw_new(i) = logw(i) + log_obs + log_prior - log_prop;
      new_particles[i] = x_draw;
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
      std::vector<double> resampled(N, 0.0);
      for (int i = 0; i < N; ++i) resampled[i] = particles[idx[i]];
      particles.swap(resampled);
      logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
    } else {
      logw = logw_new;
    }
  }

  return diag;
}

std::vector<double> tabulate_values(
    const std::vector<double>& grid,
    const std::function<double(const models::BurnsideParams&, double)>& value_fn,
    const models::BurnsideParams& p) {
  std::vector<double> vals(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) vals[i] = value_fn(p, grid[i]);
  return vals;
}

double interp_value(const std::vector<double>& grid, const std::vector<double>& vals, double x) {
  const int M = static_cast<int>(grid.size());
  if (static_cast<int>(vals.size()) != M) throw std::invalid_argument("interp_value: grid/value size mismatch");
  if (x <= grid.front()) return vals.front();
  if (x >= grid.back()) return vals.back();
  const auto it = std::lower_bound(grid.begin(), grid.end(), x);
  const int hi = static_cast<int>(it - grid.begin());
  const int lo = hi - 1;
  const double x_lo = grid[lo];
  const double x_hi = grid[hi];
  const double w_hi = (x - x_lo) / (x_hi - x_lo);
  return (1.0 - w_hi) * vals[lo] + w_hi * vals[hi];
}

std::vector<Eigen::VectorXd> simulate_data(const models::BurnsideParams& p, int T, double meas_sd,
                                           std::uint64_t seed_data, std::vector<double>* x_out) {
  util::Rng rng(seed_data);
  const double sd0 = stationary_sd(p);
  double x = p.mu + sd0 * rng.normal();
  std::vector<Eigen::VectorXd> y(T);
  x_out->resize(T);
  for (int t = 0; t < T; ++t) {
    x = (1.0 - p.rho) * p.mu + p.rho * x + p.sigma * rng.normal();
    (*x_out)[t] = x;
    const double y_t = models::burnside_v_exact(p, x) + meas_sd * rng.normal();
    y[t] = Eigen::VectorXd::Constant(1, y_t);
  }
  return y;
}

double grid_loglik(const models::BurnsideParams& p, const std::vector<double>& grid, const Eigen::MatrixXd& P,
                   const Eigen::VectorXd& alpha0, const std::vector<Eigen::VectorXd>& y, double meas_sd,
                   const std::vector<double>& value_grid) {
  const int M = static_cast<int>(grid.size());
  Eigen::VectorXd alpha = alpha0;
  double ll = 0.0;
  for (int t = 0; t < static_cast<int>(y.size()); ++t) {
    Eigen::VectorXd alpha_pred = P.transpose() * alpha;
    Eigen::VectorXd loglik_j(M);
    for (int j = 0; j < M; ++j) {
      loglik_j(j) = log_norm_pdf_scalar(y[t](0), value_grid[j], meas_sd);
    }
    const Eigen::VectorXd obs = loglik_j.array().exp().matrix();
    const double z = alpha_pred.dot(obs);
    ll += std::log(z);
    alpha = alpha_pred.cwiseProduct(obs) / z;
  }
  return ll;
}

void write_summary_json(const std::filesystem::path& path,
                        const std::map<std::string, std::vector<double>>& loglik_by_method,
                        const std::map<std::string, std::vector<double>>& rt_by_method,
                        const std::map<std::string, double>& mean_ess_by_method, int T, int N, int R,
                        int grid_size, double grid_width_sd, double meas_sd, std::uint64_t seed_data) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"config\": {\n";
  oss << "    \"T\": " << T << ",\n";
  oss << "    \"N\": " << N << ",\n";
  oss << "    \"R\": " << R << ",\n";
  oss << "    \"grid_size\": " << grid_size << ",\n";
  oss << "    \"grid_width_sd\": " << grid_width_sd << ",\n";
  oss << "    \"meas_sd\": " << meas_sd << ",\n";
  oss << "    \"seed_data\": " << seed_data << "\n";
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
    const auto it_ess = mean_ess_by_method.find(name);
    const double mean_ess = (it_ess == mean_ess_by_method.end()) ? -1.0 : it_ess->second;
    oss << "      \"mean_ess\": " << mean_ess << "\n";
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

    models::BurnsideParams p;
    p.beta = 0.99;
    p.gamma = 2.0;
    p.rho = 0.95;
    p.mu = 0.0;
    p.sigma = 0.006;

    util::ensure_dir(cli.out_dir);

    std::vector<double> x_sim;
    const std::vector<Eigen::VectorXd> y =
        simulate_data(p, cli.T, cli.meas_sd, cli.seed_data, &x_sim);

    std::ofstream data_out(cli.out_dir / "sim_data.csv");
    if (!data_out) throw std::runtime_error("failed to open sim_data.csv");
    util::write_csv_header(data_out, {"t", "x", "y"});
    for (int t = 0; t < cli.T; ++t) util::write_csv_row(data_out, t, x_sim[t], y[t](0));

    const std::vector<double> grid = make_grid(p, cli.grid_size, cli.grid_width_sd);
    const std::vector<double> edges = make_edges(grid);
    const Eigen::MatrixXd P = transition_matrix(p, grid, edges);
    const Eigen::VectorXd alpha0 = stationary_weights(p, edges);

    const std::map<std::string, std::function<double(const models::BurnsideParams&, double)>> value_fns = {
        {"grid_exact", [](const models::BurnsideParams& pp, double x) { return models::burnside_v_exact(pp, x); }},
        {"grid_ce", [](const models::BurnsideParams& pp, double x) { return models::burnside_v_ce(pp, x); }},
        {"grid_ut", [](const models::BurnsideParams& pp, double x) { return models::burnside_v_ut_one_step(pp, x); }},
    };
    std::map<std::string, std::vector<double>> value_grid_by_method;
    for (const auto& kv : value_fns) value_grid_by_method[kv.first] = tabulate_values(grid, kv.second, p);

    std::map<std::string, std::vector<double>> loglik_by_method;
    std::map<std::string, std::vector<double>> rt_by_method;
    std::map<std::string, double> mean_ess_by_method;

    std::ofstream ll_out(cli.out_dir / "loglik_repeats.csv");
    if (!ll_out) throw std::runtime_error("failed to open loglik_repeats.csv");
    util::write_csv_header(ll_out, {"method", "rep", "loglik", "runtime_ms"});

    std::ofstream ess_out(cli.out_dir / "ess.csv");
    if (!ess_out) throw std::runtime_error("failed to open ess.csv");
    util::write_csv_header(ess_out, {"method", "rep", "t", "ess"});

    for (const auto& kv : value_grid_by_method) {
      const std::string& name = kv.first;
      const std::vector<double>& value_grid = kv.second;
      util::Timer timer;
      const double ll = grid_loglik(p, grid, P, alpha0, y, cli.meas_sd, value_grid);
      const double rt = timer.elapsed_ms();
      loglik_by_method[name].push_back(ll);
      rt_by_method[name].push_back(rt);
      mean_ess_by_method[name] = -1.0;
      util::write_csv_row(ll_out, name, 0, ll, rt);
    }

    const Eigen::VectorXd mean0 = Eigen::VectorXd::Constant(1, p.mu);
    Eigen::MatrixXd P0_sqrt(1, 1);
    P0_sqrt(0, 0) = stationary_sd(p);

    const auto transition = [&](Eigen::VectorXd* state, util::Rng& rng, int /*t*/) {
      (*state)(0) = (1.0 - p.rho) * p.mu + p.rho * (*state)(0) + p.sigma * rng.normal();
    };

    for (const auto& kv : value_grid_by_method) {
      const std::string& grid_name = kv.first;
      const std::vector<double>& value_grid = kv.second;
      std::string pf_name = grid_name;
      pf_name.replace(0, 4, "pf");
      std::string copf_name = grid_name;
      copf_name.replace(0, 4, "copf");
      std::vector<double> ess_means;
      std::vector<double> copf_ess_means;
      for (int rep = 0; rep < cli.R; ++rep) {
        util::Rng rng_pf(930000ULL + static_cast<std::uint64_t>(rep) * 97ULL);
        const auto log_obs = [&](const Eigen::VectorXd& state, const Eigen::VectorXd& y_t, int /*t*/) {
          const double mean_y = interp_value(grid, value_grid, state(0));
          return log_norm_pdf_scalar(y_t(0), mean_y, cli.meas_sd);
        };
        util::Timer timer;
        const filters::PfDiagnostics diag =
            filters::bootstrap_pf(cli.N, mean0, P0_sqrt, y, transition, log_obs, rng_pf, 0.5);
        const double rt = timer.elapsed_ms();
        loglik_by_method[pf_name].push_back(diag.loglik);
        rt_by_method[pf_name].push_back(rt);
        util::write_csv_row(ll_out, pf_name, rep, diag.loglik, rt);
        double ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(diag.ess.size()); ++t) {
          util::write_csv_row(ess_out, pf_name, rep, t, diag.ess[t]);
          ess_mean += diag.ess[t];
        }
        ess_mean /= static_cast<double>(diag.ess.size());
        ess_means.push_back(ess_mean);

        util::Rng rng_copf(1950000ULL + static_cast<std::uint64_t>(rep) * 131ULL);
        util::Timer timer_copf;
        const filters::PfDiagnostics diag_copf =
            copf_scalar_burnside(cli.N, mean0, P0_sqrt, y, p, cli.meas_sd, value_grid, grid, rng_copf, 0.5);
        const double rt_copf = timer_copf.elapsed_ms();
        loglik_by_method[copf_name].push_back(diag_copf.loglik);
        rt_by_method[copf_name].push_back(rt_copf);
        util::write_csv_row(ll_out, copf_name, rep, diag_copf.loglik, rt_copf);
        double copf_ess_mean = 0.0;
        for (int t = 0; t < static_cast<int>(diag_copf.ess.size()); ++t) {
          util::write_csv_row(ess_out, copf_name, rep, t, diag_copf.ess[t]);
          copf_ess_mean += diag_copf.ess[t];
        }
        copf_ess_mean /= static_cast<double>(diag_copf.ess.size());
        copf_ess_means.push_back(copf_ess_mean);
      }
      mean_ess_by_method[pf_name] = util::mean(ess_means);
      mean_ess_by_method[copf_name] = util::mean(copf_ess_means);
    }

    write_summary_json(cli.out_dir / "summary.json", loglik_by_method, rt_by_method, mean_ess_by_method,
                       cli.T, cli.N, cli.R, cli.grid_size, cli.grid_width_sd, cli.meas_sd, cli.seed_data);

    std::cout << "Wrote " << (cli.out_dir / "sim_data.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "loglik_repeats.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "ess.csv").string() << "\n";
    std::cout << "Wrote " << (cli.out_dir / "summary.json").string() << "\n";
    std::cout << "Mean loglik gaps vs grid_exact:\n";
    const double ll_exact = util::mean(loglik_by_method.at("grid_exact"));
    for (const auto& [name, vals] : loglik_by_method) {
      std::cout << "  " << name << ": " << util::mean(vals) - ll_exact << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "burnside_filter_compare error: " << e.what() << "\n";
    return 1;
  }
}
