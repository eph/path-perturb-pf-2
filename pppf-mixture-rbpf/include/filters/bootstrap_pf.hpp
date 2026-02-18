#pragma once

#include "filters/resampling.hpp"
#include "util/stats.hpp"

#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace filters {

struct PfDiagnostics {
  double loglik = 0.0;
  std::vector<double> ess;
};

// A simple bootstrap PF for a Markov state x_t and observation y_t.
//
// TransitionSampler:  void (Eigen::VectorXd* state, util::Rng& rng, int t)
// LogObsDensity:      double (const Eigen::VectorXd& state, const Eigen::VectorXd& y, int t)
inline PfDiagnostics bootstrap_pf(int N, const Eigen::VectorXd& x0,
                                  const Eigen::MatrixXd& P0_sqrt,  // Cholesky of initial cov
                                  const std::vector<Eigen::VectorXd>& y,
                                  const std::function<void(Eigen::VectorXd*, util::Rng&, int)>&
                                      transition_sample,
                                  const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, int)>&
                                      log_obs_density,
                                  util::Rng& rng, double ess_frac = 0.5) {
  const int T = static_cast<int>(y.size());
  const int n = static_cast<int>(x0.size());
  if (N <= 0) throw std::invalid_argument("bootstrap_pf: N<=0");
  if (P0_sqrt.rows() != n || P0_sqrt.cols() != n) throw std::invalid_argument("bootstrap_pf: P0 dim");

  std::vector<Eigen::VectorXd> particles(N, x0);
  for (int i = 0; i < N; ++i) {
    particles[i] = x0 + P0_sqrt * rng.normal_vec(n);
  }

  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
  PfDiagnostics diag;
  diag.ess.reserve(T);

  for (int t = 0; t < T; ++t) {
    // Propagate.
    for (int i = 0; i < N; ++i) transition_sample(&particles[i], rng, t);

    // Incremental weights.
    Eigen::VectorXd logw_new(N);
    for (int i = 0; i < N; ++i) {
      const double ll = log_obs_density(particles[i], y[t], t);
      logw_new(i) = logw(i) + ll;
    }

    const double logZ = util::log_sum_exp(logw_new);
    diag.loglik += logZ;
    logw_new.array() -= logZ;  // normalize to sum=1 in linear space

    const double ess_t = util::ess_from_logw(logw_new);
    diag.ess.push_back(ess_t);

    // Resample if needed.
    if (ess_t < ess_frac * static_cast<double>(N)) {
      const Eigen::VectorXd w_norm = logw_new.array().exp().matrix();
      const std::vector<int> idx = systematic_resample(w_norm, rng);
      std::vector<Eigen::VectorXd> new_particles(N);
      for (int i = 0; i < N; ++i) new_particles[i] = particles[idx[i]];
      particles.swap(new_particles);
      logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
    } else {
      logw = logw_new;
    }
  }

  return diag;
}

}  // namespace filters

