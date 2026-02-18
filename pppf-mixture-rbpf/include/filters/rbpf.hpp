#pragma once

#include "filters/resampling.hpp"
#include "statespace/gaussian_mixture.hpp"
#include "statespace/linear_gaussian.hpp"
#include "util/rng.hpp"
#include "util/stats.hpp"

#include <Eigen/Dense>

#include <functional>
#include <stdexcept>
#include <vector>

namespace filters {

struct RbpfParticle {
  statespace::KalmanState kf;
  int k = 0;  // last sampled mixture index
};

struct RbpfDiagnostics {
  double loglik = 0.0;
  std::vector<double> ess;
};

// RBPF over a discrete mixture index; continuous state is filtered with Kalman.
//
// ObsModelProvider: (int t, const statespace::KalmanState& pred, int k) -> (H, d, R)
struct ObsModel {
  Eigen::MatrixXd H;
  Eigen::VectorXd d;
  Eigen::MatrixXd R;
};

inline RbpfDiagnostics rbpf(int N, const Eigen::VectorXd& mean0, const Eigen::MatrixXd& cov0,
                            const std::vector<Eigen::VectorXd>& y,
                            const std::function<statespace::GaussianMixtureTransition(
                                int, const Eigen::VectorXd&)>&
                                mixture_builder,
                            const std::function<ObsModel(int, const statespace::KalmanState&, int)>&
                                obs_model,
                            util::Rng& rng, double ess_frac = 0.5) {
  if (N <= 0) throw std::invalid_argument("rbpf: N<=0");
  const int T = static_cast<int>(y.size());

  std::vector<RbpfParticle> particles(N);
  for (int i = 0; i < N; ++i) {
    particles[i].kf.mean = mean0;
    particles[i].kf.cov = cov0;
    particles[i].k = 0;
  }

  Eigen::VectorXd logw = Eigen::VectorXd::Constant(N, -std::log(static_cast<double>(N)));
  RbpfDiagnostics diag;
  diag.ess.reserve(T);

  for (int t = 0; t < T; ++t) {
    // Reference for localization: weighted mean of filtered means at t-1 (or initial at t=0).
    const Eigen::VectorXd w_norm = logw.array().exp().matrix();
    Eigen::VectorXd ref = Eigen::VectorXd::Zero(mean0.size());
    for (int i = 0; i < N; ++i) ref += w_norm(i) * particles[i].kf.mean;

    const statespace::GaussianMixtureTransition mix = mixture_builder(t, ref);
    mix.check();

    // Predict + update under sampled k per particle.
    Eigen::VectorXd logw_new(N);
    for (int i = 0; i < N; ++i) {
      const int k = rng.categorical(mix.weights);
      particles[i].k = k;
      const auto& tr = mix.components[k];

      statespace::KalmanState pred = particles[i].kf;
      statespace::kalman_predict(tr.A, tr.a, tr.Q, &pred);

      const ObsModel om = obs_model(t, pred, k);
      const double ll = statespace::kalman_update(om.H, om.d, om.R, y[t], &pred);

      particles[i].kf = pred;
      logw_new(i) = logw(i) + ll;
    }

    const double logZ = util::log_sum_exp(logw_new);
    diag.loglik += logZ;
    logw_new.array() -= logZ;

    const double ess_t = util::ess_from_logw(logw_new);
    diag.ess.push_back(ess_t);

    if (ess_t < ess_frac * static_cast<double>(N)) {
      const Eigen::VectorXd w_norm = logw_new.array().exp().matrix();
      const std::vector<int> idx = systematic_resample(w_norm, rng);
      std::vector<RbpfParticle> new_particles(N);
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
