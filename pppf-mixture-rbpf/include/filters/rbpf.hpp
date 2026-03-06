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

// ObsUpdate: returns log p(y_t | pred, k, ...) and updates the KalmanState in place.
using ObsUpdate =
    std::function<double(int, const Eigen::VectorXd&, int, statespace::KalmanState*)>;
using AnchorGroupsFn = std::function<std::vector<int>(
    int, const std::vector<Eigen::VectorXd>&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;

inline RbpfDiagnostics rbpf(int N, const Eigen::VectorXd& mean0, const Eigen::MatrixXd& cov0,
                            const std::vector<Eigen::VectorXd>& y,
                            const std::function<statespace::GaussianMixtureTransition(
                                int, const Eigen::VectorXd&)>&
                                mixture_builder,
                            const ObsUpdate& obs_update, util::Rng& rng, double ess_frac = 0.5,
                            bool optimal_index_proposal = false,
                            bool per_particle_anchor = false, int num_anchor_groups = 1,
                            const AnchorGroupsFn& anchor_groups_fn = {}) {
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
    const Eigen::VectorXd w_norm = logw.array().exp().matrix();
    Eigen::VectorXd ref_shared = Eigen::VectorXd::Zero(mean0.size());
    for (int i = 0; i < N; ++i) ref_shared += w_norm(i) * particles[i].kf.mean;

    statespace::GaussianMixtureTransition mix_shared;
    std::vector<statespace::GaussianMixtureTransition> mix_groups;
    std::vector<char> has_group;
    std::vector<int> particle_group;
    if (!per_particle_anchor) {
      mix_shared = mixture_builder(t, ref_shared);
      mix_shared.check();
      if (anchor_groups_fn && num_anchor_groups > 1) {
        mix_groups.resize(num_anchor_groups);
        has_group.assign(num_anchor_groups, 0);
        std::vector<Eigen::VectorXd> particle_means(N);
        for (int i = 0; i < N; ++i) particle_means[i] = particles[i].kf.mean;
        particle_group = anchor_groups_fn(t, particle_means, w_norm, ref_shared);
        if (static_cast<int>(particle_group.size()) != N) {
          throw std::runtime_error("rbpf: anchor_groups_fn returned wrong number of assignments");
        }
        std::vector<double> group_mass(num_anchor_groups, 0.0);
        std::vector<Eigen::VectorXd> group_ref(
            num_anchor_groups, Eigen::VectorXd::Zero(mean0.size()));

        for (int i = 0; i < N; ++i) {
          int g = particle_group[i];
          if (g < 0) g = 0;
          if (g >= num_anchor_groups) g = num_anchor_groups - 1;
          particle_group[i] = g;
          group_mass[g] += w_norm(i);
          group_ref[g] += w_norm(i) * particles[i].kf.mean;
        }

        for (int g = 0; g < num_anchor_groups; ++g) {
          if (group_mass[g] <= 0.0) continue;
          mix_groups[g] = mixture_builder(t, group_ref[g] / group_mass[g]);
          mix_groups[g].check();
          has_group[g] = 1;
        }
      }
    }

    // Predict + update under sampled k per particle.
    Eigen::VectorXd logw_new(N);
    for (int i = 0; i < N; ++i) {
      const statespace::GaussianMixtureTransition* mix = &mix_shared;
      if (per_particle_anchor) {
        mix_shared = mixture_builder(t, particles[i].kf.mean);
        mix_shared.check();
      } else if (anchor_groups_fn && num_anchor_groups > 1) {
        const int g = particle_group[i];
        if (has_group[g]) mix = &mix_groups[g];
      }

      if (!optimal_index_proposal) {
        const int k = rng.categorical(mix->weights);
        particles[i].k = k;
        const auto& tr = mix->components[k];

        statespace::KalmanState pred = particles[i].kf;
        statespace::kalman_predict(tr.A, tr.a, tr.Q, &pred);

        const double ll = obs_update(t, y[t], k, &pred);

        particles[i].kf = pred;
        logw_new(i) = logw(i) + ll;
        continue;
      }

      const int K = mix->size();
      std::vector<statespace::KalmanState> preds(K);
      Eigen::VectorXd logwk(K);

      for (int k = 0; k < K; ++k) {
        preds[k] = particles[i].kf;
        const auto& tr = mix->components[k];
        statespace::kalman_predict(tr.A, tr.a, tr.Q, &preds[k]);
        const double ll = obs_update(t, y[t], k, &preds[k]);
        logwk(k) = std::log(mix->weights[k]) + ll;
      }

      const double log_norm = util::log_sum_exp(logwk);
      std::vector<double> probs(K);
      for (int k = 0; k < K; ++k) probs[k] = std::exp(logwk(k) - log_norm);

      const int k = rng.categorical(probs);
      particles[i].k = k;
      particles[i].kf = preds[k];
      logw_new(i) = logw(i) + log_norm;
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

// Backward-compatible overload: linear-Gaussian observation model provided as (H,d,R).
inline RbpfDiagnostics rbpf(int N, const Eigen::VectorXd& mean0, const Eigen::MatrixXd& cov0,
                            const std::vector<Eigen::VectorXd>& y,
                            const std::function<statespace::GaussianMixtureTransition(
                                int, const Eigen::VectorXd&)>&
                                mixture_builder,
                            const std::function<ObsModel(int, const statespace::KalmanState&, int)>&
                                obs_model,
                            util::Rng& rng, double ess_frac = 0.5) {
  const ObsUpdate obs_update = [&](int t, const Eigen::VectorXd& y_t, int k,
                                   statespace::KalmanState* st) -> double {
    const ObsModel om = obs_model(t, *st, k);
    return statespace::kalman_update(om.H, om.d, om.R, y_t, st);
  };
  return rbpf(N, mean0, cov0, y, mixture_builder, obs_update, rng, ess_frac, false, false);
}

}  // namespace filters
