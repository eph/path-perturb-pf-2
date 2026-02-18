#include "filters/bootstrap_pf.hpp"
#include "filters/rbpf.hpp"
#include "statespace/linear_gaussian.hpp"
#include "util/rng.hpp"
#include "util/stats.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    // 1D linear-Gaussian state-space model:
    //   x_{t+1} = a x_t + eta_t,  eta_t ~ N(0, q)
    //   y_t     = h x_t + eps_t,  eps_t ~ N(0, r)
    //
    // For K=1 mixture component, RBPF should match the exact Kalman log-likelihood
    // (up to numerical error), regardless of particle count.
    const int T = 50;
    const double a = 0.9;
    const double q = 0.1 * 0.1;
    const double h = 1.0;
    const double r = 0.2 * 0.2;

    Eigen::MatrixXd A(1, 1);
    A(0, 0) = a;
    Eigen::VectorXd a0(1);
    a0(0) = 0.0;
    Eigen::MatrixXd Q(1, 1);
    Q(0, 0) = q;

    Eigen::MatrixXd H(1, 1);
    H(0, 0) = h;
    Eigen::VectorXd d(1);
    d(0) = 0.0;
    Eigen::MatrixXd R(1, 1);
    R(0, 0) = r;

    const Eigen::VectorXd mean0 = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd cov0(1, 1);
    cov0(0, 0) = 1.0;

    // Simulate data.
    const std::uint64_t seed_data = 20260218ULL;
    util::Rng rng_data(seed_data);
    double x = 0.0;
    std::vector<Eigen::VectorXd> y(T);
    for (int t = 0; t < T; ++t) {
      x = a * x + std::sqrt(q) * rng_data.normal();
      const double yt = h * x + std::sqrt(r) * rng_data.normal();
      Eigen::VectorXd y1(1);
      y1(0) = yt;
      y[t] = y1;
    }

    // Exact Kalman log-likelihood.
    statespace::KalmanState st;
    st.mean = mean0;
    st.cov = cov0;
    double ll_kf = 0.0;
    for (int t = 0; t < T; ++t) {
      statespace::kalman_predict(A, a0, Q, &st);
      ll_kf += statespace::kalman_update(H, d, R, y[t], &st);
    }

    // RBPF with a single mixture component.
    const int N = 32;
    const std::uint64_t seed_pf = 20260219ULL;
    util::Rng rng_pf(seed_pf);

    const auto mixture_builder = [&](int /*t*/, const Eigen::VectorXd& /*ref_prev*/) {
      statespace::GaussianMixtureTransition mix;
      mix.components.resize(1);
      mix.weights = {1.0};
      mix.components[0].A = A;
      mix.components[0].a = a0;
      mix.components[0].Q = Q;
      return mix;
    };

    const auto obs_model = [&](int /*t*/, const statespace::KalmanState& /*pred*/, int /*k*/) {
      filters::ObsModel om;
      om.H = H;
      om.d = d;
      om.R = R;
      return om;
    };

    const filters::RbpfDiagnostics diag =
        filters::rbpf(N, mean0, cov0, y, mixture_builder, obs_model, rng_pf, 0.5);

    const double abs_diff = std::abs(diag.loglik - ll_kf);
    std::cout << "Kalman loglik: " << ll_kf << "\n";
    std::cout << "RBPF loglik:   " << diag.loglik << "\n";
    std::cout << "abs diff:      " << abs_diff << "\n";

    const double tol = 1e-8;
    if (!(abs_diff <= tol)) {
      std::cerr << "linear_gaussian_validate: FAIL (diff > " << tol << ")\n";
      return 1;
    }

    // Bootstrap PF sanity check (not exact, but should be close with enough particles).
    {
      util::Rng rng_bpf(20260220ULL);
      const int Nbpf = 4096;
      Eigen::MatrixXd P0_sqrt(1, 1);
      P0_sqrt(0, 0) = std::sqrt(cov0(0, 0));

      const auto transition = [&](Eigen::VectorXd* state, util::Rng& rin, int /*t*/) {
        const double eta = std::sqrt(q) * rin.normal();
        (*state)(0) = a * (*state)(0) + eta;
      };
      const auto log_obs = [&](const Eigen::VectorXd& state, const Eigen::VectorXd& y_t, int /*t*/) {
        Eigen::VectorXd yhat(1);
        yhat(0) = h * state(0);
        return util::log_mvnorm_pdf(y_t, yhat, R);
      };

      const filters::PfDiagnostics bpf =
          filters::bootstrap_pf(Nbpf, mean0, P0_sqrt, y, transition, log_obs, rng_bpf, 0.5);
      std::cout << "Bootstrap PF loglik (N=" << Nbpf << "): " << bpf.loglik << "\n";
      std::cout << "Bootstrap PF abs diff vs KF: " << std::abs(bpf.loglik - ll_kf) << "\n";
    }

    std::cout << "linear_gaussian_validate: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "linear_gaussian_validate error: " << e.what() << "\n";
    return 1;
  }
}
