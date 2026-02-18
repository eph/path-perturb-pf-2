#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace util {

class Rng {
 public:
  explicit Rng(std::uint64_t seed)
      : seed_(seed), gen_(seed), unif_(0.0, 1.0), norm_(0.0, 1.0) {}

  std::uint64_t seed() const { return seed_; }

  double uniform() { return unif_(gen_); }

  double normal() { return norm_(gen_); }

  Eigen::VectorXd normal_vec(int dim) {
    Eigen::VectorXd z(dim);
    for (int i = 0; i < dim; ++i) z(i) = normal();
    return z;
  }

  // Returns an index in [0, probs.size()).
  int categorical(const std::vector<double>& probs) {
    if (probs.empty()) throw std::invalid_argument("categorical: empty probs");
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(gen_);
  }

 private:
  std::uint64_t seed_{0};
  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> unif_;
  std::normal_distribution<double> norm_;
};

}  // namespace util
