#pragma once

#include <Eigen/Dense>

#include <stdexcept>
#include <vector>

namespace solvers {

// Solves a block-tridiagonal linear system:
//
//   A0 x0 + B0 x1                 = d0
//   C0 x0 + A1 x1 + B1 x2         = d1
//           ...
//          C_{T-2} x_{T-2} + A_{T-1} x_{T-1} = d_{T-1}
//
// where Ai are n×n, Bi and Ci are n×n, and di are n×1.
//
// Returns x blocks (T vectors of size n).
inline std::vector<Eigen::VectorXd> solve_block_tridiagonal(
    std::vector<Eigen::MatrixXd> A, const std::vector<Eigen::MatrixXd>& B,
    const std::vector<Eigen::MatrixXd>& C, std::vector<Eigen::VectorXd> d) {
  const int T = static_cast<int>(A.size());
  if (T == 0) throw std::invalid_argument("block_tridiag: empty");
  if (static_cast<int>(d.size()) != T) throw std::invalid_argument("block_tridiag: d size");
  if (static_cast<int>(B.size()) != T - 1) throw std::invalid_argument("block_tridiag: B size");
  if (static_cast<int>(C.size()) != T - 1) throw std::invalid_argument("block_tridiag: C size");

  const int n = static_cast<int>(A[0].rows());
  if (n == 0 || A[0].cols() != n) throw std::invalid_argument("block_tridiag: A0 dim");
  for (int i = 0; i < T; ++i) {
    if (A[i].rows() != n || A[i].cols() != n) throw std::invalid_argument("block_tridiag: Ai dim");
    if (d[i].size() != n) throw std::invalid_argument("block_tridiag: di dim");
    if (i < T - 1) {
      if (B[i].rows() != n || B[i].cols() != n) throw std::invalid_argument("block_tridiag: Bi dim");
      if (C[i].rows() != n || C[i].cols() != n) throw std::invalid_argument("block_tridiag: Ci dim");
    }
  }

  std::vector<Eigen::MatrixXd> G(T - 1);

  // Forward elimination.
  for (int i = 0; i < T - 1; ++i) {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A[i]);
    if (!lu.isInvertible()) throw std::runtime_error("block_tridiag: singular Ai");
    const Eigen::MatrixXd Ai_inv_Bi = lu.solve(B[i]);
    const Eigen::VectorXd Ai_inv_di = lu.solve(d[i]);
    G[i] = Ai_inv_Bi;
    A[i + 1] = A[i + 1] - C[i] * Ai_inv_Bi;
    d[i + 1] = d[i + 1] - C[i] * Ai_inv_di;
  }

  // Back substitution.
  std::vector<Eigen::VectorXd> x(T);
  {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A[T - 1]);
    if (!lu.isInvertible()) throw std::runtime_error("block_tridiag: singular last A");
    x[T - 1] = lu.solve(d[T - 1]);
  }
  for (int i = T - 2; i >= 0; --i) {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A[i]);
    if (!lu.isInvertible()) throw std::runtime_error("block_tridiag: singular Ai (back)");
    x[i] = lu.solve(d[i] - B[i] * x[i + 1]);
  }

  return x;
}

}  // namespace solvers

