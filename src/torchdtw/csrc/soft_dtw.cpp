#include "soft_dtw.h"
#include <algorithm>
#include <cmath>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>
#include <vector>

namespace torchdtw {

// R(i, j) of the forward pass, see soft_dtw.h.
template <typename scalar_t> struct SoftForwardCell {
  using acc = acc_t<scalar_t>;
  PairDistances<scalar_t> distances;
  int64_t nx;
  int64_t ny;
  int64_t bandwidth;
  acc gamma;
  acc inv_gamma;

  acc origin() const { return distances(0, 0); }
  acc edge(int64_t i, int64_t j, acc parent) const {
    return in_band(i, j, nx, ny, bandwidth) ? distances(i, j) + parent : infinity<acc>();
  }
  acc inner(int64_t i, int64_t j, acc diag, acc left, acc up) const {
    return in_band(i, j, nx, ny, bandwidth) ? distances(i, j) + softmin3(diag, left, up, gamma, inv_gamma)
                                            : infinity<acc>();
  }
};

// log E of the backward pass, on the flipped grid: the cell (fi, fj) is (i, j) = (nx - 1 - fi, ny - 1 - fj), and
// its flipped diag/left/up parents are its successors (i + 1, j + 1), (i, j + 1) and (i + 1, j). On the first
// flipped row and column, its single successor is (i, j + 1) and (i + 1, j) respectively.
template <typename scalar_t> struct SoftBackwardCell {
  using acc = acc_t<scalar_t>;
  PairDistances<scalar_t> distances;
  const acc* R;
  int64_t ld;
  int64_t nx;
  int64_t ny;
  acc inv_gamma;

  acc term(acc log_e_s, int64_t si, int64_t sj, acc r) const {
    return backward_term(log_e_s, R[si * ld + sj], distances(si, sj), r, inv_gamma);
  }

  acc origin() const { return 0; }
  acc edge(int64_t fi, int64_t fj, acc parent) const {
    const int64_t i = nx - 1 - fi;
    const int64_t j = ny - 1 - fj;
    const acc r = R[i * ld + j];
    if (r == infinity<acc>())
      return -infinity<acc>();
    return fi == 0 ? term(parent, i, j + 1, r) : term(parent, i + 1, j, r);
  }
  acc inner(int64_t fi, int64_t fj, acc diag, acc left, acc up) const {
    const int64_t i = nx - 1 - fi;
    const int64_t j = ny - 1 - fj;
    const acc r = R[i * ld + j];
    if (r == infinity<acc>())
      return -infinity<acc>();
    return logsumexp3(term(diag, i + 1, j + 1, r), term(left, i, j + 1, r), term(up, i + 1, j, r));
  }
};

template <typename T> static void fill_padding(T* block, int64_t s1, int64_t s2, int64_t N, int64_t M, T value) {
  for (int64_t i = 0; i < s1; i++) {
    for (int64_t j = i < N ? M : 0; j < s2; j++)
      block[i * s2 + j] = value;
  }
}

template <typename distances_t>
void soft_dtw_cpu_impl(
    Tensor& out, Tensor& R, bool keep_R, const Tensor& pairs, const std::optional<Tensor>& sx_opt,
    const std::optional<Tensor>& sy_opt, bool paired, bool symmetric, double gamma, int64_t bandwidth, bool normalize) {
  using acc = acc_t<distances_t>;
  const int64_t n2 = pairs.size(1);
  const int64_t s1 = pairs.size(2);
  const int64_t s2 = pairs.size(3);
  const Lengths sx = make_lengths(sx_opt, s1);
  const Lengths sy = make_lengths(sy_opt, s2);
  check_lengths(sx, sy, pairs.size(0), paired ? pairs.size(0) : n2, s1, s2);
  const auto distances_a = accessor<const distances_t, 4>(pairs);
  auto out_a = accessor<distances_t, 2>(out);
  acc* const R_data = keep_R ? reinterpret_cast<acc*>(R.data_ptr()) : nullptr;
  const acc gamma_acc = static_cast<acc>(gamma);
  const acc inv_gamma = static_cast<acc>(1) / gamma_acc;

  parallel_pairs<acc>(pairs.size(0), n2, symmetric, true, [&](int64_t x, int64_t y, std::vector<acc>& workspace) {
    const bool mirror = symmetric && x != y;
    const int64_t N = std::max<int64_t>(sx[x], 0);
    const int64_t M = std::max<int64_t>(sy[paired ? x : y], 0);
    if (R_data != nullptr) {
      fill_padding(R_data + (x * n2 + y) * s1 * s2, s1, s2, N, M, infinity<acc>());
      if (mirror)
        fill_padding(R_data + (y * n2 + x) * s1 * s2, s1, s2, 0, 0, infinity<acc>());
    }
    distances_t value = 0;
    if (N > 0 && M > 0) {
      acc* cost;
      int64_t ld;
      if (R_data != nullptr) {
        cost = R_data + (x * n2 + y) * s1 * s2;
        ld = s2;
      } else {
        if (static_cast<int64_t>(workspace.size()) < N * M)
          workspace.resize(N * M);
        cost = workspace.data();
        ld = M;
      }
      const SoftForwardCell<distances_t> cell{
          PairDistances<distances_t>(distances_a[x][y]), N, M, bandwidth, gamma_acc, inv_gamma};
      fill_grid(cell, N, M, cost, ld);
      const acc r = cost[(N - 1) * ld + M - 1];
      value = static_cast<distances_t>(normalize ? r / longest_path<acc>(N, M) : r);
    }
    out_a[x][y] = value;
    if (mirror)
      out_a[y][x] = value;
  });
}

template <typename distances_t>
void soft_dtw_backward_cpu_impl(
    Tensor& grad_distances, const Tensor& grad, const Tensor& pairs, const Tensor& R,
    const std::optional<Tensor>& sx_opt, const std::optional<Tensor>& sy_opt, bool paired, bool symmetric, double gamma,
    bool normalize) {
  using acc = acc_t<distances_t>;
  const int64_t n2 = pairs.size(1);
  const int64_t s1 = pairs.size(2);
  const int64_t s2 = pairs.size(3);
  const Lengths sx = make_lengths(sx_opt, s1);
  const Lengths sy = make_lengths(sy_opt, s2);
  check_lengths(sx, sy, pairs.size(0), paired ? pairs.size(0) : n2, s1, s2);
  const auto distances_a = accessor<const distances_t, 4>(pairs);
  const auto grad_a = accessor<const distances_t, 2>(grad);
  const acc* const R_data = reinterpret_cast<const acc*>(R.data_ptr());
  distances_t* const out = reinterpret_cast<distances_t*>(grad_distances.data_ptr());
  const acc inv_gamma = static_cast<acc>(1) / static_cast<acc>(gamma);

  parallel_pairs<acc>(pairs.size(0), n2, symmetric, true, [&](int64_t x, int64_t y, std::vector<acc>& log_e) {
    const bool mirror = symmetric && x != y;
    const int64_t N = std::max<int64_t>(sx[x], 0);
    const int64_t M = std::max<int64_t>(sy[paired ? x : y], 0);
    distances_t* const out_xy = out + (x * n2 + y) * s1 * s2;
    fill_padding(out_xy, s1, s2, N, M, static_cast<distances_t>(0));
    if (mirror)
      fill_padding(out + (y * n2 + x) * s1 * s2, s1, s2, 0, 0, static_cast<distances_t>(0));
    if (N == 0 || M == 0)
      return;
    if (static_cast<int64_t>(log_e.size()) < N * M)
      log_e.resize(N * M);
    const SoftBackwardCell<distances_t> cell{
        PairDistances<distances_t>(distances_a[x][y]), R_data + (x * n2 + y) * s1 * s2, s2, N, M, inv_gamma};
    fill_grid(cell, N, M, log_e.data(), M);
    acc g = static_cast<acc>(grad_a[x][y]);
    if (mirror)
      g += static_cast<acc>(grad_a[y][x]);
    if (normalize)
      g /= longest_path<acc>(N, M);
    for (int64_t i = 0; i < N; i++) {
      const acc* const log_e_i = log_e.data() + (N - 1 - i) * M;
      for (int64_t j = 0; j < M; j++)
        out_xy[i * s2 + j] = static_cast<distances_t>(g * std::exp(log_e_i[M - 1 - j]));
    }
  });
}

std::tuple<Tensor, Tensor> soft_dtw_cpu(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, double gamma,
    int64_t bandwidth, bool symmetric, bool keep_R, bool normalize) {
  return soft_dtw_op(
      distances,
      sx,
      sy,
      gamma,
      symmetric,
      keep_R,
      [&]<typename distances_t>(Tensor& out, Tensor& R, const Tensor& pairs, bool paired) {
        soft_dtw_cpu_impl<distances_t>(out, R, keep_R, pairs, sx, sy, paired, symmetric, gamma, bandwidth, normalize);
      });
}

Tensor soft_dtw_backward_cpu(
    const Tensor& grad, const Tensor& distances, const Tensor& R, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, double gamma, bool symmetric, bool normalize) {
  return soft_dtw_backward_op(
      grad,
      distances,
      R,
      sx,
      sy,
      gamma,
      symmetric,
      [&]<typename distances_t>(Tensor& grad_distances, const Tensor& grad_pairs, const Tensor& pairs, bool paired) {
        soft_dtw_backward_cpu_impl<distances_t>(
            grad_distances, grad_pairs, pairs, R, sx, sy, paired, symmetric, gamma, normalize);
      });
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchdtw, m) {
  m.def(
      "soft_dtw(Tensor distances, Tensor? sx, Tensor? sy, float gamma, int bandwidth, bool symmetric, bool keep_R, "
      "bool normalize=False) -> (Tensor, Tensor)");
  m.def(
      "soft_dtw_backward(Tensor grad, Tensor distances, Tensor R, Tensor? sx, Tensor? sy, float gamma, bool symmetric, "
      "bool normalize=False) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CPU, m) {
  m.impl("soft_dtw", &TORCH_BOX(soft_dtw_cpu));
  m.impl("soft_dtw_backward", &TORCH_BOX(soft_dtw_backward_cpu));
}

} // namespace torchdtw
