#include "soft_dtw.h"
#include <algorithm>
#include <cmath>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>
#include <vector>

namespace torchdtw {

// Distances of one pair. As in HardCell, the strides are plain values so that fill_grid's inner loop can be
// specialized for contiguous rows.
template <typename scalar_t> struct PairDistances {
  const scalar_t* data;
  int64_t stride_i;
  int64_t stride_j;

  explicit PairDistances(const TensorAccessor<const scalar_t, 2>& distances)
      : data(distances.data()), stride_i(distances.stride(0)), stride_j(distances.stride(1)) {}

  acc_t<scalar_t> operator()(int64_t i, int64_t j) const {
    return static_cast<acc_t<scalar_t>>(data[i * stride_i + j * stride_j]);
  }
};

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

// Writes `value` to the cells of a pair's (s1, s2) block outside its (N, M) sub-block.
template <typename T> static void fill_padding(T* block, int64_t s1, int64_t s2, int64_t N, int64_t M, T value) {
  for (int64_t i = 0; i < s1; i++) {
    for (int64_t j = i < N ? M : 0; j < s2; j++)
      block[i * s2 + j] = value;
  }
}

template <typename sx_t>
static void check_lengths(const Lengths<sx_t>& sx, const Lengths<sx_t>& sy, const Tensor& pairs, bool paired) {
  for (int64_t x = 0; x < pairs.size(0); x++) {
    STD_TORCH_CHECK(sx[x] <= pairs.size(2), "sx values must not exceed distances.size(-2)");
  }
  for (int64_t y = 0; y < (paired ? pairs.size(0) : pairs.size(1)); y++) {
    STD_TORCH_CHECK(sy[y] <= pairs.size(3), "sy values must not exceed distances.size(-1)");
  }
}

template <typename distances_t, typename sx_t>
void soft_dtw_cpu_impl(
    Tensor& out, Tensor& R, bool keep_R, const Tensor& pairs, const std::optional<Tensor>& sx_opt,
    const std::optional<Tensor>& sy_opt, bool paired, double gamma, int64_t bandwidth) {
  using acc = acc_t<distances_t>;
  const int64_t n2 = pairs.size(1);
  const int64_t s1 = pairs.size(2);
  const int64_t s2 = pairs.size(3);
  const Lengths<sx_t> sx = make_lengths<sx_t>(sx_opt, s1);
  const Lengths<sx_t> sy = make_lengths<sx_t>(sy_opt, s2);
  check_lengths(sx, sy, pairs, paired);
  const auto distances_a = accessor<const distances_t, 4>(pairs);
  auto out_a = accessor<distances_t, 2>(out);
  acc* const R_data = keep_R ? reinterpret_cast<acc*>(R.data_ptr()) : nullptr;
  const acc gamma_acc = static_cast<acc>(gamma);
  const acc inv_gamma = static_cast<acc>(1) / gamma_acc;

  torch::stable::parallel_for(0, pairs.size(0) * n2, 1, [&](int64_t start, int64_t end) {
    std::vector<acc> workspace;
    for (int64_t p = start; p < end; p++) {
      const int64_t x = p / n2;
      const int64_t y = p % n2;
      const int64_t N = std::max<int64_t>(sx[x], 0);
      const int64_t M = std::max<int64_t>(sy[paired ? x : y], 0);
      // The padding of the kept R is +inf, as no alignment reaches it.
      if (R_data != nullptr)
        fill_padding(R_data + p * s1 * s2, s1, s2, N, M, infinity<acc>());
      if (N == 0 || M == 0) {
        out_a[x][y] = static_cast<distances_t>(0);
        continue;
      }
      // Fill R in place when it is kept for the backward pass, a reused workspace otherwise.
      acc* cost;
      int64_t ld;
      if (R_data != nullptr) {
        cost = R_data + p * s1 * s2;
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
      out_a[x][y] = static_cast<distances_t>(cost[(N - 1) * ld + M - 1]);
    }
  });
}

template <typename distances_t, typename sx_t>
void soft_dtw_backward_cpu_impl(
    Tensor& grad_distances, const Tensor& grad, const Tensor& pairs, const Tensor& R,
    const std::optional<Tensor>& sx_opt, const std::optional<Tensor>& sy_opt, bool paired, double gamma) {
  using acc = acc_t<distances_t>;
  const int64_t n2 = pairs.size(1);
  const int64_t s1 = pairs.size(2);
  const int64_t s2 = pairs.size(3);
  const Lengths<sx_t> sx = make_lengths<sx_t>(sx_opt, s1);
  const Lengths<sx_t> sy = make_lengths<sx_t>(sy_opt, s2);
  check_lengths(sx, sy, pairs, paired);
  const auto distances_a = accessor<const distances_t, 4>(pairs);
  const auto grad_a = accessor<const distances_t, 2>(grad);
  const acc* const R_data = reinterpret_cast<const acc*>(R.data_ptr());
  distances_t* const out = reinterpret_cast<distances_t*>(grad_distances.data_ptr());
  const acc inv_gamma = static_cast<acc>(1) / static_cast<acc>(gamma);

  torch::stable::parallel_for(0, pairs.size(0) * n2, 1, [&](int64_t start, int64_t end) {
    std::vector<acc> log_e;
    for (int64_t p = start; p < end; p++) {
      const int64_t x = p / n2;
      const int64_t y = p % n2;
      const int64_t N = std::max<int64_t>(sx[x], 0);
      const int64_t M = std::max<int64_t>(sy[paired ? x : y], 0);
      distances_t* const out_xy = out + p * s1 * s2;
      // Padding beyond the pair's lengths gets no gradient.
      fill_padding(out_xy, s1, s2, N, M, static_cast<distances_t>(0));
      if (N == 0 || M == 0)
        continue;
      if (static_cast<int64_t>(log_e.size()) < N * M)
        log_e.resize(N * M);
      const SoftBackwardCell<distances_t> cell{
          PairDistances<distances_t>(distances_a[x][y]), R_data + p * s1 * s2, s2, N, M, inv_gamma};
      fill_grid(cell, N, M, log_e.data(), M);
      const acc g = static_cast<acc>(grad_a[x][y]);
      for (int64_t i = 0; i < N; i++) {
        const acc* const log_e_i = log_e.data() + (N - 1 - i) * M;
        for (int64_t j = 0; j < M; j++)
          out_xy[i * s2 + j] = static_cast<distances_t>(g * std::exp(log_e_i[M - 1 - j]));
      }
    }
  });
}

std::tuple<Tensor, Tensor> soft_dtw_cpu(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, double gamma,
    int64_t bandwidth, bool keep_R) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  Tensor out = torch::stable::new_empty(distances, {pairs.size(0), pairs.size(1)});
  const auto acc_dtype = acc_scalar_type(distances.scalar_type());
  Tensor R = keep_R ? torch::stable::new_empty(distances, distances.sizes(), acc_dtype)
                    : torch::stable::new_empty(distances, {0}, acc_dtype);
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "soft_dtw_cpu",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            lengths_dtype,
            "soft_dtw_cpu_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (soft_dtw_cpu_impl<distances_t, sx_t>(out, R, keep_R, pairs, sx, sy, paired, gamma, bandwidth));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_FLOATING_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  if (paired)
    out = torch::stable::view(out, {pairs.size(0)});
  return {out, R};
}

Tensor soft_dtw_backward_cpu(
    const Tensor& grad, const Tensor& distances, const Tensor& R, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, double gamma) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  check_soft_dtw_backward_inputs(grad, distances, R);
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  const Tensor grad_pairs = paired ? torch::stable::view(grad, {pairs.size(0), 1}) : grad;
  Tensor grad_distances = torch::stable::new_empty(distances, distances.sizes());
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "soft_dtw_backward_cpu",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            lengths_dtype,
            "soft_dtw_backward_cpu_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (soft_dtw_backward_cpu_impl<distances_t, sx_t>(
                  grad_distances, grad_pairs, pairs, R, sx, sy, paired, gamma));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_FLOATING_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return grad_distances;
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchdtw, m) {
  m.def(
      "soft_dtw(Tensor distances, Tensor? sx, Tensor? sy, float gamma, int bandwidth, bool keep_R) -> (Tensor, "
      "Tensor)");
  m.def("soft_dtw_backward(Tensor grad, Tensor distances, Tensor R, Tensor? sx, Tensor? sy, float gamma) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CPU, m) {
  m.impl("soft_dtw", &TORCH_BOX(soft_dtw_cpu));
  m.impl("soft_dtw_backward", &TORCH_BOX(soft_dtw_backward_cpu));
}

} // namespace torchdtw
