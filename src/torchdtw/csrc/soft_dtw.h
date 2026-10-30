#pragma once

#include "common.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>

// Soft-DTW (Cuturi & Blondel, 2017) cell math and input handling, shared by csrc/soft_dtw.cpp and
// csrc/cuda/soft_dtw.cu so that both backends run the same recurrence.
//
// Forward: R(i, j) = D(i, j) + softmin_gamma(R(i - 1, j - 1), R(i, j - 1), R(i - 1, j)), where cells on the
// first row or column take their single parent exactly, and cells outside the Sakoe-Chiba band are +inf.
// Backward: with E = dR(N - 1, M - 1) / dR, each cell sums its successors' E weighted by the softmin weight it
// had in them. It is computed in log space, on the grid flipped through (N - 1 - i, M - 1 - j): there, the
// successors of a cell are its diag/left/up parents and (N - 1, M - 1) is the origin, so both passes run on the
// same DP skeletons.

namespace torchdtw {

template <typename T> TORCHDTW_HOST_DEVICE inline T infinity() { return std::numeric_limits<T>::infinity(); }

/* Sakoe-Chiba band around the diagonal joining (0, 0) to (nx - 1, ny - 1).

   With K = min(nx, ny), L = max(nx, ny) and slope s = (L - 1) / (K - 1), a cell at short-axis index k and
   long-axis index l is in band iff |l - k * s| <= max(bandwidth, s / 2), i.e. the bandwidth is measured in
   long-axis steps. The s / 2 floor guarantees a connected warping path to the terminal cell even for
   bandwidth = 0. For square inputs s = 1 and this reduces to |i - j| <= bandwidth. Evaluated in integer
   arithmetic (both sides multiplied by 2 * (K - 1)) so the CPU and CUDA kernels agree exactly.
   A negative bandwidth disables the band. */
TORCHDTW_HOST_DEVICE inline bool in_band(int64_t i, int64_t j, int64_t nx, int64_t ny, int64_t bandwidth) {
  if (bandwidth < 0)
    return true;
  const bool x_short = nx <= ny;
  const int64_t K = x_short ? nx : ny;
  const int64_t L = x_short ? ny : nx;
  if (K <= 1)
    return true;
  const int64_t k = x_short ? i : j;
  const int64_t l = x_short ? j : i;
  int64_t diff = l * (K - 1) - k * (L - 1);
  if (diff < 0)
    diff = -diff;
  const int64_t band = 2 * bandwidth * (K - 1);
  const int64_t width = band > L - 1 ? band : L - 1;
  return 2 * diff <= width;
}

// -gamma * log(exp(-a / gamma) + exp(-b / gamma) + exp(-c / gamma)), shifted by the minimum for stability.
template <typename T> TORCHDTW_HOST_DEVICE inline T softmin3(T a, T b, T c, T gamma, T inv_gamma) {
  T m = a < b ? a : b;
  m = m < c ? m : c;
  if (m == infinity<T>())
    return m;
  return m - gamma * std::log(
                         std::exp((m - a) * inv_gamma) + std::exp((m - b) * inv_gamma) + std::exp((m - c) * inv_gamma));
}

// log(exp(a) + exp(b) + exp(c)), shifted by the maximum for stability.
template <typename T> TORCHDTW_HOST_DEVICE inline T logsumexp3(T a, T b, T c) {
  T m = a > b ? a : b;
  m = m > c ? m : c;
  if (m == -infinity<T>())
    return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m) + std::exp(c - m));
}

// Log-space contribution of a successor s, with log E(s) = log_e_s, to a cell of cost r: the softmin weight
// of the cell in s is exp((R(s) - D(s) - r) / gamma). A successor outside the band (R(s) = +inf) contributes 0.
template <typename T> TORCHDTW_HOST_DEVICE inline T backward_term(T log_e_s, T r_s, T d_s, T r, T inv_gamma) {
  if (r_s == infinity<T>())
    return -infinity<T>();
  return log_e_s + (r_s - d_s - r) * inv_gamma;
}

// The soft_dtw ops take distances of shape (B, N, M), for B independent pairs, or (n1, n2, s1, s2), for all pairs
// between two batches. Both are handled as (n1, n2, s1, s2), the former viewed as (B, 1, N, M); pair (x, y) has
// lengths sx[x] and sy[y] for all pairs, and sx[x] and sy[x] when paired.
inline Tensor as_all_pairs(const Tensor& distances) {
  if (distances.dim() == 4)
    return distances;
  return torch::stable::view(distances, {distances.size(0), 1, distances.size(1), distances.size(2)});
}

// Storage dtype of the R matrix, matching acc_t: float for half/bfloat16, the dtype of distances otherwise.
inline torch::headeronly::ScalarType acc_scalar_type(torch::headeronly::ScalarType dtype) {
  using torch::headeronly::ScalarType;
  return dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ? ScalarType::Float : dtype;
}

// Checks shared by the CPU and CUDA soft_dtw ops, and the dtype to dispatch the lengths on (Long if both absent).
inline torch::headeronly::ScalarType check_soft_dtw_inputs(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, double gamma) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(
      distances.dim() == 3 || distances.dim() == 4, "distances must be a 3D (paired) or 4D (all pairs) tensor");
  const ScalarType dtype = distances.scalar_type();
  STD_TORCH_CHECK(
      dtype == ScalarType::Float || dtype == ScalarType::Double || dtype == ScalarType::Half ||
          dtype == ScalarType::BFloat16,
      "distances must be float16, bfloat16, float32 or float64");
  STD_TORCH_CHECK(gamma > 0, "gamma must be positive");
  const bool paired = distances.dim() == 3;
  const int64_t sizes[2] = {distances.size(0), paired ? distances.size(0) : distances.size(1)};
  const std::optional<Tensor>* lengths[2] = {&sx, &sy};
  for (int k = 0; k < 2; k++) {
    if (!lengths[k]->has_value())
      continue;
    const Tensor& t = lengths[k]->value();
    STD_TORCH_CHECK(t.dim() == 1, "sx and sy must be 1D tensors");
    STD_TORCH_CHECK(t.size(0) == sizes[k], "sx and sy sizes must match the batch dimensions of distances");
    const ScalarType t_dtype = t.scalar_type();
    STD_TORCH_CHECK(
        t_dtype == ScalarType::Byte || t_dtype == ScalarType::Char || t_dtype == ScalarType::Short ||
            t_dtype == ScalarType::Int || t_dtype == ScalarType::Long || t_dtype == ScalarType::UInt16 ||
            t_dtype == ScalarType::UInt32 || t_dtype == ScalarType::UInt64,
        "sx and sy must have an integer dtype");
    STD_TORCH_CHECK(
        t.is_cuda() == distances.is_cuda() && t.get_device() == distances.get_device(),
        "sx and sy must be on the same device as distances");
  }
  if (sx.has_value() && sy.has_value())
    STD_TORCH_CHECK(sx->scalar_type() == sy->scalar_type(), "sx and sy dtypes do not match");
  if (sx.has_value())
    return sx->scalar_type();
  return sy.has_value() ? sy->scalar_type() : ScalarType::Long;
}

// Checks of the backward op's own inputs: the gradient of the cost and the R matrix saved by the forward.
inline void check_soft_dtw_backward_inputs(const Tensor& grad, const Tensor& distances, const Tensor& R) {
  STD_TORCH_CHECK(
      R.numel() > 0 || distances.numel() == 0,
      "soft_dtw_backward needs the R matrix of the forward pass, computed with keep_R=True");
  STD_TORCH_CHECK(R.dim() == distances.dim(), "R must have the shape of distances");
  for (int64_t d = 0; d < R.dim(); d++)
    STD_TORCH_CHECK(R.size(d) == distances.size(d), "R must have the shape of distances");
  STD_TORCH_CHECK(R.is_contiguous(), "R must be contiguous");
  STD_TORCH_CHECK(
      R.scalar_type() == acc_scalar_type(distances.scalar_type()), "R must have the accumulation dtype of distances");
  STD_TORCH_CHECK(grad.dim() == distances.dim() - 2, "grad must have the shape of the soft_dtw output");
  for (int64_t d = 0; d < grad.dim(); d++)
    STD_TORCH_CHECK(grad.size(d) == distances.size(d), "grad must have the shape of the soft_dtw output");
  STD_TORCH_CHECK(grad.scalar_type() == distances.scalar_type(), "grad must have the dtype of distances");
}

// Calls f.template operator()<distances_t, sx_t>() for the dtypes of the distances and of the lengths.
template <typename F>
void dispatch_soft_dtw(
    torch::headeronly::ScalarType distances_dtype, torch::headeronly::ScalarType lengths_dtype, const F& f) {
  THO_DISPATCH_V2(
      distances_dtype,
      "soft_dtw",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            lengths_dtype,
            "soft_dtw_lengths",
            AT_WRAP([&] {
              // Parenthesized, or the outer dispatch macro splits the template arguments on their comma.
              (f.template operator()<distances_t, scalar_t>());
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_FLOATING_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
}

// The soft_dtw op, shared by the backends: run.template operator()<distances_t, sx_t>(out, R, pairs, paired) fills
// the (n1, n2) costs `out`, and the R matrix when keep_R, from the distances viewed as all pairs.
template <typename Run>
std::tuple<Tensor, Tensor> soft_dtw_op(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, double gamma,
    bool keep_R, const Run& run) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  Tensor out = torch::stable::new_empty(distances, {pairs.size(0), pairs.size(1)});
  const auto acc_dtype = acc_scalar_type(distances.scalar_type());
  Tensor R = keep_R ? torch::stable::new_empty(distances, distances.sizes(), acc_dtype)
                    : torch::stable::new_empty(distances, {0}, acc_dtype);
  dispatch_soft_dtw(distances.scalar_type(), lengths_dtype, [&]<typename distances_t, typename sx_t>() {
    run.template operator()<distances_t, sx_t>(out, R, pairs, paired);
  });
  if (paired)
    out = torch::stable::view(out, {pairs.size(0)});
  return {out, R};
}

// The soft_dtw_backward op, shared by the backends: run.template operator()<distances_t, sx_t>(grad_distances,
// grad_pairs, pairs, paired) fills `grad_distances`, with the gradient of the costs viewed as (n1, n2).
template <typename Run>
Tensor soft_dtw_backward_op(
    const Tensor& grad, const Tensor& distances, const Tensor& R, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, double gamma, const Run& run) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  check_soft_dtw_backward_inputs(grad, distances, R);
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  const Tensor grad_pairs = paired ? torch::stable::view(grad, {pairs.size(0), 1}) : grad;
  Tensor grad_distances = torch::stable::new_empty(distances, distances.sizes());
  dispatch_soft_dtw(distances.scalar_type(), lengths_dtype, [&]<typename distances_t, typename sx_t>() {
    run.template operator()<distances_t, sx_t>(grad_distances, grad_pairs, pairs, paired);
  });
  return grad_distances;
}

} // namespace torchdtw
