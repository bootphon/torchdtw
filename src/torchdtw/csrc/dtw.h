#pragma once

#include "common.h"
#include <cstdint>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <type_traits>

// DTW input handling and result, shared by csrc/dtw.cpp and csrc/cuda/dtw.cu so that both backends check the same
// inputs and give the same results.

namespace torchdtw {

// Checks shared by the CPU and CUDA dtw_batch ops. Returns the sizes (n1, n2, s1, s2) of the distances, queried once:
// each query is a call through the stable ABI, measured on the small-batch overhead.
inline torch::headeronly::IntHeaderOnlyArrayRef
check_dtw_batch_inputs(const Tensor& distances, const Tensor* sx, const Tensor* sy, bool symmetric) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");
  const torch::headeronly::IntHeaderOnlyArrayRef sizes = distances.sizes();
  check_lengths_tensors(distances, sx, sy, sizes[0], sizes[1]);
  STD_TORCH_CHECK(
      !symmetric || sizes[0] == sizes[1], "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  return sizes;
}

// The DTW cost of a pair: the accumulated cost of its last cell divided by the length of the path reaching it. For
// integral dtypes the division is in int64: casting the path length to a narrow dtype can yield 0 (e.g. 256 as int8)
// and divide by zero.
template <typename scalar_t>
TORCHDTW_HOST_DEVICE inline scalar_t normalized_cost(acc_t<scalar_t> cost, int64_t path_len) {
  if constexpr (std::is_integral_v<scalar_t>) {
    return static_cast<scalar_t>(static_cast<int64_t>(cost) / path_len);
  } else {
    return static_cast<scalar_t>(cost / static_cast<acc_t<scalar_t>>(path_len));
  }
}

// The dtw_batch op, shared by the backends, with the lengths sx and sy, nullptr when absent:
// run.template operator()<distances_t>(out, sizes, sx, sy) fills the (n1, n2) costs `out` from the distances of sizes
// (n1, n2, s1, s2). Every pair is written, with a cost of 0 when empty, except with `symmetric` the diagonal, left
// to the zeros of `out`.
template <typename Run>
Tensor dtw_batch_op(const Tensor& distances, const Tensor* sx, const Tensor* sy, bool symmetric, const Run& run) {
  const torch::headeronly::IntHeaderOnlyArrayRef sizes = check_dtw_batch_inputs(distances, sx, sy, symmetric);
  Tensor out = symmetric ? torch::stable::new_zeros(distances, {sizes[0], sizes[1]})
                         : torch::stable::new_empty(distances, {sizes[0], sizes[1]});
  dispatch_dtw(distances.scalar_type(), [&]<typename distances_t>() {
    run.template operator()<distances_t>(out, sizes, make_lengths(sx, sizes[2]), make_lengths(sy, sizes[3]));
  });
  return out;
}

} // namespace torchdtw
