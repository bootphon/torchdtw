#pragma once

#include "common.h"
#include <cstdint>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <type_traits>

namespace torchdtw {

inline torch::headeronly::IntHeaderOnlyArrayRef
check_dtw_batch_inputs(const Tensor& distances, const Tensor* sx, const Tensor* sy, bool symmetric) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");
  const torch::headeronly::IntHeaderOnlyArrayRef sizes = distances.sizes();
  check_lengths_tensors(distances, sx, sy, sizes[0], sizes[1]);
  STD_TORCH_CHECK(
      !symmetric || sizes[0] == sizes[1], "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  return sizes;
}

template <typename scalar_t, typename acc>
TORCHDTW_HOST_DEVICE inline scalar_t normalized_cost(acc cost, int64_t path_len) {
  if constexpr (std::is_integral_v<scalar_t>) {
    return static_cast<scalar_t>(static_cast<int64_t>(cost) / path_len);
  } else {
    return static_cast<scalar_t>(cost / static_cast<acc>(path_len));
  }
}

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
