#pragma once

#include <cstdint>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <type_traits>

namespace torchdtw {

using torch::stable::Tensor;
template <typename T, size_t N> using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;
template <typename T, size_t N> inline TensorAccessor<T, N> accessor(Tensor t) {
  return TensorAccessor<T, N>(reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

// Accumulation dtype of the DPs, shared by the CPU and GPU backends: float for half/bfloat16, scalar_t otherwise.
template <typename scalar_t>
using acc_t = std::conditional_t<
    std::is_same_v<scalar_t, torch::headeronly::Half> || std::is_same_v<scalar_t, torch::headeronly::BFloat16>, float,
    scalar_t>;

#if defined(_MSC_VER)
#define TORCHDTW_ALWAYS_INLINE __forceinline
#else
#define TORCHDTW_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// Row-major DP over an N x M grid, stored in `out` with row stride `ld`. `cell` computes each cell's value:
// origin() for (0, 0), edge(i, j, parent) on the first row or column where a cell has a single parent,
// and inner(i, j, diag, left, up) elsewhere. Always inlined, with `cell` taken by value, so that the
// compiler sees the cell's members as plain values: it can then specialize the inner loop for contiguous
// rows, and passes them in registers to the caller instead of on the stack.
template <typename T, typename Cell>
TORCHDTW_ALWAYS_INLINE void fill_grid(const Cell cell, int64_t N, int64_t M, T* out, int64_t ld) {
  out[0] = cell.origin();
  for (int64_t i = 1; i < N; i++) {
    out[i * ld] = cell.edge(i, 0, out[(i - 1) * ld]);
  }
  for (int64_t j = 1; j < M; j++) {
    out[j] = cell.edge(0, j, out[j - 1]);
  }
  for (int64_t i = 1; i < N; i++) {
    T* row = out + i * ld;
    const T* prev = row - ld;
    for (int64_t j = 1; j < M; j++) {
      row[j] = cell.inner(i, j, prev[j - 1], row[j - 1], prev[j]);
    }
  }
}

} // namespace torchdtw
