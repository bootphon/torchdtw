#pragma once

#include <cmath>
#include <cstdint>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
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

// Calls f.template operator()<distances_t>() for the dtype of the distances of the dtw ops.
template <typename F> void dispatch_dtw(torch::headeronly::ScalarType distances_dtype, const F& f) {
  THO_DISPATCH_V2(
      distances_dtype,
      "dtw",
      AT_WRAP([&] { f.template operator()<scalar_t>(); }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
}

#if defined(_MSC_VER)
#define TORCHDTW_ALWAYS_INLINE __forceinline
#else
#define TORCHDTW_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// Calls f(x, y) on the pairs of flat indices [start, end) of the upper triangle of an n x n grid of pairs, ordered
// by y: pair (x, y) has index b = y * (y - 1) / 2 + x for x < y, or with the diagonal b = y * (y + 1) / 2 + x for
// x <= y. Parallelizing over b rather than over rows gives every worker as many pairs: row x owns n - 1 - x of them.
template <typename F> inline void for_upper_triangle(int64_t start, int64_t end, bool diagonal, const F& f) {
  // With the diagonal, the pairs (x, y) are the pairs x < y + 1 of the strict triangle, shifted by one.
  const int64_t shift = diagonal ? 1 : 0;
  // Seek (x, y) for flat index `start` via the triangular-number inverse, corrected with integer steps to absorb
  // double rounding; then advance incrementally to avoid a sqrt per pair.
  int64_t y = static_cast<int64_t>((1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(start))) / 2.0);
  while (y > 0 && y * (y - 1) / 2 > start)
    y--;
  while ((y + 1) * y / 2 <= start)
    y++;
  int64_t x = start - y * (y - 1) / 2;
  for (int64_t b = start; b < end; b++) {
    f(x, y - shift);
    if (++x == y) {
      x = 0;
      y++;
    }
  }
}

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
