#pragma once

#include <cstdint>
#include <optional>
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

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TORCHDTW_HOST_DEVICE __host__ __device__
#else
#define TORCHDTW_HOST_DEVICE
#endif

// Calls f.template operator()<distances_t, sx_t>() for the dtypes of the distances and of the lengths of dtw_batch.
template <typename F>
void dispatch_dtw(
    torch::headeronly::ScalarType distances_dtype, torch::headeronly::ScalarType lengths_dtype, const F& f) {
  THO_DISPATCH_V2(
      distances_dtype,
      "dtw",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            lengths_dtype,
            "dtw_lengths",
            AT_WRAP([&] {
              // Parenthesized, or the outer dispatch macro splits the template arguments on their comma.
              (f.template operator()<distances_t, scalar_t>());
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
}

#if defined(_MSC_VER)
#define TORCHDTW_ALWAYS_INLINE __forceinline
#else
#define TORCHDTW_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// Sequence lengths, read from a 1D tensor, or all equal to `fallback` (the padded size) when it is absent.
template <typename sx_t> struct Lengths {
  const sx_t* data;
  int64_t stride;
  int64_t fallback;

  TORCHDTW_HOST_DEVICE int64_t operator[](int64_t i) const {
    return data == nullptr ? fallback : static_cast<int64_t>(data[i * stride]);
  }
};

template <typename sx_t> inline Lengths<sx_t> make_lengths(const std::optional<Tensor>& t, int64_t fallback) {
  if (!t.has_value())
    return {nullptr, 0, fallback};
  return {reinterpret_cast<const sx_t*>(t->data_ptr()), t->strides()[0], fallback};
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
