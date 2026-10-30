#pragma once

#include <cmath>
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

// Whether sequence lengths can have this dtype: any integer dtype.
inline bool is_lengths_dtype(torch::headeronly::ScalarType dtype) {
  using torch::headeronly::ScalarType;
  return dtype == ScalarType::Byte || dtype == ScalarType::Char || dtype == ScalarType::Short ||
         dtype == ScalarType::Int || dtype == ScalarType::Long || dtype == ScalarType::UInt16 ||
         dtype == ScalarType::UInt32 || dtype == ScalarType::UInt64;
}

// Calls f(T()) with the C++ type T of an integer dtype of sequence lengths.
template <typename F>
TORCHDTW_HOST_DEVICE inline int64_t visit_lengths_dtype(torch::headeronly::ScalarType dtype, F f) {
  using torch::headeronly::ScalarType;
  switch (dtype) {
  case ScalarType::Byte:
    return f(uint8_t());
  case ScalarType::Char:
    return f(int8_t());
  case ScalarType::Short:
    return f(int16_t());
  case ScalarType::Int:
    return f(int32_t());
  case ScalarType::UInt16:
    return f(uint16_t());
  case ScalarType::UInt32:
    return f(uint32_t());
  case ScalarType::UInt64:
    return f(uint64_t());
  default:
    return f(int64_t());
  }
}

// Checks of the optional lengths sx and sy of the ops, for batch dimensions of sizes nx and ny.
inline void check_lengths_tensors(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, int64_t nx, int64_t ny) {
  const int64_t sizes[2] = {nx, ny};
  const std::optional<Tensor>* lengths[2] = {&sx, &sy};
  for (int k = 0; k < 2; k++) {
    if (!lengths[k]->has_value())
      continue;
    const Tensor& t = lengths[k]->value();
    STD_TORCH_CHECK(t.dim() == 1, "sx and sy must be 1D tensors");
    STD_TORCH_CHECK(t.size(0) == sizes[k], "sx and sy sizes must match the batch dimensions of distances");
    STD_TORCH_CHECK(is_lengths_dtype(t.scalar_type()), "sx and sy must have an integer dtype");
    STD_TORCH_CHECK(
        t.is_cuda() == distances.is_cuda() && t.get_device() == distances.get_device(),
        "sx and sy must be on the same device as distances");
  }
  if (sx.has_value() && sy.has_value())
    STD_TORCH_CHECK(sx->scalar_type() == sy->scalar_type(), "sx and sy dtypes do not match");
}

// Sequence lengths, read from a 1D integer tensor, or all equal to `fallback` (the padded size) when it is absent.
// The dtype is switched on at run time instead of being a template parameter: kernels are then compiled once instead
// of once per integer dtype, which divides the size of the binaries.
struct Lengths {
  const void* data;
  int64_t stride;
  int64_t fallback;
  torch::headeronly::ScalarType dtype;

  TORCHDTW_HOST_DEVICE int64_t operator[](int64_t i) const {
    if (data == nullptr)
      return fallback;
    return visit_lengths_dtype(
        dtype, [&](auto t) { return static_cast<int64_t>(static_cast<const decltype(t)*>(data)[i * stride]); });
  }
};

inline Lengths make_lengths(const std::optional<Tensor>& t, int64_t fallback) {
  if (!t.has_value())
    return {nullptr, 0, fallback, torch::headeronly::ScalarType::Long};
  return {t->data_ptr(), t->strides()[0], fallback, t->scalar_type()};
}

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
