#pragma once

#include "../common.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <type_traits>

// Default shared memory budget of 48KiB per block
#define MAX_SHARED_BYTES 49152
// Launches are chunked along grid.y, limited to 65535 blocks.
#define MAX_GRID_Y 65535
// Path lengths are stored as uint16_t: a pair's lengths must satisfy N + M - 1 <= 65535.
#define MAX_PATH_LENGTH 65535

#ifdef USE_ROCM
#define TORCHDTW_WARP_SIZE 64
#else
#define TORCHDTW_WARP_SIZE 32
#endif

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace torchdtw {

template <typename T, size_t N, typename index_t>
using PackedTensorAccessor =
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, torch::headeronly::RestrictPtrTraits, index_t>;
template <typename T, size_t N> using PackedTensorAccessor32 = PackedTensorAccessor<T, N, int32_t>;
template <typename T, size_t N> using PackedTensorAccessor64 = PackedTensorAccessor<T, N, int64_t>;
template <typename T, size_t N, typename index_t>
inline PackedTensorAccessor<T, N, index_t> packed_accessor(torch::stable::Tensor t) {
  return PackedTensorAccessor<T, N, index_t>(
      static_cast<typename PackedTensorAccessor<T, N, index_t>::PtrType>(t.data_ptr()),
      t.sizes().data(),
      t.strides().data());
}
template <typename T, size_t N> inline PackedTensorAccessor32<T, N> packed_accessor32(torch::stable::Tensor t) {
  return packed_accessor<T, N, int32_t>(t);
}
template <typename T, size_t N> inline PackedTensorAccessor64<T, N> packed_accessor64(torch::stable::Tensor t) {
  return packed_accessor<T, N, int64_t>(t);
}

inline cudaStream_t current_stream() {
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  // Available in PyTorch >= 2.13
  // return static_cast<cudaStream_t>(torch::stable::accelerator::getCurrentStream(device_idx).nativeHandle());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

inline int wavefront_threads(int64_t max_x, int64_t max_y, unsigned int num_blocks) {
  const int64_t max_diag = max_x < max_y ? max_x : max_y;
  const int capped_diag = max_diag > 1024 ? 1024 : max_diag < 1 ? 1 : static_cast<int>(max_diag);
  const int rounded_diag = ((capped_diag + TORCHDTW_WARP_SIZE - 1) / TORCHDTW_WARP_SIZE) * TORCHDTW_WARP_SIZE;
  const int num_threads = rounded_diag > 1024 ? 1024 : rounded_diag;
#ifdef USE_ROCM
  constexpr int64_t max_grid_threads = 4294967295; // 2^32 - 1
  STD_TORCH_CHECK(
      static_cast<int64_t>(num_blocks) * num_threads <= max_grid_threads,
      "Too many threads: the grid exceeds the HIP limit of 2^32-1 threads per dimension, got ",
      static_cast<int64_t>(num_blocks),
      " blocks of ",
      num_threads,
      " threads");
#endif
  return num_threads;
}

inline dim3 pair_grid(int64_t n1, int64_t n2) {
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  STD_TORCH_CHECK(n1 <= max_grid_x, "distances.size(0) exceeds the CUDA grid limit of 2^31-1");
  STD_TORCH_CHECK(n2 <= max_grid_x, "distances.size(1) exceeds the CUDA grid limit of 2^31-1");
  return dim3(static_cast<unsigned int>(n1), static_cast<unsigned int>(n2));
}

template <typename length_t = void> struct PairLengths {
  int32_t N;
  int32_t M;
  bool in_range;

  __device__ __forceinline__ PairLengths(const Lengths& sx, int64_t x, const Lengths& sy, int64_t y) {
    if constexpr (std::is_void_v<length_t>) {
      int64_t m;
      const int64_t n = visit_lengths_dtype(sx.data != nullptr ? sx.dtype : sy.dtype, [&](auto t) {
        using T = decltype(t);
        const T* const px = static_cast<const T*>(sx.data);
        const T* const py = static_cast<const T*>(sy.data);
        const T a = px != nullptr ? px[x * sx.stride] : T();
        const T b = py != nullptr ? py[y * sy.stride] : T();
        m = py != nullptr ? static_cast<int64_t>(b) : sy.fallback;
        return px != nullptr ? static_cast<int64_t>(a) : sx.fallback;
      });
      in_range = n <= sx.fallback && m <= sy.fallback && n + m - 1 <= MAX_PATH_LENGTH;
      N = static_cast<int32_t>(n);
      M = static_cast<int32_t>(m);
    } else {
      const length_t n = static_cast<const length_t*>(sx.data)[x * sx.stride];
      const length_t m = static_cast<const length_t*>(sy.data)[y * sy.stride];
      in_range = n <= static_cast<length_t>(sx.fallback) && m <= static_cast<length_t>(sy.fallback) &&
                 n + m - 1 <= MAX_PATH_LENGTH;
      N = static_cast<int32_t>(n);
      M = static_cast<int32_t>(m);
    }
  }

  __device__ __forceinline__ bool empty() const { return N <= 0 || M <= 0 || !in_range; }

  __device__ __forceinline__ ~PairLengths() {
    if (threadIdx.x == 0)
      CUDA_KERNEL_ASSERT_MSG(
          in_range,
          "sx and sy values must not exceed distances.size(-2) and distances.size(-1), and sx + sy - 1 must not "
          "exceed 65535");
  }

  PairLengths(const PairLengths&) = delete;
  PairLengths& operator=(const PairLengths&) = delete;
};

inline dim3 upper_triangle_grid(int64_t n, bool diagonal) {
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  const int64_t num_pairs = diagonal ? n * (n + 1) / 2 : n * (n - 1) / 2;
  STD_TORCH_CHECK(
      num_pairs <= max_grid_x, "symmetric batch too large: its number of pairs exceeds the CUDA grid limit");
  return dim3(static_cast<unsigned int>(num_pairs));
}

__device__ __forceinline__ void upper_triangle_pair(int32_t b, bool diagonal, int32_t& x, int32_t& y) {
  y = static_cast<int32_t>((1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(b))) / 2.0f);
  while (static_cast<int64_t>(y) * (y - 1) / 2 > b)
    y--;
  while (static_cast<int64_t>(y + 1) * y / 2 <= b)
    y++;
  x = static_cast<int32_t>(b - static_cast<int64_t>(y) * (y - 1) / 2);
  if (diagonal)
    y--;
}

template <typename F> void dispatch_index(bool needs_64bit, const F& f) {
  if (needs_64bit)
    f.template operator()<int64_t>();
  else
    f.template operator()<int32_t>();
}

inline void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what, " failed: ", cudaGetErrorString(err));
}

inline void check_launch(const char* kernel_name) { check_cuda(cudaGetLastError(), kernel_name); }

struct DiagonalStorage {
  size_t smem_size;      // Dynamic shared memory per block, 0 when the diagonals are in global memory.
  size_t scratch_stride; // Global memory per block, 0 when the diagonals are in shared memory.
};

inline DiagonalStorage plan_diagonals(size_t bytes) {
  if (bytes <= MAX_SHARED_BYTES)
    return {bytes, 0};
  return {0, (bytes + 15) & ~static_cast<size_t>(15)};
}

constexpr int64_t kMaxScratchBytes = int64_t{256} << 20;

template <typename Launch>
void launch_wavefront(const torch::stable::Tensor& like, const DiagonalStorage& storage, dim3 grid, Launch&& launch) {
  if (storage.scratch_stride == 0) {
    for (int64_t y0 = 0; y0 < grid.y; y0 += MAX_GRID_Y) {
      const dim3 chunk(grid.x, static_cast<unsigned int>(std::min<int64_t>(MAX_GRID_Y, grid.y - y0)));
      launch(std::false_type{}, chunk, storage.smem_size, nullptr, int64_t{0}, y0);
    }
    return;
  }
  const int64_t row_bytes =
      static_cast<int64_t>(std::min<int64_t>(grid.y, MAX_GRID_Y)) * static_cast<int64_t>(storage.scratch_stride);
  const int64_t rows = std::min<int64_t>(grid.x, std::max<int64_t>(1, kMaxScratchBytes / row_bytes));
  torch::stable::Tensor scratch =
      torch::stable::new_empty(like, {rows * row_bytes}, torch::headeronly::ScalarType::Byte);
  unsigned char* const scratch_ptr = reinterpret_cast<unsigned char*>(scratch.data_ptr());
  for (int64_t y0 = 0; y0 < grid.y; y0 += MAX_GRID_Y)
    for (int64_t x0 = 0; x0 < grid.x; x0 += rows) {
      const dim3 chunk(
          static_cast<unsigned int>(std::min<int64_t>(rows, grid.x - x0)),
          static_cast<unsigned int>(std::min<int64_t>(MAX_GRID_Y, grid.y - y0)));
      launch(std::true_type{}, chunk, size_t{0}, scratch_ptr, x0, y0);
    }
}

template <bool global>
__device__ __forceinline__ unsigned char* block_diagonals(unsigned char* scratch, size_t scratch_stride) {
  if constexpr (global) {
    return scratch + (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * scratch_stride;
  } else {
    extern __shared__ unsigned char smem[];
    return smem;
  }
}

__device__ __forceinline__ void
block_pair(bool symmetric, bool diagonal, int64_t x0, int64_t y0, int32_t& x, int32_t& y) {
  const int32_t row = static_cast<int32_t>(x0) + static_cast<int32_t>(blockIdx.x);
  if (symmetric) {
    upper_triangle_pair(row, diagonal, x, y);
  } else {
    x = row;
    y = static_cast<int32_t>(y0) + static_cast<int32_t>(blockIdx.y);
  }
}

struct WavefrontLaunch {
  dim3 num_blocks;
  int num_threads;
  size_t diagonal_bytes;
  bool needs_64bit;
};

inline bool needs_64bit_indexing(const torch::stable::Tensor& distances) {
  const torch::headeronly::IntHeaderOnlyArrayRef sizes = distances.sizes();
  const torch::headeronly::IntHeaderOnlyArrayRef strides = distances.strides();
  constexpr int64_t max_index = std::numeric_limits<int32_t>::max();
  if (sizes[0] * sizes[1] > max_index)
    return true;
  int64_t max_offset = 0;
  for (size_t d = 0; d < sizes.size(); d++) {
    if (sizes[d] == 0)
      return false;
    max_offset += (sizes[d] - 1) * strides[d];
  }
  return max_offset > max_index;
}

inline WavefrontLaunch plan_wavefront(
    const torch::stable::Tensor& distances, torch::headeronly::IntHeaderOnlyArrayRef sizes, bool symmetric,
    bool diagonal, size_t diagonal_bytes) {
  const int64_t n1 = sizes[0];
  const int64_t n2 = sizes[1];
  const int64_t s1 = sizes[2];
  const int64_t s2 = sizes[3];
  const dim3 num_blocks = symmetric ? upper_triangle_grid(n1, diagonal) : pair_grid(n1, n2);
  return {num_blocks, wavefront_threads(s1, s2, num_blocks.x), diagonal_bytes, needs_64bit_indexing(distances)};
}

template <typename Cell> __device__ __forceinline__ void wavefront(Cell& cell, int32_t N, int32_t M) {
  if (threadIdx.x == 0)
    cell.origin();
  __syncthreads();
  cell.rotate();

  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      cell.update(start_i - k, start_j + k);
    }
    __syncthreads();
    cell.rotate();
  }
}

} // namespace torchdtw
