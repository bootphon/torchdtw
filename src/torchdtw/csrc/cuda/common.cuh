#pragma once

#include "../common.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <type_traits>

// Default shared memory budget of 48KiB per block, available without opting in to a larger carveout.
#define MAX_SHARED_BYTES 49152

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

// The current PyTorch stream on the current device, so that kernels are ordered with PyTorch's own work.
inline cudaStream_t current_stream() {
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  // Available in PyTorch >= 2.13
  // return static_cast<cudaStream_t>(torch::stable::accelerator::getCurrentStream(device_idx).nativeHandle());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

// Threads per block for a wavefront over grids of up to max_x x max_y cells: one per cell of the longest
// anti-diagonal, rounded up to a whole warp and capped at 1024.
inline int wavefront_threads(int64_t max_x, int64_t max_y, unsigned int num_blocks) {
  const int64_t max_diag = max_x < max_y ? max_x : max_y;
  const int capped_diag = max_diag > 1024 ? 1024 : static_cast<int>(max_diag);
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

// Grid of a kernel with one block per pair (x, y) of an (n1, n2) batch.
inline dim3 pair_grid(int64_t n1, int64_t n2) {
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  constexpr int64_t max_grid_y = 65535;
  STD_TORCH_CHECK(n1 <= max_grid_x, "distances.size(0) exceeds the CUDA grid limit of 2^31-1");
  STD_TORCH_CHECK(n2 <= max_grid_y, "distances.size(1) exceeds the CUDA grid limit of 65535");
  return dim3(static_cast<unsigned int>(n1), static_cast<unsigned int>(n2));
}

// Calls f.template operator()<index_t>() with the index type of the distances: int64_t if needs_64bit, else int32_t.
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

// Where the blocks of a wavefront kernel keep their three diagonals.
struct DiagonalStorage {
  size_t smem_size;      // Dynamic shared memory per block, 0 when the diagonals are in global memory.
  size_t scratch_stride; // Global memory per block, 0 when the diagonals are in shared memory.
};

// Places `bytes` of diagonals per block: in shared memory up to MAX_SHARED_BYTES, in global memory beyond, which
// leaves the sequence lengths unbounded. Opting in to the device's larger shared memory was measured up to 2x slower
// than global memory near its limit: it leaves too little L1 cache for the reads of the distances.
inline DiagonalStorage plan_diagonals(size_t bytes) {
  if (bytes <= MAX_SHARED_BYTES)
    return {bytes, 0};
  return {0, (bytes + 15) & ~static_cast<size_t>(15)};
}

// Global memory scratch of the diagonals, per launch.
constexpr int64_t kMaxScratchBytes = int64_t{256} << 20;

// Launches a wavefront kernel over `grid` (with grid.z == 1), calling launch(global, grid, smem_size, scratch, x0):
// `global` is std::true_type when the diagonals are in global memory, and the launch covers the blocks of rows
// [x0, x0 + grid.x) of the full grid. In global memory, the rows are split across launches so that their scratch,
// storage.scratch_stride bytes per block from `scratch`, stays within kMaxScratchBytes.
template <typename Launch>
void launch_wavefront(const torch::stable::Tensor& like, const DiagonalStorage& storage, dim3 grid, Launch&& launch) {
  if (storage.scratch_stride == 0) {
    launch(std::false_type{}, grid, storage.smem_size, nullptr, int64_t{0});
    return;
  }
  const int64_t row_bytes = static_cast<int64_t>(grid.y) * static_cast<int64_t>(storage.scratch_stride);
  const int64_t rows = std::min<int64_t>(grid.x, std::max<int64_t>(1, kMaxScratchBytes / row_bytes));
  torch::stable::Tensor scratch =
      torch::stable::new_empty(like, {rows * row_bytes}, torch::headeronly::ScalarType::Byte);
  unsigned char* const scratch_ptr = reinterpret_cast<unsigned char*>(scratch.data_ptr());
  for (int64_t x0 = 0; x0 < grid.x; x0 += rows) {
    const dim3 chunk(static_cast<unsigned int>(std::min<int64_t>(rows, grid.x - x0)), grid.y);
    launch(std::true_type{}, chunk, size_t{0}, scratch_ptr, x0);
  }
}

// The block's diagonals: dynamic shared memory, or its slice of the global memory scratch.
template <bool global>
__device__ __forceinline__ unsigned char* block_diagonals(unsigned char* scratch, size_t scratch_stride) {
  if constexpr (global) {
    return scratch + (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * scratch_stride;
  } else {
    extern __shared__ unsigned char smem[];
    return smem;
  }
}

// Wavefront DP over the anti-diagonals of an N x M grid, run by a whole block. `cell` owns three diagonal
// buffers, from block_diagonals, indexed by column: alpha (last diagonal), beta (second to last) and gamma (next).
// origin() writes the cell (0, 0) into gamma, and update(i, j) writes the cell (i, j) into gamma from its parents:
// alpha[j - 1] alone if i == 0, alpha[j] alone if j == 0, and otherwise alpha[j] (up), alpha[j - 1] (left) and
// beta[j - 1] (diag). rotate() then shifts gamma -> alpha -> beta -> gamma. On return, the cell (N - 1, M - 1) is
// at index M - 1 of alpha.
// Unlike fill_grid on CPU, the edge cases are left to update(): with separate hooks per case, each one issues its
// own global and shared memory loads instead of sharing them after the parent selection, which measured ~20% slower.
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
