#pragma once

#include "../common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

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
template <typename T, size_t N> inline PackedTensorAccessor32<T, N> packed_accessor32(torch::stable::Tensor t) {
  return PackedTensorAccessor32<T, N>(
      static_cast<typename PackedTensorAccessor32<T, N>::PtrType>(t.data_ptr()), t.sizes().data(), t.strides().data());
}
template <typename T, size_t N> inline PackedTensorAccessor64<T, N> packed_accessor64(torch::stable::Tensor t) {
  return PackedTensorAccessor64<T, N>(
      static_cast<typename PackedTensorAccessor64<T, N>::PtrType>(t.data_ptr()), t.sizes().data(), t.strides().data());
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
      "dtw_batch too large: the number of threads exceeds the HIP grid limit of 2^32-1, got ",
      static_cast<int64_t>(num_blocks),
      " blocks of ",
      num_threads,
      " threads");
#endif
  return num_threads;
}

inline void check_launch(const char* kernel_name) {
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, kernel_name, " launch failed: ", cudaGetErrorString(err));
}

// Wavefront DP over the anti-diagonals of an N x M grid, run by a whole block. `cell` owns three diagonal
// buffers in shared memory, indexed by column: alpha (last diagonal), beta (second to last) and gamma (next).
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
