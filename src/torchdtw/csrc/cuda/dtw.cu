#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>

// Shared memory has a size of 48kB
// Maximum diagonal length is N such that N * 3 * sizeof(float) = 48kB
#define MAX_DIAG_LEN 4096

namespace torchdtw {

using torch::stable::Tensor;
template <typename T, size_t N>
using PackedTensorAccessor32 =
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, torch::headeronly::RestrictPtrTraits, int32_t>;
template <typename T, size_t N> inline PackedTensorAccessor32<T, N> packed_accessor32(torch::stable::Tensor t) {
  return PackedTensorAccessor32<T, N>(static_cast<typename PackedTensorAccessor32<T, N>::PtrType>(t.data_ptr()),
                                      t.sizes().data(), t.strides().data());
}

__global__ void dtw_wavefront_kernel(PackedTensorAccessor32<float, 4> cost,
                                     const PackedTensorAccessor32<float, 4> distances,
                                     const PackedTensorAccessor32<int64_t, 1> sx,
                                     const PackedTensorAccessor32<int64_t, 1> sy, bool symmetric) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost.size(0) || y >= cost.size(1))
    return;
  if (symmetric && x >= y)
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];

  __shared__ float buffers[3][MAX_DIAG_LEN];
  int alpha = 0; // Last diagonal
  int beta = 1;  // Second to last diagonal
  int gamma = 2; // Buffer for the last diagonal

  for (int64_t diag = 0; diag <= N + M - 1; diag++) {
    const int64_t start_i = min(diag, N - 1);
    const int64_t start_j = max(int64_t(0), diag - start_i);
    const int64_t length = start_i - max(int64_t(0), diag - M + 1) + 1;

    for (int k = threadIdx.x; k < length; k += blockDim.x) {
      const int64_t i = start_i - k;
      const int64_t j = start_j + k;
      const float c_up = (i > 0) ? buffers[alpha][j] : FLT_MAX;
      const float c_left = (j > 0) ? buffers[alpha][j - 1] : FLT_MAX;
      const float c_diag = (i > 0 && j > 0) ? buffers[beta][j - 1] : FLT_MAX;
      const float min_cost = (i == 0 && j == 0) ? 0 : fminf(c_left, fminf(c_diag, c_up));
      const float cij = distances[x][y][i][j] + min_cost;
      cost[x][y][i][j] = cij;
      buffers[gamma][j] = cij;
    }
    __syncthreads();

    int temp = beta;
    beta = alpha;
    alpha = gamma;
    gamma = temp;
  }
}

__global__ void dtw_backtrack_kernel(PackedTensorAccessor32<float, 2> out, const PackedTensorAccessor32<float, 4> cost,
                                     const PackedTensorAccessor32<int64_t, 1> sx,
                                     const PackedTensorAccessor32<int64_t, 1> sy, bool symmetric) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost.size(0) || y >= cost.size(1))
    return;
  if (symmetric && x >= y)
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = cost[x][y][i - 1][j];
    const float c_left = cost[x][y][i][j - 1];
    const float c_diag = cost[x][y][i - 1][j - 1];
    if (c_diag <= c_left && c_diag <= c_up) {
      i--;
      j--;
    } else if (c_left <= c_up) {
      j--;
    } else {
      i--;
    }
    path_len++;
  }
  if (i == 0)
    path_len += j;
  if (j == 0)
    path_len += i;

  out[x][y] = cost[x][y][N - 1][M - 1] / path_len;
  if (symmetric)
    out[y][x] = out[x][y];
}

Tensor dtw_batch_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(max_x < MAX_DIAG_LEN, "Diagonal too large to use CUDA shared memory");

  Tensor cost = torch::stable::new_zeros(distances, {nx, ny, max_x, max_y});
  Tensor out = torch::stable::new_zeros(distances, {nx, ny});

  const dim3 num_blocks(nx, ny);
  const int num_threads = max_x > 1024 ? 1024 : max_x;
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();

  dtw_wavefront_kernel<<<num_blocks, num_threads, 0, stream>>>(
      packed_accessor32<float, 4>(cost), packed_accessor32<float, 4>(distances), packed_accessor32<int64_t, 1>(sx),
      packed_accessor32<int64_t, 1>(sy), symmetric);
  dtw_backtrack_kernel<<<num_blocks, 1, 0, stream>>>(
      packed_accessor32<float, 2>(out), packed_accessor32<float, 4>(cost), packed_accessor32<int64_t, 1>(sx),
      packed_accessor32<int64_t, 1>(sy), symmetric);
  return out;
}

Tensor dtw_cuda(const Tensor& distances) {
  Tensor sx = torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sx, distances.size(0));
  Tensor sy = torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sy, distances.size(1));
  Tensor result =
      dtw_batch_cuda(torch::stable::view(distances, {1, 1, distances.size(0), distances.size(1)}), sx, sy, false);
  return torch::stable::view(result, {});
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CUDA, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cuda));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_cuda));
}

} // namespace torchdtw
