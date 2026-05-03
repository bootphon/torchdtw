#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>

// Shared memory has a size of 48kB, shared by 3 diagonal buffers.
#define SHARED_MEM_SIZE 49152

namespace torchdtw {

using torch::stable::Tensor;

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

// Trace directions: 0 = diag (i-1, j-1), 1 = left (i, j-1), 2 = up (i-1, j).
template <typename scalar_t, typename sx_t, typename index_t>
__global__ void dtw_wavefront_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, PackedTensorAccessor<int8_t, 4, index_t> trace,
    const PackedTensorAccessor<scalar_t, 4, index_t> distances, const PackedTensorAccessor32<sx_t, 1> sx,
    const PackedTensorAccessor32<sx_t, 1> sy, bool symmetric) {
  const int32_t x = blockIdx.x;
  const int32_t y = blockIdx.y;
  if (x >= trace.size(0) || y >= trace.size(1))
    return;
  if (symmetric && x >= y)
    return;
  const int32_t N = static_cast<int32_t>(sx[x]);
  const int32_t M = static_cast<int32_t>(sy[y]);

  constexpr int32_t max_diag_len = SHARED_MEM_SIZE / (3 * sizeof(scalar_t));
  __shared__ scalar_t buffers[3][max_diag_len];
  int32_t alpha = 0; // Last diagonal
  int32_t beta = 1;  // Second to last diagonal
  int32_t gamma = 2; // Buffer for the last diagonal

  auto trace_xy = trace[x][y];
  const auto distances_xy = distances[x][y];

  if (threadIdx.x == 0) {
    const scalar_t c00 = distances_xy[0][0];
    buffers[gamma][0] = c00;
    if (N == 1 && M == 1)
      out[x][y] = c00;
  }
  __syncthreads();
  {
    const int32_t temp = beta;
    beta = alpha;
    alpha = gamma;
    gamma = temp;
  }

  const scalar_t max_val = std::numeric_limits<scalar_t>::max();
  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      const scalar_t c_up = (i > 0) ? buffers[alpha][j] : max_val;
      const scalar_t c_left = (j > 0) ? buffers[alpha][j - 1] : max_val;
      const scalar_t c_diag = (i > 0 && j > 0) ? buffers[beta][j - 1] : max_val;
      int8_t direction;
      scalar_t min_cost;
      if (c_diag <= c_left && c_diag <= c_up) {
        direction = 0;
        min_cost = c_diag;
      } else if (c_left <= c_up) {
        direction = 1;
        min_cost = c_left;
      } else {
        direction = 2;
        min_cost = c_up;
      }
      const scalar_t cij = distances_xy[i][j] + min_cost;
      trace_xy[i][j] = direction;
      buffers[gamma][j] = cij;
      if (i == N - 1 && j == M - 1)
        out[x][y] = cij;
    }
    __syncthreads();

    const int32_t temp = beta;
    beta = alpha;
    alpha = gamma;
    gamma = temp;
  }
}

template <typename scalar_t, typename sx_t, typename index_t>
__global__ void dtw_backtrack_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, const PackedTensorAccessor<int8_t, 4, index_t> trace,
    const PackedTensorAccessor32<sx_t, 1> sx, const PackedTensorAccessor32<sx_t, 1> sy, bool symmetric) {
  const int32_t x = blockIdx.x;
  const int32_t y = blockIdx.y;
  if (x >= trace.size(0) || y >= trace.size(1))
    return;
  if (symmetric && x >= y)
    return;
  const int32_t N = static_cast<int32_t>(sx[x]);
  const int32_t M = static_cast<int32_t>(sy[y]);

  const auto trace_xy = trace[x][y];
  int32_t path_len = 1;
  int32_t i = N - 1;
  int32_t j = M - 1;
  while (i > 0 && j > 0) {
    const int8_t d = trace_xy[i][j];
    if (d == 0) {
      i--;
      j--;
    } else if (d == 1) {
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

  out[x][y] = out[x][y] / static_cast<scalar_t>(path_len);
  if (symmetric)
    out[y][x] = out[x][y];
}

template <typename distances_t, typename sx_t>
void dtw_batch_cuda_impl(
    Tensor& out, Tensor& trace, const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric,
    dim3 num_blocks, int num_threads, cudaStream_t stream, bool needs_64bit) {
  STD_TORCH_CHECK(
      sy.scalar_type() == torch::headeronly::CppTypeToScalarType<sx_t>::value, "sy dtype does not match sx dtype");
  if (needs_64bit) {
    dtw_wavefront_kernel<distances_t, sx_t, int64_t><<<num_blocks, num_threads, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor64<int8_t, 4>(trace),
        packed_accessor64<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
    dtw_backtrack_kernel<distances_t, sx_t, int64_t><<<num_blocks, 1, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor64<int8_t, 4>(trace),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
  } else {
    dtw_wavefront_kernel<distances_t, sx_t, int32_t><<<num_blocks, num_threads, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor32<int8_t, 4>(trace),
        packed_accessor32<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
    dtw_backtrack_kernel<distances_t, sx_t, int32_t><<<num_blocks, 1, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor32<int8_t, 4>(trace),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
  }
}

Tensor dtw_batch_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");

  Tensor trace = torch::stable::new_empty(
      distances, {nx, ny, max_x, max_y}, std::make_optional(torch::headeronly::ScalarType::Char));
  Tensor out = torch::stable::new_zeros(distances, {nx, ny});

  const dim3 num_blocks(nx, ny);
  const int num_threads = max_x > 1024 ? 1024 : max_x;
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
  const bool needs_64bit = nx * ny * max_x * max_y > std::numeric_limits<int32_t>::max();

  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_batch_cuda_impl",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        constexpr int64_t max_diag_len = SHARED_MEM_SIZE / (3 * sizeof(distances_t));
        STD_TORCH_CHECK(max_y <= max_diag_len, "Diagonal too large to use CUDA shared memory");
        THO_DISPATCH_V2(
            sx.scalar_type(),
            "dtw_batch_cuda_impl_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (dtw_batch_cuda_impl<distances_t, sx_t>(
                  out, trace, distances, sx, sy, symmetric, num_blocks, num_threads, stream, needs_64bit));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
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
