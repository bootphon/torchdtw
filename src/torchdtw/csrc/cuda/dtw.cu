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
#include <type_traits>

// Default shared memory budget of 48KiB per block, shared by 3 cost diagonal buffers and
// 3 path-length diagonal buffers, each of length distances.size(3). The buffers are allocated
// dynamically at launch so the per-dtype capacity is 49152 / (3 * (sizeof(acc_t) + sizeof(uint16_t))):
// 1638 for double, 2730 for float/half/bfloat16. The cost buffers hold the accumulator type
// (acc_t<scalar_t>, float for half/bfloat16), not the storage type, so half/bfloat16 share float's
// capacity rather than getting the wider 4096 that 2-byte storage would allow.
#define MAX_SHARED_BYTES 49152

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

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

template <typename scalar_t>
using acc_t = std::conditional_t<
    std::is_same_v<scalar_t, torch::headeronly::Half> || std::is_same_v<scalar_t, torch::headeronly::BFloat16>, float,
    scalar_t>;

// Wavefront DP over anti-diagonals. Each cell tracks its cost and the length of the optimal path
// reaching it (one more than its chosen parent's path length). The cost of (N-1, M-1) divided by
// its path length is the final result, so no traceback is needed.
template <typename scalar_t, typename sx_t, typename index_t>
__global__ void dtw_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, const PackedTensorAccessor<scalar_t, 4, index_t> distances,
    const PackedTensorAccessor32<sx_t, 1> sx, const PackedTensorAccessor32<sx_t, 1> sy, bool symmetric) {
  int32_t x, y;
  if (symmetric) {
    const int32_t b = static_cast<int32_t>(blockIdx.x);
    y = static_cast<int32_t>((1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(b))) / 2.0f);
    while (static_cast<int64_t>(y) * (y - 1) / 2 > b)
      y--;
    while (static_cast<int64_t>(y + 1) * y / 2 <= b)
      y++;
    x = b - y * (y - 1) / 2;
  } else {
    x = blockIdx.x;
    y = blockIdx.y;
  }
  if (x >= out.size(0) || y >= out.size(1))
    return;
  const int32_t N = static_cast<int32_t>(sx[x]);
  const int32_t M = static_cast<int32_t>(sy[y]);
  if (N <= 0 || M <= 0) {
    if (threadIdx.x == 0) {
      out[x][y] = static_cast<scalar_t>(0);
      if (symmetric)
        out[y][x] = static_cast<scalar_t>(0);
    }
    return;
  }

  // Accumulate cost in acc_t<scalar_t> (float for half/bfloat16) to limit rounding drift.
  using acc_t = torchdtw::acc_t<scalar_t>;
  extern __shared__ unsigned char smem[];
  const int32_t buf_len = static_cast<int32_t>(distances.size(3));
  acc_t* const cost_smem = reinterpret_cast<acc_t*>(smem);
  // Round the cost buffers' size up to 2 bytes so the uint16_t buffers are aligned for 1-byte dtypes.
  const size_t len_offset = (3 * static_cast<size_t>(buf_len) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  uint16_t* const len_smem = reinterpret_cast<uint16_t*>(smem + len_offset);
  // Named pointers rotated by register swaps: a dynamically indexed pointer array would be
  // spilled to local memory and slow down every shared memory access in the wavefront loop.
  acc_t* cost_alpha = cost_smem;               // Last diagonal
  acc_t* cost_beta = cost_smem + buf_len;      // Second to last diagonal
  acc_t* cost_gamma = cost_smem + 2 * buf_len; // Buffer for the next diagonal
  uint16_t* len_alpha = len_smem;
  uint16_t* len_beta = len_smem + buf_len;
  uint16_t* len_gamma = len_smem + 2 * buf_len;

  const auto distances_xy = distances[x][y];

  if (threadIdx.x == 0) {
    cost_gamma[0] = static_cast<acc_t>(distances_xy[0][0]);
    len_gamma[0] = 1;
  }
  __syncthreads();
  acc_t* cost_temp = cost_beta;
  cost_beta = cost_alpha;
  cost_alpha = cost_gamma;
  cost_gamma = cost_temp;
  uint16_t* len_temp = len_beta;
  len_beta = len_alpha;
  len_alpha = len_gamma;
  len_gamma = len_temp;

  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      acc_t min_cost;
      uint16_t parent_len;
      // Boundary cells have a single valid parent. No sentinel cost is used for the invalid
      // parents: a real cost can reach the maximum of narrow integer dtypes and tie with it.
      if (i == 0) {
        min_cost = cost_alpha[j - 1];
        parent_len = len_alpha[j - 1];
      } else if (j == 0) {
        min_cost = cost_alpha[j];
        parent_len = len_alpha[j];
      } else {
        const acc_t c_up = cost_alpha[j];
        const acc_t c_left = cost_alpha[j - 1];
        const acc_t c_diag = cost_beta[j - 1];
        if (c_diag <= c_left && c_diag <= c_up) {
          min_cost = c_diag;
          parent_len = len_beta[j - 1];
        } else if (c_left <= c_up) {
          min_cost = c_left;
          parent_len = len_alpha[j - 1];
        } else {
          min_cost = c_up;
          parent_len = len_alpha[j];
        }
      }
      cost_gamma[j] = static_cast<acc_t>(distances_xy[i][j]) + min_cost;
      len_gamma[j] = static_cast<uint16_t>(parent_len + 1);
    }
    __syncthreads();

    acc_t* const cost_temp = cost_beta;
    cost_beta = cost_alpha;
    cost_alpha = cost_gamma;
    cost_gamma = cost_temp;
    uint16_t* const len_temp = len_beta;
    len_beta = len_alpha;
    len_alpha = len_gamma;
    len_gamma = len_temp;
  }

  if (threadIdx.x == 0) {
    const acc_t final_cost = cost_alpha[M - 1];
    const uint16_t path_len = len_alpha[M - 1];
    // For integral dtypes divide in int64 (as on CPU): casting the path length to a
    // narrow dtype can yield 0 (e.g. 256 as int8) and divide by zero.
    scalar_t result;
    if constexpr (std::is_integral_v<scalar_t>) {
      result = static_cast<scalar_t>(static_cast<int64_t>(final_cost) / static_cast<int64_t>(path_len));
    } else {
      result = static_cast<scalar_t>(final_cost / static_cast<acc_t>(path_len));
    }
    out[x][y] = result;
    if (symmetric)
      out[y][x] = result;
  }
}

template <typename distances_t, typename sx_t>
void dtw_batch_cuda_impl(Tensor& out, const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);
  const dim3 num_blocks = symmetric ? dim3(static_cast<unsigned int>(nx * (nx - 1) / 2)) : dim3(nx, ny);
  const int64_t max_diag = max_x < max_y ? max_x : max_y;
  const int capped_diag = max_diag > 1024 ? 1024 : static_cast<int>(max_diag);
  const int num_threads = ((capped_diag + 31) / 32) * 32 > 1024 ? 1024 : ((capped_diag + 31) / 32) * 32;
  const bool needs_64bit = nx * ny * max_x * max_y > std::numeric_limits<int32_t>::max();
  using acc_t = torchdtw::acc_t<distances_t>;
  const size_t cost_bytes = (3 * static_cast<size_t>(max_y) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  const size_t smem_size = cost_bytes + 3 * static_cast<size_t>(max_y) * sizeof(uint16_t);
  STD_TORCH_CHECK(
      smem_size <= MAX_SHARED_BYTES,
      "distances.size(3) > ",
      MAX_SHARED_BYTES / (3 * (sizeof(acc_t) + sizeof(uint16_t))),
      ": too large to use CUDA shared memory for this dtype");
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  // Available in PyTorch >= 2.13
  // cudaStream_t stream =
  //     static_cast<cudaStream_t>(torch::stable::accelerator::getCurrentStream(device_idx).nativeHandle());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  if (needs_64bit) {
    dtw_kernel<distances_t, sx_t, int64_t><<<num_blocks, num_threads, smem_size, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor64<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
  } else {
    dtw_kernel<distances_t, sx_t, int32_t><<<num_blocks, num_threads, smem_size, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor32<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric);
  }
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "dtw_kernel launch failed: ", cudaGetErrorString(err));
}

Tensor dtw_batch_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");

  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(sx.dim() == 1 && sy.dim() == 1, "sx and sy must be 1D tensors");
  STD_TORCH_CHECK(sx.is_cuda() && sy.is_cuda(), "sx and sy must be on CUDA");
  STD_TORCH_CHECK(
      sx.size(0) == nx && sy.size(0) == ny, "sx and sy sizes must match the first two dimensions of distances");
  STD_TORCH_CHECK(!symmetric || nx == ny, "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(
      max_x + max_y - 1 <= std::numeric_limits<uint16_t>::max(),
      "Sum of sequence lengths exceeds uint16_t path-length capacity");
  STD_TORCH_CHECK(sy.scalar_type() == sx.scalar_type(), "sx and sy dtypes do not match");

  Tensor out =
      symmetric ? torch::stable::new_zeros(distances, {nx, ny}) : torch::stable::new_empty(distances, {nx, ny});
  if (symmetric && nx <= 1)
    return out;
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_batch_cuda_impl",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            sx.scalar_type(),
            "dtw_batch_cuda_impl_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (dtw_batch_cuda_impl<distances_t, sx_t>(out, distances, sx, sy, symmetric));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

Tensor dtw_cuda(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  Tensor sxy = torch::stable::new_empty(distances, {2}, std::make_optional(torch::headeronly::ScalarType::Long));
  Tensor sx = torch::stable::narrow(sxy, 0, 0, 1);
  Tensor sy = torch::stable::narrow(sxy, 0, 1, 1);
  torch::stable::fill_(sx, distances.size(0));
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
