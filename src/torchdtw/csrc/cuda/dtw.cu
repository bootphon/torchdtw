#include <Python.h>

#include "common.cuh"
#include <limits>
#include <optional>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <type_traits>

#ifdef USE_ROCM
#define TORCHDTW_MODULE_NAME "_C_rocm"
#define TORCHDTW_MODULE_INIT PyInit__C_rocm
#else
#define TORCHDTW_MODULE_NAME "_C_cuda"
#define TORCHDTW_MODULE_INIT PyInit__C_cuda
#endif

/* Creates a dummy empty _C_cuda (or _C_rocm) module, imported from Python only when
   PyTorch is built with CUDA (or ROCm). Keeping it separate from _C lets CPU-only builds
   of PyTorch load _C without libtorch_cuda, libtorch_hip or the GPU runtime. */
PyMODINIT_FUNC TORCHDTW_MODULE_INIT(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      TORCHDTW_MODULE_NAME, /* name of module */
      NULL,                 /* module documentation, may be NULL */
      -1,                   /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
      NULL,                 /* methods */
  };
  return PyModule_Create(&module_def);
}

namespace torchdtw {

// Each cell tracks its cost and the length of the optimal path reaching it (one more than its chosen
// parent's path length). The cost of (N-1, M-1) divided by its path length is the final result,
// so no traceback is needed.
template <typename acc_t, typename distances_t> struct HardCell {
  distances_t distances;
  // Named pointers rotated by register swaps: a dynamically indexed pointer array would be
  // spilled to local memory and slow down every shared memory access in the wavefront loop.
  acc_t* cost_alpha; // Last diagonal
  acc_t* cost_beta;  // Second to last diagonal
  acc_t* cost_gamma; // Buffer for the next diagonal
  uint16_t* len_alpha;
  uint16_t* len_beta;
  uint16_t* len_gamma;

  __device__ __forceinline__ void origin() {
    cost_gamma[0] = static_cast<acc_t>(distances[0][0]);
    len_gamma[0] = 1;
  }

  __device__ __forceinline__ void update(int32_t i, int32_t j) {
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
    cost_gamma[j] = static_cast<acc_t>(distances[i][j]) + min_cost;
    len_gamma[j] = static_cast<uint16_t>(parent_len + 1);
  }

  __device__ __forceinline__ void rotate() {
    acc_t* const cost_temp = cost_beta;
    cost_beta = cost_alpha;
    cost_alpha = cost_gamma;
    cost_gamma = cost_temp;
    uint16_t* const len_temp = len_beta;
    len_beta = len_alpha;
    len_alpha = len_gamma;
    len_gamma = len_temp;
  }
};

// One block per pair, the launch covering rows [x0, x0 + gridDim.x) of the full grid (see launch_wavefront).
template <typename scalar_t, typename sx_t, typename index_t, bool global>
__global__ void dtw_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, const PackedTensorAccessor<scalar_t, 4, index_t> distances,
    const Lengths<sx_t> sx, const Lengths<sx_t> sy, bool symmetric, unsigned char* scratch, size_t scratch_stride,
    int64_t x0) {
  const int32_t row = static_cast<int32_t>(x0) + static_cast<int32_t>(blockIdx.x);
  int32_t x, y;
  if (symmetric) {
    const int32_t b = row;
    y = static_cast<int32_t>((1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(b))) / 2.0f);
    while (static_cast<int64_t>(y) * (y - 1) / 2 > b)
      y--;
    while (static_cast<int64_t>(y + 1) * y / 2 <= b)
      y++;
    x = b - y * (y - 1) / 2;
  } else {
    x = row;
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
  unsigned char* const smem = block_diagonals<global>(scratch, scratch_stride);
  const int32_t buf_len = static_cast<int32_t>(distances.size(3));
  acc_t* const cost_smem = reinterpret_cast<acc_t*>(smem);
  // Round the cost buffers' size up to 2 bytes so the uint16_t buffers are aligned for 1-byte dtypes.
  const size_t len_offset = (3 * static_cast<size_t>(buf_len) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  uint16_t* const len_smem = reinterpret_cast<uint16_t*>(smem + len_offset);
  const auto distances_xy = distances[x][y];
  HardCell<acc_t, decltype(distances_xy)> cell{distances_xy,
                                               cost_smem,
                                               cost_smem + buf_len,
                                               cost_smem + 2 * buf_len,
                                               len_smem,
                                               len_smem + buf_len,
                                               len_smem + 2 * buf_len};
  wavefront(cell, N, M);

  if (threadIdx.x == 0) {
    const acc_t final_cost = cell.cost_alpha[M - 1];
    const uint16_t path_len = cell.len_alpha[M - 1];
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
void dtw_batch_cuda_impl(
    Tensor& out, const Tensor& distances, const Lengths<sx_t> sx, const Lengths<sx_t> sy, bool symmetric,
    dim3 num_blocks) {
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);
  const int num_threads = wavefront_threads(max_x, max_y, num_blocks.x);
  const bool needs_64bit = nx * ny * max_x * max_y > std::numeric_limits<int32_t>::max();
  using acc_t = torchdtw::acc_t<distances_t>;
  // The diagonals are 3 cost buffers and 3 path-length buffers, each of length distances.size(3). They fit
  // MAX_SHARED_BYTES up to 49152 / (3 * (sizeof(acc_t) + sizeof(uint16_t))) = 1638 for double, 2730 for
  // float/half/bfloat16 (whose costs accumulate in float): beyond, see plan_diagonals.
  const size_t cost_bytes = (3 * static_cast<size_t>(max_y) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  const size_t diagonal_bytes = cost_bytes + 3 * static_cast<size_t>(max_y) * sizeof(uint16_t);
  const DiagonalStorage storage = plan_diagonals(diagonal_bytes);
  const cudaStream_t stream = current_stream();

  launch_wavefront(
      distances, storage, num_blocks, [&](auto global, dim3 grid, size_t smem, unsigned char* scratch, int64_t x0) {
        constexpr bool kGlobal = decltype(global)::value;
        dispatch_index(needs_64bit, [&]<typename index_t>() {
          dtw_kernel<distances_t, sx_t, index_t, kGlobal><<<grid, num_threads, smem, stream>>>(
              packed_accessor32<distances_t, 2>(out),
              packed_accessor<distances_t, 4, index_t>(distances),
              sx,
              sy,
              symmetric,
              scratch,
              storage.scratch_stride,
              x0);
        });
        check_launch("dtw_kernel");
      });
}

// DTW of each pair of the 4D distances, with lengths sx and sy, which default to the padded sizes when absent.
Tensor dtw_pairs_cuda(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, bool symmetric,
    torch::headeronly::ScalarType lengths_dtype) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);
  STD_TORCH_CHECK(!symmetric || nx == ny, "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(
      max_x + max_y - 1 <= std::numeric_limits<uint16_t>::max(),
      "Sum of sequence lengths exceeds uint16_t path-length capacity");
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  STD_TORCH_CHECK(
      !symmetric || nx * (nx - 1) / 2 <= max_grid_x,
      "symmetric dtw_batch too large: nx*(nx-1)/2 exceeds the CUDA grid limit");
  const dim3 num_blocks = symmetric ? dim3(static_cast<unsigned int>(nx * (nx - 1) / 2)) : pair_grid(nx, ny);

  Tensor out =
      symmetric ? torch::stable::new_zeros(distances, {nx, ny}) : torch::stable::new_empty(distances, {nx, ny});
  if (symmetric && nx <= 1)
    return out;
  dispatch_dtw(distances.scalar_type(), lengths_dtype, [&]<typename distances_t, typename sx_t>() {
    dtw_batch_cuda_impl<distances_t, sx_t>(
        out, distances, make_lengths<sx_t>(sx, max_x), make_lengths<sx_t>(sy, max_y), symmetric, num_blocks);
  });
  return out;
}

Tensor dtw_batch_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");
  STD_TORCH_CHECK(sx.dim() == 1 && sy.dim() == 1, "sx and sy must be 1D tensors");
  STD_TORCH_CHECK(sx.is_cuda() && sy.is_cuda(), "sx and sy must be on CUDA");
  STD_TORCH_CHECK(
      sx.get_device() == distances.get_device() && sy.get_device() == distances.get_device(),
      "sx and sy must be on the same CUDA device as distances");
  STD_TORCH_CHECK(
      sx.size(0) == distances.size(0) && sy.size(0) == distances.size(1),
      "sx and sy sizes must match the first two dimensions of distances");
  STD_TORCH_CHECK(sy.scalar_type() == sx.scalar_type(), "sx and sy dtypes do not match");
  return dtw_pairs_cuda(distances, sx, sy, symmetric, sx.scalar_type());
}

Tensor dtw_cuda(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const Tensor pairs = torch::stable::view(distances, {1, 1, distances.size(0), distances.size(1)});
  const Tensor result = dtw_pairs_cuda(pairs, std::nullopt, std::nullopt, false, torch::headeronly::ScalarType::Long);
  return torch::stable::view(result, torch::headeronly::IntHeaderOnlyArrayRef());
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CUDA, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cuda));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_cuda));
}

} // namespace torchdtw
