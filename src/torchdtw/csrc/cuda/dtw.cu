#include <Python.h>

#include "../dtw.h"
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

template <typename acc_t, typename distances_t> struct HardCell {
  distances_t distances;
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

template <typename scalar_t, typename index_t, bool global, typename length_t>
__global__ void dtw_kernel(
    PackedTensorAccessor<scalar_t, 2, index_t> out, const PackedTensorAccessor<scalar_t, 4, index_t> distances,
    const Lengths sx, const Lengths sy, bool symmetric, unsigned char* scratch, size_t scratch_stride, int64_t x0,
    int64_t y0) {
  int32_t x, y;
  block_pair(symmetric, false, x0, y0, x, y);
  if (x >= out.size(0) || y >= out.size(1))
    return;
  const PairLengths<length_t> lengths(sx, x, sy, y);
  const int32_t N = lengths.N;
  const int32_t M = lengths.M;
  if (lengths.empty()) {
    if (threadIdx.x == 0) {
      out[x][y] = static_cast<scalar_t>(0);
      if (symmetric)
        out[y][x] = static_cast<scalar_t>(0);
    }
    return;
  }

  using acc_t = torchdtw::acc_t<scalar_t>;
  unsigned char* const smem = block_diagonals<global>(scratch, scratch_stride);
  const int32_t buf_len = static_cast<int32_t>(distances.size(3));
  acc_t* const cost_smem = reinterpret_cast<acc_t*>(smem);
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
    const scalar_t result = normalized_cost<scalar_t>(cell.cost_alpha[M - 1], cell.len_alpha[M - 1]);
    out[x][y] = result;
    if (symmetric)
      out[y][x] = result;
  }
}

template <typename distances_t>
void dtw_batch_cuda_impl(
    Tensor& out, const Tensor& distances, const Lengths sx, const Lengths sy, bool symmetric,
    const WavefrontLaunch& launch) {
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const DiagonalStorage storage = plan_diagonals(launch.diagonal_bytes);
  const cudaStream_t stream = current_stream();
  const bool typed = sx.data != nullptr && sy.data != nullptr && !launch.needs_64bit;
  const bool int32_lengths = typed && sx.dtype == torch::headeronly::ScalarType::Int;
  const bool int64_lengths = typed && sx.dtype == torch::headeronly::ScalarType::Long;

  launch_wavefront(
      distances,
      storage,
      launch.num_blocks,
      [&](auto global, dim3 grid, size_t smem, unsigned char* scratch, int64_t x0, int64_t y0) {
        constexpr bool kGlobal = decltype(global)::value;
        auto launch_kernel = [&]<typename index_t, typename length_t>() {
          dtw_kernel<distances_t, index_t, kGlobal, length_t><<<grid, launch.num_threads, smem, stream>>>(
              packed_accessor<distances_t, 2, index_t>(out),
              packed_accessor<distances_t, 4, index_t>(distances),
              sx,
              sy,
              symmetric,
              scratch,
              storage.scratch_stride,
              x0,
              y0);
        };
        if constexpr (!kGlobal) {
          if (int32_lengths)
            launch_kernel.template operator()<int32_t, int32_t>();
          else if (int64_lengths)
            launch_kernel.template operator()<int32_t, int64_t>();
          else
            dispatch_index(
                launch.needs_64bit, [&]<typename index_t>() { launch_kernel.template operator()<index_t, void>(); });
        } else {
          dispatch_index(
              launch.needs_64bit, [&]<typename index_t>() { launch_kernel.template operator()<index_t, void>(); });
        }
        check_launch("dtw_kernel");
      });
}

static Tensor dtw_batch_cuda(const Tensor& distances, const Tensor* sx, const Tensor* sy, bool symmetric) {
  return dtw_batch_op(
      distances,
      sx,
      sy,
      symmetric,
      [&]<typename distances_t>(
          Tensor& out, torch::headeronly::IntHeaderOnlyArrayRef sizes, const Lengths& sx_l, const Lengths& sy_l) {
        const int64_t n1 = sizes[0];
        const int64_t n2 = sizes[1];
        const int64_t s1 = sizes[2];
        const int64_t s2 = sizes[3];
        STD_TORCH_CHECK(
            sx_l.data != nullptr || sy_l.data != nullptr || s1 + s2 - 1 <= MAX_PATH_LENGTH,
            "Sum of sequence lengths exceeds the CUDA path-length capacity: s1 + s2 - 1 must not exceed 65535");
        if (n1 == 0 || n2 == 0 || (symmetric && n1 == 1))
          return;
        using acc_t = torchdtw::acc_t<distances_t>;
        const size_t cost_bytes = (3 * static_cast<size_t>(s2) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
        const size_t diagonal_bytes = cost_bytes + 3 * static_cast<size_t>(s2) * sizeof(uint16_t);
        dtw_batch_cuda_impl<distances_t>(
            out, distances, sx_l, sy_l, symmetric, plan_wavefront(distances, sizes, symmetric, false, diagonal_bytes));
      });
}

Tensor dtw_batch_lengths_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  return dtw_batch_cuda(distances, &sx, &sy, symmetric);
}

Tensor dtw_batch_optional_lengths_cuda(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, bool symmetric) {
  return dtw_batch_cuda(distances, optional_ptr(sx), optional_ptr(sy), symmetric);
}

Tensor dtw_cuda(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  const Tensor result = dtw_batch_cuda(torch::stable::view(distances, {1, 1, N, M}), nullptr, nullptr, false);
  return torch::stable::view(result, torch::headeronly::IntHeaderOnlyArrayRef());
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CUDA, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cuda));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_lengths_cuda));
  m.impl("dtw_batch.optional_lengths", &TORCH_BOX(dtw_batch_optional_lengths_cuda));
}

} // namespace torchdtw
