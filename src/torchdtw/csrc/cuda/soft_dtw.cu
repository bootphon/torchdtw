#include "../soft_dtw.h"
#include "common.cuh"
#include <limits>
#include <optional>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <tuple>

namespace torchdtw {

// R(i, j) of the forward pass, see soft_dtw.h. `R` points to the pair's R matrix (row stride `ld`) when it is
// kept for the backward pass, and is nullptr otherwise.
template <typename acc_t, typename distances_t, typename index_t> struct SoftForwardCell {
  distances_t distances;
  acc_t* R;
  index_t ld;
  int32_t nx;
  int32_t ny;
  int64_t bandwidth;
  acc_t gamma;
  acc_t inv_gamma;
  acc_t* r_alpha; // Last diagonal
  acc_t* r_beta;  // Second to last diagonal
  acc_t* r_gamma; // Buffer for the next diagonal

  __device__ __forceinline__ void store(int32_t i, int32_t j, acc_t value) {
    r_gamma[j] = value;
    if (R != nullptr)
      R[i * ld + j] = value;
  }

  __device__ __forceinline__ void origin() { store(0, 0, static_cast<acc_t>(distances[0][0])); }

  __device__ __forceinline__ void update(int32_t i, int32_t j) {
    acc_t value = infinity<acc_t>();
    if (in_band(i, j, nx, ny, bandwidth)) {
      acc_t parent;
      if (i == 0) {
        parent = r_alpha[j - 1];
      } else if (j == 0) {
        parent = r_alpha[j];
      } else {
        parent = softmin3(r_beta[j - 1], r_alpha[j - 1], r_alpha[j], gamma, inv_gamma);
      }
      value = static_cast<acc_t>(distances[i][j]) + parent;
    }
    store(i, j, value);
  }

  __device__ __forceinline__ void rotate() {
    acc_t* const temp = r_beta;
    r_beta = r_alpha;
    r_alpha = r_gamma;
    r_gamma = temp;
  }
};

// log E of the backward pass, on the flipped grid: the cell (fi, fj) is (i, j) = (nx - 1 - fi, ny - 1 - fj), and
// its flipped diag/left/up parents are its successors (i + 1, j + 1), (i, j + 1) and (i + 1, j). On the first
// flipped row and column, its single successor is (i, j + 1) and (i + 1, j) respectively. Each cell writes
// grad * E(i, j), the gradient of its distance, to the pair's `grad_distances` (row stride `ld`, as `R`).
template <typename acc_t, typename scalar_t, typename distances_t, typename index_t> struct SoftBackwardCell {
  distances_t distances;
  const acc_t* R;
  scalar_t* grad_distances;
  index_t ld;
  int32_t nx;
  int32_t ny;
  acc_t grad;
  acc_t inv_gamma;
  acc_t* e_alpha; // Last diagonal
  acc_t* e_beta;  // Second to last diagonal
  acc_t* e_gamma; // Buffer for the next diagonal

  __device__ __forceinline__ acc_t term(acc_t log_e_s, int32_t si, int32_t sj, acc_t r) const {
    return backward_term(log_e_s, R[si * ld + sj], static_cast<acc_t>(distances[si][sj]), r, inv_gamma);
  }

  __device__ __forceinline__ void store(int32_t i, int32_t j, int32_t fj, acc_t log_e) {
    e_gamma[fj] = log_e;
    grad_distances[i * ld + j] = static_cast<scalar_t>(grad * std::exp(log_e));
  }

  __device__ __forceinline__ void origin() { store(nx - 1, ny - 1, 0, 0); }

  __device__ __forceinline__ void update(int32_t fi, int32_t fj) {
    const int32_t i = nx - 1 - fi;
    const int32_t j = ny - 1 - fj;
    const acc_t r = R[i * ld + j];
    acc_t log_e = -infinity<acc_t>();
    if (r != infinity<acc_t>()) {
      if (fi == 0) {
        log_e = term(e_alpha[fj - 1], i, j + 1, r);
      } else if (fj == 0) {
        log_e = term(e_alpha[fj], i + 1, j, r);
      } else {
        log_e = logsumexp3(
            term(e_beta[fj - 1], i + 1, j + 1, r), term(e_alpha[fj - 1], i, j + 1, r), term(e_alpha[fj], i + 1, j, r));
      }
    }
    store(i, j, fj, log_e);
  }

  __device__ __forceinline__ void rotate() {
    acc_t* const temp = e_beta;
    e_beta = e_alpha;
    e_alpha = e_gamma;
    e_gamma = temp;
  }
};

// Writes `value` to the cells of a pair's (s1, s2) block outside its (N, M) sub-block, with the whole thread block.
template <typename T, typename index_t>
__device__ __forceinline__ void fill_padding(T* block, index_t s1, index_t s2, int32_t N, int32_t M, T value) {
  if (N == s1 && M == s2)
    return;
  for (index_t k = threadIdx.x; k < s1 * s2; k += blockDim.x) {
    if (k / s2 >= N || k % s2 >= M)
      block[k] = value;
  }
}

// One block per pair (x, y), see as_all_pairs, the launch covering rows [x0, x0 + gridDim.x) (see launch_wavefront).
template <typename scalar_t, typename sx_t, typename index_t, bool global>
__global__ void soft_dtw_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, acc_t<scalar_t>* R,
    const PackedTensorAccessor<scalar_t, 4, index_t> distances, const Lengths<sx_t> sx, const Lengths<sx_t> sy,
    bool paired, acc_t<scalar_t> gamma, int64_t bandwidth, unsigned char* scratch, size_t scratch_stride, int64_t x0) {
  using acc_t = torchdtw::acc_t<scalar_t>;
  const int32_t x = static_cast<int32_t>(x0) + static_cast<int32_t>(blockIdx.x);
  const int32_t y = blockIdx.y;
  const int32_t N = max(static_cast<int32_t>(sx[x]), 0);
  const int32_t M = max(static_cast<int32_t>(sy[paired ? x : y]), 0);
  const index_t s1 = distances.size(2);
  const index_t ld = distances.size(3);
  acc_t* const R_xy = R == nullptr ? nullptr : R + (static_cast<index_t>(x) * distances.size(1) + y) * s1 * ld;
  // The padding of the kept R is +inf, as no alignment reaches it.
  if (R_xy != nullptr)
    fill_padding(R_xy, s1, ld, N, M, infinity<acc_t>());
  if (N == 0 || M == 0) {
    if (threadIdx.x == 0)
      out[x][y] = static_cast<scalar_t>(0);
    return;
  }

  unsigned char* const smem = block_diagonals<global>(scratch, scratch_stride);
  const int32_t buf_len = static_cast<int32_t>(ld);
  acc_t* const r_smem = reinterpret_cast<acc_t*>(smem);
  const auto distances_xy = distances[x][y];
  SoftForwardCell<acc_t, decltype(distances_xy), index_t> cell{distances_xy,
                                                               R_xy,
                                                               ld,
                                                               N,
                                                               M,
                                                               bandwidth,
                                                               gamma,
                                                               static_cast<acc_t>(1) / gamma,
                                                               r_smem,
                                                               r_smem + buf_len,
                                                               r_smem + 2 * buf_len};
  wavefront(cell, N, M);

  if (threadIdx.x == 0)
    out[x][y] = static_cast<scalar_t>(cell.r_alpha[M - 1]);
}

template <typename scalar_t, typename sx_t, typename index_t, bool global>
__global__ void soft_dtw_backward_kernel(
    scalar_t* grad_distances, const PackedTensorAccessor32<scalar_t, 2> grad, const acc_t<scalar_t>* R,
    const PackedTensorAccessor<scalar_t, 4, index_t> distances, const Lengths<sx_t> sx, const Lengths<sx_t> sy,
    bool paired, acc_t<scalar_t> inv_gamma, unsigned char* scratch, size_t scratch_stride, int64_t x0) {
  using acc_t = torchdtw::acc_t<scalar_t>;
  const int32_t x = static_cast<int32_t>(x0) + static_cast<int32_t>(blockIdx.x);
  const int32_t y = blockIdx.y;
  const int32_t N = max(static_cast<int32_t>(sx[x]), 0);
  const int32_t M = max(static_cast<int32_t>(sy[paired ? x : y]), 0);
  const index_t s1 = distances.size(2);
  const index_t ld = distances.size(3);
  const index_t offset = (static_cast<index_t>(x) * distances.size(1) + y) * s1 * ld;
  scalar_t* const grad_xy = grad_distances + offset;

  // Padding beyond the pair's lengths gets no gradient.
  fill_padding(grad_xy, s1, ld, N, M, static_cast<scalar_t>(0));
  if (N == 0 || M == 0)
    return;

  unsigned char* const smem = block_diagonals<global>(scratch, scratch_stride);
  const int32_t buf_len = static_cast<int32_t>(ld);
  acc_t* const e_smem = reinterpret_cast<acc_t*>(smem);
  const auto distances_xy = distances[x][y];
  SoftBackwardCell<acc_t, scalar_t, decltype(distances_xy), index_t> cell{distances_xy,
                                                                          R + offset,
                                                                          grad_xy,
                                                                          ld,
                                                                          N,
                                                                          M,
                                                                          static_cast<acc_t>(grad[x][y]),
                                                                          inv_gamma,
                                                                          e_smem,
                                                                          e_smem + buf_len,
                                                                          e_smem + 2 * buf_len};
  wavefront(cell, N, M);
}

// Launch configuration shared by the forward and backward kernels: one block per pair, holding three diagonals of
// acc_t, of length distances.size(-1).
struct SoftLaunch {
  dim3 num_blocks;
  int num_threads;
  size_t diagonal_bytes;
  bool needs_64bit;
};

template <typename acc_t> SoftLaunch soft_dtw_launch(const Tensor& pairs) {
  const int64_t n1 = pairs.size(0);
  const int64_t n2 = pairs.size(1);
  const int64_t s1 = pairs.size(2);
  const int64_t s2 = pairs.size(3);
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  constexpr int64_t max_grid_y = 65535;
  STD_TORCH_CHECK(n1 <= max_grid_x, "distances.size(0) exceeds the CUDA grid limit of 2^31-1");
  STD_TORCH_CHECK(n2 <= max_grid_y, "distances.size(1) exceeds the CUDA grid limit of 65535");
  const dim3 num_blocks(static_cast<unsigned int>(n1), static_cast<unsigned int>(n2));
  return {num_blocks,
          wavefront_threads(s1, s2, num_blocks.x),
          3 * static_cast<size_t>(s2) * sizeof(acc_t),
          n1 * n2 * s1 * s2 > std::numeric_limits<int32_t>::max()};
}

template <typename distances_t, typename sx_t>
void soft_dtw_cuda_impl(
    Tensor& out, Tensor& R, bool keep_R, const Tensor& pairs, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, bool paired, double gamma, int64_t bandwidth) {
  using acc_t = torchdtw::acc_t<distances_t>;
  const SoftLaunch launch = soft_dtw_launch<acc_t>(pairs);
  const Lengths<sx_t> sx_l = make_lengths<sx_t>(sx, pairs.size(2));
  const Lengths<sx_t> sy_l = make_lengths<sx_t>(sy, pairs.size(3));
  acc_t* const R_data = keep_R ? reinterpret_cast<acc_t*>(R.data_ptr()) : nullptr;
  const DiagonalStorage storage = plan_diagonals(launch.diagonal_bytes);
  const cudaStream_t stream = current_stream();
  launch_wavefront(
      pairs, storage, launch.num_blocks, [&](auto global, dim3 grid, size_t smem, unsigned char* scratch, int64_t x0) {
        constexpr bool kGlobal = decltype(global)::value;
        if (launch.needs_64bit) {
          soft_dtw_kernel<distances_t, sx_t, int64_t, kGlobal><<<grid, launch.num_threads, smem, stream>>>(
              packed_accessor32<distances_t, 2>(out),
              R_data,
              packed_accessor64<distances_t, 4>(pairs),
              sx_l,
              sy_l,
              paired,
              static_cast<acc_t>(gamma),
              bandwidth,
              scratch,
              storage.scratch_stride,
              x0);
        } else {
          soft_dtw_kernel<distances_t, sx_t, int32_t, kGlobal><<<grid, launch.num_threads, smem, stream>>>(
              packed_accessor32<distances_t, 2>(out),
              R_data,
              packed_accessor32<distances_t, 4>(pairs),
              sx_l,
              sy_l,
              paired,
              static_cast<acc_t>(gamma),
              bandwidth,
              scratch,
              storage.scratch_stride,
              x0);
        }
        check_launch("soft_dtw_kernel");
      });
}

template <typename distances_t, typename sx_t>
void soft_dtw_backward_cuda_impl(
    Tensor& grad_distances, const Tensor& grad, const Tensor& pairs, const Tensor& R, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, bool paired, double gamma) {
  using acc_t = torchdtw::acc_t<distances_t>;
  const SoftLaunch launch = soft_dtw_launch<acc_t>(pairs);
  const Lengths<sx_t> sx_l = make_lengths<sx_t>(sx, pairs.size(2));
  const Lengths<sx_t> sy_l = make_lengths<sx_t>(sy, pairs.size(3));
  distances_t* const out = reinterpret_cast<distances_t*>(grad_distances.data_ptr());
  const acc_t* const R_data = reinterpret_cast<const acc_t*>(R.data_ptr());
  const acc_t inv_gamma = static_cast<acc_t>(1) / static_cast<acc_t>(gamma);
  const DiagonalStorage storage = plan_diagonals(launch.diagonal_bytes);
  const cudaStream_t stream = current_stream();
  launch_wavefront(
      pairs, storage, launch.num_blocks, [&](auto global, dim3 grid, size_t smem, unsigned char* scratch, int64_t x0) {
        constexpr bool kGlobal = decltype(global)::value;
        if (launch.needs_64bit) {
          soft_dtw_backward_kernel<distances_t, sx_t, int64_t, kGlobal><<<grid, launch.num_threads, smem, stream>>>(
              out,
              packed_accessor32<distances_t, 2>(grad),
              R_data,
              packed_accessor64<distances_t, 4>(pairs),
              sx_l,
              sy_l,
              paired,
              inv_gamma,
              scratch,
              storage.scratch_stride,
              x0);
        } else {
          soft_dtw_backward_kernel<distances_t, sx_t, int32_t, kGlobal><<<grid, launch.num_threads, smem, stream>>>(
              out,
              packed_accessor32<distances_t, 2>(grad),
              R_data,
              packed_accessor32<distances_t, 4>(pairs),
              sx_l,
              sy_l,
              paired,
              inv_gamma,
              scratch,
              storage.scratch_stride,
              x0);
        }
        check_launch("soft_dtw_backward_kernel");
      });
}

std::tuple<Tensor, Tensor> soft_dtw_cuda(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, double gamma,
    int64_t bandwidth, bool keep_R) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  Tensor out = torch::stable::new_empty(distances, {pairs.size(0), pairs.size(1)});
  const auto acc_dtype = acc_scalar_type(distances.scalar_type());
  Tensor R = keep_R ? torch::stable::new_empty(distances, distances.sizes(), acc_dtype)
                    : torch::stable::new_empty(distances, {0}, acc_dtype);
  if (pairs.numel() == 0) {
    // No cell to compute: every pair is empty.
    torch::stable::fill_(out, 0);
  } else {
    THO_DISPATCH_V2(
        distances.scalar_type(),
        "soft_dtw_cuda",
        AT_WRAP([&] {
          using distances_t = scalar_t;
          THO_DISPATCH_V2(
              lengths_dtype,
              "soft_dtw_cuda_2",
              AT_WRAP([&] {
                using sx_t = scalar_t;
                (soft_dtw_cuda_impl<distances_t, sx_t>(out, R, keep_R, pairs, sx, sy, paired, gamma, bandwidth));
              }),
              AT_INTEGRAL_TYPES_V2);
        }),
        AT_FLOATING_TYPES,
        torch::headeronly::ScalarType::Half,
        torch::headeronly::ScalarType::BFloat16);
  }
  if (paired)
    out = torch::stable::view(out, {pairs.size(0)});
  return {out, R};
}

Tensor soft_dtw_backward_cuda(
    const Tensor& grad, const Tensor& distances, const Tensor& R, const std::optional<Tensor>& sx,
    const std::optional<Tensor>& sy, double gamma) {
  const torch::headeronly::ScalarType lengths_dtype = check_soft_dtw_inputs(distances, sx, sy, gamma);
  check_soft_dtw_backward_inputs(grad, distances, R);
  STD_TORCH_CHECK(
      grad.is_cuda() && R.is_cuda() && grad.get_device() == distances.get_device() &&
          R.get_device() == distances.get_device(),
      "grad and R must be on the same CUDA device as distances");
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const bool paired = distances.dim() == 3;
  const Tensor pairs = as_all_pairs(distances);
  const Tensor grad_pairs = paired ? torch::stable::view(grad, {pairs.size(0), 1}) : grad;
  Tensor grad_distances = torch::stable::new_empty(distances, distances.sizes());
  if (pairs.numel() == 0)
    return grad_distances;
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "soft_dtw_backward_cuda",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            lengths_dtype,
            "soft_dtw_backward_cuda_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (soft_dtw_backward_cuda_impl<distances_t, sx_t>(
                  grad_distances, grad_pairs, pairs, R, sx, sy, paired, gamma));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_FLOATING_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return grad_distances;
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CUDA, m) {
  m.impl("soft_dtw", &TORCH_BOX(soft_dtw_cuda));
  m.impl("soft_dtw_backward", &TORCH_BOX(soft_dtw_backward_cuda));
}

} // namespace torchdtw
