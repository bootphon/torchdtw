#include <Python.h>
#include <algorithm>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>
#include <type_traits>
#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the STABLE_TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace torchdtw {

using torch::stable::Tensor;
template <typename T, size_t N> using TensorAccessor = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;
template <typename T, size_t N> inline TensorAccessor<T, N> accessor(Tensor t) {
  return TensorAccessor<T, N>(reinterpret_cast<T*>(t.data_ptr()), t.sizes().data(), t.strides().data());
}

template <typename scalar_t>
using acc_t = std::conditional_t<
    std::is_same_v<scalar_t, torch::headeronly::Half> || std::is_same_v<scalar_t, torch::headeronly::BFloat16>, float,
    scalar_t>;

template <typename scalar_t>
static void
compute_dtw_cost(const TensorAccessor<const scalar_t, 2>& distances_a, int64_t N, int64_t M, acc_t<scalar_t>* cost) {
  using acc = acc_t<scalar_t>;
  cost[0] = static_cast<acc>(distances_a[0][0]);
  for (int64_t i = 1; i < N; i++) {
    cost[i * M] = static_cast<acc>(distances_a[i][0]) + cost[(i - 1) * M];
  }
  for (int64_t j = 1; j < M; j++) {
    cost[j] = static_cast<acc>(distances_a[0][j]) + cost[j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    const auto distances_i = distances_a[i];
    acc* row = cost + i * M;
    const acc* prev = row - M;
    for (int64_t j = 1; j < M; j++) {
      row[j] = static_cast<acc>(distances_i[j]) + std::min({prev[j], prev[j - 1], row[j - 1]});
    }
  }
}

template <typename cost_t>
static std::vector<std::pair<int64_t, int64_t>> compute_dtw_path(const cost_t* cost, int64_t N, int64_t M) {
  std::vector<std::pair<int64_t, int64_t>> path;
  path.reserve(N + M - 1);
  int64_t i = N - 1;
  int64_t j = M - 1;
  path.push_back({i, j});
  while (i > 0 && j > 0) {
    const cost_t c_up = cost[(i - 1) * M + j];
    const cost_t c_left = cost[i * M + j - 1];
    const cost_t c_diag = cost[(i - 1) * M + j - 1];
    if (c_diag <= c_left && c_diag <= c_up) {
      i--;
      j--;
    } else if (c_left <= c_up) {
      j--;
    } else {
      i--;
    }
    path.push_back({i, j});
  }
  while (i > 0) {
    i--;
    path.push_back({i, j});
  }
  while (j > 0) {
    j--;
    path.push_back({i, j});
  }
  std::reverse(path.begin(), path.end());
  return path;
}

template <typename scalar_t>
static scalar_t compute_dtw_value(
    const TensorAccessor<const scalar_t, 2>& distances_a, int64_t N, int64_t M,
    std::vector<acc_t<scalar_t>>& workspace) {
  using acc = acc_t<scalar_t>;
  if (static_cast<int64_t>(workspace.size()) < N * M) {
    workspace.resize(N * M);
  }
  compute_dtw_cost<scalar_t>(distances_a, N, M, workspace.data());
  const auto path = compute_dtw_path<acc>(workspace.data(), N, M);
  // For integral dtypes divide in int64: casting the path length to a narrow dtype
  // can yield 0 (e.g. 256 as int8) and trap on integer division by zero.
  if constexpr (std::is_integral_v<scalar_t>) {
    return static_cast<scalar_t>(workspace[N * M - 1] / static_cast<int64_t>(path.size()));
  } else {
    return static_cast<scalar_t>(workspace[N * M - 1] / static_cast<acc>(path.size()));
  }
}

Tensor dtw_cpu(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  Tensor out = torch::stable::new_empty(distances, {});
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "compute_dtw",
      AT_WRAP([&] {
        // Reuse scratch across calls on this thread to avoid malloc/free churn for many small DTWs.
        thread_local std::vector<acc_t<scalar_t>> workspace;
        const scalar_t result = compute_dtw_value<scalar_t>(accessor<const scalar_t, 2>(distances), N, M, workspace);
        torch::stable::fill_(out, result);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

Tensor dtw_path_cpu(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  std::vector<std::pair<int64_t, int64_t>> path;
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "compute_dtw_path",
      AT_WRAP([&] {
        std::vector<acc_t<scalar_t>> cost(N * M);
        compute_dtw_cost<scalar_t>(accessor<const scalar_t, 2>(distances), N, M, cost.data());
        path = compute_dtw_path<acc_t<scalar_t>>(cost.data(), N, M);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  Tensor out = torch::stable::new_empty(distances, {(int64_t)path.size(), 2}, torch::headeronly::ScalarType::Long);
  std::memcpy(
      reinterpret_cast<int64_t*>(out.data_ptr()),
      reinterpret_cast<const int64_t*>(path.data()),
      static_cast<size_t>(path.size() * 2) * sizeof(int64_t));
  return out;
}

template <typename distances_t, typename sx_t>
void dtw_batch_cpu_impl(Tensor& out, const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);

  STD_TORCH_CHECK(
      sy.scalar_type() == torch::headeronly::CppTypeToScalarType<sx_t>::value, "sy dtype does not match sx dtype");
  const auto sx_a = accessor<const sx_t, 1>(sx);
  const auto sy_a = accessor<const sx_t, 1>(sy);
  const auto distances_a = accessor<const distances_t, 4>(distances);
  auto out_a = accessor<distances_t, 2>(out);

  for (int64_t i = 0; i < nx; i++) {
    STD_TORCH_CHECK(static_cast<int64_t>(sx_a[i]) <= distances.size(2), "sx values must not exceed distances.size(2)");
  }
  for (int64_t j = 0; j < ny; j++) {
    STD_TORCH_CHECK(static_cast<int64_t>(sy_a[j]) <= distances.size(3), "sy values must not exceed distances.size(3)");
  }

  torch::stable::parallel_for(0, nx, 1, [&](int64_t start, int64_t end) {
    std::vector<acc_t<distances_t>> workspace;
    for (int64_t i = start; i < end; i++) {
      const int64_t N = static_cast<int64_t>(sx_a[i]);
      const int64_t start_j = symmetric ? i + 1 : 0;
      for (int64_t j = start_j; j < ny; j++) {
        const int64_t M = static_cast<int64_t>(sy_a[j]);
        if (N <= 0 || M <= 0)
          continue;
        out_a[i][j] = compute_dtw_value<distances_t>(distances_a[i][j], N, M, workspace);
        if (symmetric) {
          out_a[j][i] = out_a[i][j];
        }
      }
    }
  });
}

Tensor dtw_batch_cpu(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");
  STD_TORCH_CHECK(sx.dim() == 1 && sy.dim() == 1, "sx and sy must be 1D tensors");
  STD_TORCH_CHECK(
      sx.size(0) == distances.size(0) && sy.size(0) == distances.size(1),
      "sx and sy sizes must match the first two dimensions of distances");
  STD_TORCH_CHECK(
      !symmetric || distances.size(0) == distances.size(1),
      "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  Tensor out = torch::stable::new_zeros(distances, {distances.size(0), distances.size(1)});
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_batch_cpu_impl",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            sx.scalar_type(),
            "dtw_batch_cpu_impl_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (dtw_batch_cpu_impl<distances_t, sx_t>(out, distances, sx, sy, symmetric));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

STABLE_TORCH_LIBRARY(torchdtw, m) {
  m.def("dtw(Tensor distances) -> Tensor");
  m.def("dtw_path(Tensor distances) -> Tensor");
  m.def("dtw_batch(Tensor distances, Tensor sx, Tensor sy, bool symmetric) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CPU, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cpu));
  m.impl("dtw_path", &TORCH_BOX(dtw_path_cpu));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_cpu));
}

} // namespace torchdtw
