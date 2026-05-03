#include <Python.h>
#include <algorithm>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>
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

template <typename scalar_t> static Tensor compute_dtw_cost(const Tensor& distances) {
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  Tensor cost = torch::stable::empty_like(distances);
  auto cost_a = accessor<scalar_t, 2>(cost);
  const auto distances_a = accessor<const scalar_t, 2>(distances);

  cost_a[0][0] = distances_a[0][0];
  for (int64_t i = 1; i < N; i++) {
    cost_a[i][0] = distances_a[i][0] + cost_a[i - 1][0];
  }
  for (int64_t j = 1; j < M; j++) {
    cost_a[0][j] = distances_a[0][j] + cost_a[0][j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    for (int64_t j = 1; j < M; j++) {
      cost_a[i][j] = distances_a[i][j] + std::min({cost_a[i - 1][j], cost_a[i - 1][j - 1], cost_a[i][j - 1]});
    }
  }
  return cost;
}

template <typename scalar_t> static std::vector<std::pair<int64_t, int64_t>> compute_dtw_path(const Tensor& cost) {
  const int64_t N = cost.size(0);
  const int64_t M = cost.size(1);
  const auto cost_a = accessor<const scalar_t, 2>(cost);
  std::vector<std::pair<int64_t, int64_t>> path;
  int64_t i = N - 1;
  int64_t j = M - 1;
  path.push_back({i, j});
  while (i > 0 && j > 0) {
    const scalar_t c_up = cost_a[i - 1][j];
    const scalar_t c_left = cost_a[i][j - 1];
    const scalar_t c_diag = cost_a[i - 1][j - 1];
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

template <typename scalar_t> static scalar_t compute_dtw(const Tensor& distances) {
  Tensor cost = compute_dtw_cost<scalar_t>(distances);
  const auto path = compute_dtw_path<scalar_t>(cost);
  const auto cost_a = accessor<const scalar_t, 2>(cost);
  return cost_a[cost.size(0) - 1][cost.size(1) - 1] / static_cast<scalar_t>(path.size());
}

Tensor dtw_cpu(const Tensor& distances) {
  Tensor out = torch::stable::new_empty(distances, {});
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "compute_dtw",
      AT_WRAP([&] {
        const scalar_t result = compute_dtw<scalar_t>(distances);
        torch::stable::fill_(out, result);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

Tensor dtw_path_cpu(const Tensor& distances) {
  Tensor cost;
  std::vector<std::pair<int64_t, int64_t>> path;
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "compute_dtw_path",
      AT_WRAP([&] {
        cost = compute_dtw_cost<scalar_t>(distances);
        path = compute_dtw_path<scalar_t>(cost);
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
  const auto sx_a = accessor<sx_t, 1>(sx);
  const auto sy_a = accessor<sx_t, 1>(sy);
  auto out_a = accessor<distances_t, 2>(out);

  torch::stable::parallel_for(0, nx, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      const int64_t start_j = symmetric ? i : 0;
      for (int64_t j = start_j; j < ny; j++) {
        if (symmetric && i == j)
          continue;
        auto t1 = torch::stable::select(distances, 0, i);
        auto t2 = torch::stable::select(t1, 0, j);
        auto t3 = torch::stable::narrow(t2, 0, 0, sx_a[i]);
        auto sub_distances = torch::stable::narrow(t3, 1, 0, sy_a[j]);
        out_a[i][j] = compute_dtw<distances_t>(sub_distances);
        if (symmetric && i != j) {
          out_a[j][i] = out_a[i][j];
        }
      }
    }
  });
}

Tensor dtw_batch_cpu(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
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
