#include <Python.h>
#include <algorithm>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
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

const std::vector<float> dtw_cost(const float* distances, const int64_t N, const int64_t M, const int64_t stride_x,
                                  const int64_t stride_y) {
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  STD_TORCH_CHECK(stride_x > 0 && stride_y > 0, "Strides must be positive");
  std::vector<float> cost(N * M);

  cost[0] = distances[0];
  for (int64_t i = 1; i < N; i++) {
    cost[i * M] = distances[i * stride_x] + cost[(i - 1) * M];
  }
  for (int64_t j = 1; j < M; j++) {
    cost[j] = distances[j * stride_y] + cost[j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    for (int64_t j = 1; j < M; j++) {
      cost[i * M + j] = distances[i * stride_x + j * stride_y] +
                        std::min({cost[(i - 1) * M + j], cost[(i - 1) * M + j - 1], cost[i * M + j - 1]});
    }
  }
  return cost;
}

const std::vector<std::pair<int64_t, int64_t>> dtw_backtrack(const std::vector<float>& cost, const int64_t N,
                                                             const int64_t M) {
  std::vector<std::pair<int64_t, int64_t>> path;
  int64_t i = N - 1;
  int64_t j = M - 1;
  path.push_back({i, j});
  while (i > 0 && j > 0) {
    const float c_up = cost[(i - 1) * M + j];
    const float c_left = cost[i * M + j - 1];
    const float c_diag = cost[(i - 1) * M + j - 1];
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

float dtw(const float* distances, const int64_t N, const int64_t M, const int64_t stride_x, const int64_t stride_y) {
  const std::vector<float> cost = dtw_cost(distances, N, M, stride_x, stride_y);
  return cost.back() / dtw_backtrack(cost, N, M).size();
}

Tensor dtw_cpu(const Tensor distances) {
  float result = dtw(reinterpret_cast<const float*>(distances.data_ptr()), distances.size(0), distances.size(1),
                     distances.stride(0), distances.stride(1));
  Tensor out = torch::stable::new_empty(distances, {});
  torch::stable::fill_(out, result);
  return out;
}

Tensor dtw_path_cpu(const Tensor distances) {
  const std::vector<float> cost = dtw_cost(reinterpret_cast<const float*>(distances.data_ptr()), distances.size(0),
                                           distances.size(1), distances.stride(0), distances.stride(1));
  const std::vector<std::pair<int64_t, int64_t>> path = dtw_backtrack(cost, distances.size(0), distances.size(1));
  Tensor out = torch::stable::new_empty(distances, {(int64_t)path.size(), 2}, torch::headeronly::ScalarType::Long);
  std::memcpy(reinterpret_cast<int64_t*>(out.data_ptr()), reinterpret_cast<const int64_t*>(path.data()),
              static_cast<size_t>(path.size() * 2) * sizeof(int64_t));
  return out;
}

Tensor dtw_batch_cpu(const Tensor distances, const Tensor sx, const Tensor sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  Tensor out = torch::stable::new_zeros(distances, {nx, ny});

  const float* distances_ptr = reinterpret_cast<const float*>(distances.data_ptr());
  const int64_t* sx_ptr = reinterpret_cast<const int64_t*>(sx.data_ptr());
  const int64_t* sy_ptr = reinterpret_cast<const int64_t*>(sy.data_ptr());
  float* out_ptr = reinterpret_cast<float*>(out.data_ptr());

  torch::stable::parallel_for(0, nx, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      const int64_t start_j = symmetric ? i : 0;
      for (int64_t j = start_j; j < ny; j++) {
        if (symmetric && i == j)
          continue;
        out_ptr[i * ny + j] = dtw(distances_ptr + i * distances.stride(0) + j * distances.stride(1), sx_ptr[i],
                                  sy_ptr[j], distances.stride(2), distances.stride(3));
        if (symmetric && i != j) {
          out_ptr[j * ny + i] = out_ptr[i * ny + j];
        }
      }
    }
  });
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
